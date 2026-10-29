"""CPU-only proof that an RGB pilot bundle covers PAIR_R and R_CTX inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from operational.rgb_port.accounting import atomic_write_json

from .inventory import digest, read


def _tensor_identity(batch: object) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name, value in vars(batch).items():
        if isinstance(value, torch.Tensor):
            contiguous = value.detach().cpu().contiguous()
            fields[name] = {
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "sha256": hashlib.sha256(contiguous.numpy().tobytes()).hexdigest(),
            }
        elif isinstance(value, (list, tuple)):
            fields[name] = list(value)
    return fields


def verify_rgb_downstream_coverage(
    scientific_config: Path,
    inventory_path: Path,
    *,
    rotations: tuple[int, ...] = (2, 3, 4),
) -> dict[str, Any]:
    """Verify manifests and decode one real CPU sample for every fold role."""
    config_path = scientific_config.resolve(strict=True)
    inventory_file = inventory_path.resolve(strict=True)
    config, inventory = read(config_path), read(inventory_file)
    if (
        config.get("schema") != "scientific_oof_execution_v1"
        or inventory.get("schema") != "ejepa_runpod_inventory_v1"
        or inventory.get("scientific_config_sha256") != digest(config_path)
    ):
        raise ValueError("Scientific config and Runpod inventory are not the same authority")
    records = {str(Path(item["source"]).resolve()): item for item in inventory["files"]}
    required: dict[str, dict[str, Any]] = {}

    def require(path: Path, expected_sha: str | None, kind: str) -> None:
        resolved = path.resolve(strict=True)
        item = records.get(str(resolved))
        if item is None or "rgb_pilot" not in item.get("profiles", []):
            raise ValueError(f"RGB pilot omits required {kind}: {resolved}")
        if expected_sha is not None and item.get("sha256") != expected_sha:
            raise ValueError(f"RGB pilot hash differs for {kind}: {resolved}")
        required[str(resolved)] = {
            "kind": kind,
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
            "hash_status": str(item["hash_status"]),
        }

    require(config_path, digest(config_path), "scientific_config")
    contract = Path(config["fold_contract"])
    require(contract, config["input_sha256"][str(contract)], "fold_contract")
    for path in contract.parent.iterdir():
        if path.suffix in {".json", ".parquet"}:
            require(path, digest(path), "fold_role_manifest_or_rows")
    descriptor_path = Path(config["rgb_source"])
    require(descriptor_path, config["input_sha256"][str(descriptor_path)], "rgb_descriptor")
    descriptor = read(descriptor_path)
    for role in ("P", "H", "V"):
        manifest_path = (descriptor_path.parent / descriptor["role_manifests"][role]).resolve()
        require(manifest_path, digest(manifest_path), f"rgb_{role}_manifest")
        manifest = read(manifest_path)
        require(Path(manifest["rows_path"]), manifest["rows_sha256"], f"rgb_{role}_rows")
        binding_path = (descriptor_path.parent / descriptor["cache_bindings"][role]).resolve()
        require(binding_path, digest(binding_path), f"rgb_{role}_cache_binding")
        binding = read(binding_path)
        cache_manifest = Path(binding["cache_manifest_path"])
        require(
            cache_manifest,
            str(binding["cache_manifest_sha256"]),
            f"rgb_{role}_cache_manifest",
        )
        for shard in read(cache_manifest)["shards"]:
            require(Path(shard["path"]), str(shard["sha256"]), f"rgb_{role}_cache_shard")
    teacher_path = Path(config["teacher"])
    require(teacher_path, config["input_sha256"][str(teacher_path)], "teacher_manifest")
    teacher = read(teacher_path)
    for shard in teacher["ordered_shards"]:
        require(
            Path(teacher["directory"]) / shard["name"],
            str(shard["sha256"]),
            "teacher_shard",
        )

    from operational.final_refit.rgb import load_all_train_source
    from operational.final_refit.scientific_oof_features import role_rows

    base = load_all_train_source(descriptor_path, teacher_path)
    rotation_receipts: dict[str, Any] = {}
    try:
        for rotation in rotations:
            role_receipts: dict[str, Any] = {}
            token_sets: dict[str, set[str]] = {}
            ordinal_sets: dict[str, set[int]] = {}
            for role in ("producer", "refiner", "generation"):
                rows = role_rows(contract, rotation, role)
                tokens = rows.sample_token.astype(str).tolist()
                ordinals = rows.source_ordinal.astype(int).tolist()
                token_sets[role], ordinal_sets[role] = set(tokens), set(ordinals)
                native = ordinals[0]
                observed = base.row(native)
                if str(observed["sample_token"]) != tokens[0] or str(observed["group_id"]) != str(
                    rows.iloc[0].group_id
                ):
                    raise ValueError(f"R{rotation} {role} native RGB lookup differs")
                batch = base.batch([native], "rgb")
                role_receipts[role] = {
                    "rows": len(rows),
                    "groups": int(rows.group_id.astype(str).nunique()),
                    "ordered_sample_token_sha256": hashlib.sha256(
                        json.dumps(tokens, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "sample_native_ordinal": native,
                    "sample_batch": _tensor_identity(batch),
                }
            names = tuple(token_sets)
            if any(
                token_sets[left] & token_sets[right] or ordinal_sets[left] & ordinal_sets[right]
                for index, left in enumerate(names)
                for right in names[index + 1 :]
            ) or set().union(*ordinal_sets.values()) != set(range(base.population_size)):
                raise ValueError(f"R{rotation} roles overlap or omit native TRAIN40 rows")
            rotation_receipts[str(rotation)] = role_receipts
    finally:
        base.close()

    extra_rotation_only = [
        item
        for item in inventory["files"]
        if "rotations" in item.get("profiles", []) and "rgb_pilot" not in item.get("profiles", [])
    ]
    payload = {
        "schema": "ejepa_rgb_downstream_bundle_coverage_v1",
        "status": "PASS",
        "scientific_config_sha256": digest(config_path),
        "inventory_sha256": digest(inventory_file),
        "profile": "rgb_pilot",
        "rotations": rotation_receipts,
        "required_rgb_files": len(required),
        "required_rgb_bytes": sum(item["bytes"] for item in required.values()),
        "required_rgb_manifest_identity_sha256": hashlib.sha256(
            json.dumps(required, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "pair_r": {
            "covered": True,
            "training_role": "producer_pair",
            "physical_inputs": "prepared RGB caches plus exact fold producer rows",
        },
        "r_ctx": {
            "covered": True,
            "training_role": "refiner",
            "generation_role": "generation",
            "physical_inputs": "AllTrainRGBSource histories plus exact fold rows",
            "event_metadata_required": False,
            "targets_are_joined_from_rgb_role_rows_after_raw_feature_generation": True,
        },
        "complete_combined_finish_rotation": {
            "covered_by_rgb_pilot": False,
            "reason": "event PAIR_E/E_CTX inputs are intentionally outside rgb_pilot",
            "additional_rotation_only_files": len(extra_rotation_only),
            "additional_rotation_only_bytes": sum(
                int(item["bytes"]) for item in extra_rotation_only
            ),
            "additional_profiles_required": ["rotations"],
        },
        "gpu_used": False,
        "model_loaded": False,
        "inference_executed": False,
        "optimizer_updates": 0,
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scientific-config", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_rgb_downstream_coverage(args.scientific_config, args.inventory)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
