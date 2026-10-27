"""Prove the exact EVENT downstream data closure without running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from operational.rgb_port.accounting import atomic_write_json

from .inventory import digest, read


def verify_event_downstream_plan(
    scientific_config: Path,
    control_archive: Path,
    inventory_path: Path,
    producer_manifest: Path,
    raw_manifest: Path,
) -> dict[str, Any]:
    """Compare PAIR_E/E_CTX requirements with producer and deferred-raw manifests."""
    config_path = scientific_config.resolve(strict=True)
    archive_path = control_archive.resolve(strict=True)
    inventory_file = inventory_path.resolve(strict=True)
    producer_path = producer_manifest.resolve(strict=True)
    raw_path = raw_manifest.resolve(strict=True)
    with tarfile.open(archive_path, "r") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
    config, inventory, producer, raw = (
        read(config_path),
        read(inventory_file),
        read(producer_path),
        read(raw_path),
    )
    if (
        config.get("schema") != "scientific_oof_execution_v1"
        or inventory.get("schema") != "ejepa_runpod_inventory_v1"
        or producer.get("schema") != "ejepa_event_producer_data_manifest_v1"
        or raw.get("schema") != "ejepa_event_raw_deferred_manifest_v1"
        or inventory.get("scientific_config_sha256") != digest(config_path)
        or raw.get("file_count") != 41
    ):
        raise ValueError("EVENT bundle, raw manifest and scientific authority differ")
    included = {
        str(Path(item["source"]).resolve()): item
        for item in inventory["files"]
        if str(item["relative"]).replace("\\", "/") in members
    }
    included.update(
        {str(Path(item["source"]).resolve()): item for item in producer["files"]}
    )
    deferred = {str(Path(item["source"]).resolve()): item for item in raw["files"]}
    required: dict[str, dict[str, Any]] = {}

    def require(
        path: Path, kind: str, expected: str | None = None, *, hash_now: bool = True
    ) -> None:
        resolved = path.resolve(strict=True)
        actual = digest(resolved) if hash_now else expected
        if expected is not None and actual != expected:
            raise ValueError(f"Changed {kind}: {resolved}")
        required[str(resolved)] = {
            "kind": kind,
            "bytes": resolved.stat().st_size,
            "sha256": actual,
            "hash_verified_now": hash_now,
            "in_event_bundle": str(resolved) in included,
            "in_raw_deferred": str(resolved) in deferred,
        }

    contract = Path(config["fold_contract"])
    require(contract, "fold_contract", config["input_sha256"][str(contract)])
    for path in contract.parent.iterdir():
        if path.suffix in {".json", ".parquet"}:
            require(path, "fold_role_manifest_or_rows")
    descriptor_path = Path(config["rgb_source"])
    require(descriptor_path, "target_metadata_descriptor")
    descriptor = read(descriptor_path)
    for role in ("P", "H", "V"):
        manifest_path = (descriptor_path.parent / descriptor["role_manifests"][role]).resolve()
        require(manifest_path, f"target_metadata_{role}_manifest")
        role_manifest = read(manifest_path)
        require(Path(role_manifest["rows_path"]), f"target_metadata_{role}_rows")
        binding_path = (descriptor_path.parent / descriptor["cache_bindings"][role]).resolve()
        require(binding_path, f"metadata_loader_{role}_cache_binding")
        binding = read(binding_path)
        require(Path(binding["cache_manifest_path"]), f"metadata_loader_{role}_cache_manifest")
    teacher_path = Path(config["teacher"])
    require(teacher_path, "metadata_loader_teacher_manifest")
    teacher = read(teacher_path)
    for shard in teacher["ordered_shards"]:
        require(
            Path(teacher["directory"]) / shard["name"],
            "metadata_loader_teacher_shard",
            shard["sha256"],
            hash_now=False,
        )
    v12 = Path(config["v12_run"])
    h8_manifest = read(v12 / "H8_FEATURE_MANIFEST.json")
    require(
        v12 / "H8_FEATURES.npz",
        "event_history_H8_FEATURES.npz",
        h8_manifest["files"][0]["sha256"],
        hash_now=False,
    )
    for name in (
        "H8_FEATURE_MANIFEST.json",
        "TRAIN40_INDEX.npz",
        "DATA_AUDIT.json",
        "RAW_PLAN.json",
        "PREPARE_FREEZE.json",
    ):
        require(v12 / name, f"event_history_{name}")
    for item in raw["files"]:
        require(
            Path(item["source"]),
            "raw_event_or_train_metadata",
            item["sha256"],
            hash_now=False,
        )
    missing = [
        path
        for path, item in required.items()
        if not (item["in_event_bundle"] or item["in_raw_deferred"])
    ]
    payload = {
        "schema": "ejepa_event_downstream_coverage_plan_v1",
        "status": "PASS" if not missing else "MISSING_REQUIRED_METADATA",
        "scientific_config_sha256": digest(config_path),
        "event_control_archive_sha256": digest(archive_path),
        "source_inventory_sha256": digest(inventory_file),
        "event_producer_manifest_sha256": digest(producer_path),
        "raw_deferred_manifest_sha256": digest(raw_path),
        "raw_content_identity_sha256": raw["content_identity_sha256"],
        "raw_files": raw["file_count"],
        "raw_bytes": raw["total_bytes"],
        "raw_bytes_rehashed_by_plan": False,
        "raw_execution_requires_separate_full_hash_receipt": True,
        "required_files": len(required),
        "required_bytes": sum(item["bytes"] for item in required.values()),
        "missing_files": missing,
        "missing_required": {path: required[path] for path in missing},
        "missing_bytes": sum(required[path]["bytes"] for path in missing),
        "prepared_rgb_pixel_shards_required": False,
        "rgb_model_inference_required": False,
        "rgb_role_rows_required_for_target_metadata": True,
        "rgb_cache_bindings_and_manifests_required_by_loader": True,
        "teacher_shards_required_by_strict_loader_initialization": True,
        "gpu_used": False,
        "model_loaded": False,
        "optimizer_updates": 0,
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scientific-config", type=Path, required=True)
    parser.add_argument("--control-archive", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--producer-manifest", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_event_downstream_plan(
        args.scientific_config,
        args.control_archive,
        args.inventory,
        args.producer_manifest,
        args.raw_manifest,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
