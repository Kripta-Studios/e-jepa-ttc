"""Reuse sealed event-only predictions in an expanded, label-free population.

The operation is deliberately stricter than a filename copy.  It verifies the
complete source run, the current frozen model/source bytes, and exact query
metadata before publishing new fragments under the expanded manifest binding.
It never imports PyTorch, opens event media, or reads evaluation targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from operational.efficient_context.common import Lease, atomic_json, digest


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _canonical_sha256(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _validate_row(row: dict[str, Any], where: str) -> tuple[str, tuple[str, int]]:
    """Validate the row's self-hash and return its query-independent identity."""
    required = {"query_id", "sequence_id", "anchor_us", "metadata_sha256"}
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"{where}: missing row fields {missing}")
    hashed = dict(row)
    expected = str(hashed.pop("metadata_sha256"))
    if _canonical_sha256(hashed) != expected:
        raise ValueError(f"{where}: row metadata hash mismatch")
    # metadata_sha256 changes only because query_id is part of its input.  It is
    # independently checked above, then both the ID and its derivative are
    # removed for the exact logical-row comparison.
    hashed.pop("query_id")
    return _canonical_sha256(hashed), (str(row["sequence_id"]), int(row["anchor_us"]))


def _index_rows(rows: list[dict[str, Any]], where: str) -> dict[str, tuple[int, dict[str, Any]]]:
    result: dict[str, tuple[int, dict[str, Any]]] = {}
    identities: dict[tuple[str, int], str] = {}
    query_ids: set[str] = set()
    for index, row in enumerate(rows):
        row_hash, identity = _validate_row(row, f"{where}[{index}]")
        query_id = str(row["query_id"])
        if query_id in query_ids:
            raise ValueError(f"{where}: duplicate query_id {query_id}")
        query_ids.add(query_id)
        prior = identities.get(identity)
        if prior is not None and prior != row_hash:
            raise ValueError(f"{where}: identity collision with different metadata: {identity}")
        if prior is not None or row_hash in result:
            raise ValueError(f"{where}: duplicate logical query: {identity}")
        identities[identity] = row_hash
        result[row_hash] = (index, row)
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode())
    hasher.update(np.asarray(array.shape, dtype="<i8").tobytes())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def _verify_endpoint(campaign: Path, name: str, updates: int, expected: dict[str, Any]) -> None:
    directory = campaign / "fits" / name
    receipt = _read(directory / "CHECKPOINT_RECEIPT.json")
    checkpoint = directory / "checkpoint_last.pt"
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("committed_updates") != updates
        or digest(checkpoint) != receipt.get("sha256")
        or receipt.get("sha256") != expected.get("checkpoint_sha256")
        or expected.get("committed_updates") != updates
    ):
        raise ValueError(f"current checkpoint binding changed: {name}")


def _verify_model_binding(models: dict[str, Any]) -> None:
    """Verify the binding without loading model state or reserving a GPU."""
    campaign = Path(models["campaign"]).resolve()
    model_source = Path(__file__).parents[1] / "evttc_transfer" / "models.py"
    if digest(model_source) != models["module_sha256"]:
        raise ValueError("frozen inference model adapter changed")
    protocol = campaign / "TRAINING_PROTOCOL.json"
    if digest(protocol) != models["producers"]["protocol_sha256"]:
        raise ValueError("producer protocol changed")
    for arm, updates in (("a5", 49_932), ("c2f", 49_932), ("pair", 6_840)):
        _verify_endpoint(campaign, f"{arm}_seed7", updates, models["producers"][arm])
    for seed in (7, 13, 23):
        _verify_endpoint(campaign, f"h8_seed{seed}", 2_500, models["h8_heads"][str(seed)])

    normalization = models["normalization"]
    feature_manifest_path = campaign / "H8_FEATURE_MANIFEST.json"
    feature_path = campaign / "H8_FEATURES.npz"
    feature_manifest = _read(feature_manifest_path)
    if (
        feature_manifest.get("status") != "COMPLETE_VERIFIED"
        or feature_manifest.get("row_count") != 88_744
        or digest(feature_manifest_path) != normalization["manifest_sha256"]
        or digest(feature_path) != normalization["artifact_sha256"]
        or feature_manifest.get("consumed_ids_sha256") != normalization["consumed_ids_sha256"]
    ):
        raise ValueError("H8 normalization binding changed")
    with np.load(feature_path, allow_pickle=False) as stored:
        mean = np.asarray(stored["mean"], dtype=np.float64)
        scale = np.asarray(stored["scale"], dtype=np.float64)
    if (
        _array_sha256(mean) != normalization["mean_sha256"]
        or _array_sha256(scale) != normalization["scale_sha256"]
    ):
        raise ValueError("H8 normalization arrays changed")

    garl = models["garl"]
    delivery = _read(campaign / "DELIVERY_FREEZE.json")
    checkpoint = campaign / "public_garl" / "paper_event_only_lhr.pth"
    config = campaign / "public_garl" / "configs" / "ablation" / "event_lhr.yaml"
    if (
        digest(checkpoint) != garl["checkpoint_sha256"]
        or digest(config) != garl["config_sha256"]
        or delivery.get("public_checkpoint_sha256") != garl["checkpoint_sha256"]
        or delivery.get("config_sha256") != garl["config_sha256"]
    ):
        raise ValueError("published Garl model binding changed")
    native_root = Path(garl["native_code_root"]).resolve()
    if native_root != Path(delivery["native_code_root"]).resolve():
        raise ValueError("published Garl source root changed")
    current_sources = {}
    for item in delivery["native_source_files"]:
        path = native_root / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError(f"published Garl source changed: {item['path']}")
        current_sources[item["path"]] = item["sha256"]
    if current_sources != garl["source_sha256"]:
        raise ValueError("published Garl source set changed")


def _verify_source_run(
    old_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest_path = old_root / "QUERY_MANIFEST.json"
    freeze_path = old_root / "INFERENCE_FREEZE.json"
    seal = _read(old_root / "PREDICTIONS_SEALED.json")
    freeze = _read(freeze_path)
    manifest = _read(manifest_path)
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest has no rows")
    if (
        seal.get("status") != "COMPLETE"
        or seal.get("queries") != len(rows)
        or seal.get("manifest_sha256") != digest(manifest_path)
        or freeze.get("manifest_sha256") != digest(manifest_path)
        or seal.get("binding_sha256") != digest(freeze_path)
        or seal.get("optimizer_updates") != 0
    ):
        raise ValueError("source seal/freeze/manifest lineage mismatch")

    source_dir = Path(__file__).parents[1] / "evttc_transfer"
    actual_sources = {str(path): digest(path) for path in sorted(source_dir.glob("*.py"))}
    if actual_sources != freeze.get("sources"):
        raise ValueError("event-only inference source set or bytes changed")
    _verify_model_binding(freeze["models"])

    fragments = seal.get("fragments")
    if not isinstance(fragments, dict) or len(fragments) != len(rows):
        raise ValueError("source seal fragment inventory is incomplete")
    for index, row in enumerate(rows):
        name = f"predictions/query_{index:05d}.json"
        if name not in fragments:
            raise ValueError(f"source seal missing {name}")
        path = old_root / name
        sidecar = path.with_suffix(".sha256")
        actual = digest(path)
        if fragments[name] != actual or sidecar.read_text(encoding="ascii").strip() != actual:
            raise ValueError(f"source fragment checksum mismatch: {name}")
        saved = _read(path)
        if (
            saved.get("binding_sha256") != seal["binding_sha256"]
            or saved.get("query_id") != row["query_id"]
            or saved.get("sequence_id") != row["sequence_id"]
            or saved.get("anchor_us") != row["anchor_us"]
            or saved.get("optimizer_updates") != 0
        ):
            raise ValueError(f"source fragment identity mismatch: {name}")
    return freeze, seal, rows


def _write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"immutable reuse output differs: {path}")
        return
    atomic_json(path, value)


def reuse_predictions(old_root: Path, expanded_root: Path) -> dict[str, Any]:
    """Publish verified source predictions under the expanded manifest binding."""
    old_root = old_root.resolve()
    expanded_root = expanded_root.resolve()
    if old_root == expanded_root or old_root in expanded_root.parents:
        raise ValueError("expanded outputs must be separate from the historical source run")
    new_manifest_path = expanded_root / "QUERY_MANIFEST.json"
    new_manifest = _read(new_manifest_path)
    new_rows = new_manifest.get("rows")
    if not isinstance(new_rows, list) or not new_rows:
        raise ValueError("expanded manifest has no rows")

    old_freeze, old_seal, old_rows = _verify_source_run(old_root)
    old_index = _index_rows(old_rows, "source rows")
    new_index = _index_rows(new_rows, "expanded rows")
    missing = sorted(set(old_index) - set(new_index))
    if missing:
        raise ValueError(f"expanded manifest omits {len(missing)} sealed source queries")

    new_binding = dict(old_freeze)
    new_binding["manifest_sha256"] = digest(new_manifest_path)
    binding_path = expanded_root / "INFERENCE_FREEZE.json"
    _write_immutable_json(binding_path, new_binding)
    new_binding_sha = digest(binding_path)
    mappings: list[dict[str, Any]] = []

    with Lease(expanded_root):
        for row_hash, (old_position, old_row) in old_index.items():
            new_position, new_row = new_index[row_hash]
            old_name = f"predictions/query_{old_position:05d}.json"
            old_path = old_root / old_name
            source_sha = digest(old_path)
            source = _read(old_path)
            reused = dict(source)
            reused.update(
                status="REUSED",
                query_id=new_row["query_id"],
                sequence_id=new_row["sequence_id"],
                anchor_us=new_row["anchor_us"],
                binding_sha256=new_binding_sha,
                reuse_provenance={
                    "source_query_id": old_row["query_id"],
                    "source_fragment": old_name,
                    "source_fragment_sha256": source_sha,
                    "source_binding_sha256": old_seal["binding_sha256"],
                    "source_finished_utc": source.get("finished_utc"),
                    "new_measurement": False,
                },
            )
            new_name = f"predictions/query_{new_position:05d}.json"
            new_path = expanded_root / new_name
            sidecar = new_path.with_suffix(".sha256")
            if new_path.exists() != sidecar.exists():
                raise ValueError(f"partial existing reuse output: {new_name}")
            _write_immutable_json(new_path, reused)
            new_sha = digest(new_path)
            if sidecar.exists():
                if sidecar.read_text(encoding="ascii").strip() != new_sha:
                    raise ValueError(f"existing reuse sidecar differs: {new_name}")
            else:
                sidecar.write_text(new_sha + "\n", encoding="ascii")
            mappings.append(
                {
                    "source_index": old_position,
                    "expanded_index": new_position,
                    "source_query_id": old_row["query_id"],
                    "expanded_query_id": new_row["query_id"],
                    "sequence_id": new_row["sequence_id"],
                    "anchor_us": new_row["anchor_us"],
                    "logical_row_sha256": row_hash,
                    "source_fragment_sha256": source_sha,
                    "reused_fragment_sha256": new_sha,
                }
            )

        report = {
            "status": "COMPLETE_VERIFIED_REUSE",
            "source_root": str(old_root),
            "expanded_root": str(expanded_root),
            "source_manifest_sha256": old_seal["manifest_sha256"],
            "expanded_manifest_sha256": digest(new_manifest_path),
            "source_binding_sha256": old_seal["binding_sha256"],
            "expanded_binding_sha256": new_binding_sha,
            "source_seal_sha256": digest(old_root / "PREDICTIONS_SEALED.json"),
            "reused_queries": len(mappings),
            "expanded_queries": len(new_rows),
            "remaining_queries": len(new_rows) - len(mappings),
            "new_measurements": 0,
            "optimizer_updates": 0,
            "gpu_used": False,
            "targets_read": False,
            "mapping": sorted(mappings, key=lambda item: item["expanded_index"]),
        }
        _write_immutable_json(expanded_root / "REUSE.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    args = parser.parse_args()
    result = reuse_predictions(args.source, args.expanded)
    print(json.dumps({key: value for key, value in result.items() if key != "mapping"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
