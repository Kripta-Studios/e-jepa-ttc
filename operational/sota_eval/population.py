"""Build the immutable, label-free expanded EvTTC development population.

The builder reuses the frozen causal query contract and requests every feasible
anchor rather than the earlier uniform cap of 32 per sequence.  It reads event,
camera, calibration and bbox metadata only.  TTC tables, GT HDF5 files, the
official sealed benchmark, predictions and model metrics are outside this
module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from e_jepa_ttc.data.evttc import read_manifest
from operational.evttc_transfer.inputs import build_manifest

ALL_FEASIBLE_LIMIT = 1_000_000


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _write_immutable(path: Path, payload: object) -> None:
    data = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable population artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".pending")
    if temporary.exists():
        raise FileExistsError(f"stale pending population artifact: {temporary}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _semantic_key(row: dict[str, Any]) -> str:
    sources = [
        {
            "frame_index": source["frame_index"],
            "timestamp_us": source["timestamp_us"],
            "sha256": source["sha256"],
        }
        for source in row["bbox_sources"]
    ]
    value = {
        "sequence_id": row["sequence_id"],
        "anchor_us": row["anchor_us"],
        "windows_us": row["windows_us"],
        "boxes_xyxy3": row["boxes_xyxy3"],
        "square_xyxy": row["square_xyxy"],
        "bbox_sources": sources,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _horizontal_fov_deg(intrinsics: np.ndarray, resolution: np.ndarray) -> float:
    fx = float(np.asarray(intrinsics)[0])
    width = float(np.asarray(resolution)[0])
    if not math.isfinite(fx) or not math.isfinite(width) or fx <= 0 or width <= 0:
        raise ValueError("positive finite fx and image width required")
    return 2.0 * math.degrees(math.atan(width / (2.0 * fx)))


def _camera_metadata(handle: h5py.File, base: str) -> dict[str, Any]:
    calibration = f"{base}/calib"
    required = {
        "intrinsics": f"{calibration}/intrinsics",
        "distortion_coeffs": f"{calibration}/distortion_coeffs",
        "resolution": f"{calibration}/resolution",
    }
    missing = sorted(path for path in required.values() if path not in handle)
    if missing:
        return {"available": False, "missing": missing}
    intrinsics = np.asarray(handle[required["intrinsics"]], dtype=np.float64)
    distortion = np.asarray(handle[required["distortion_coeffs"]], dtype=np.float64)
    resolution = np.asarray(handle[required["resolution"]], dtype=np.int64)
    result = {
        "available": True,
        "intrinsics": intrinsics.tolist(),
        "distortion_coeffs": distortion.tolist(),
        "resolution": resolution.tolist(),
        "horizontal_fov_deg_pinhole": _horizontal_fov_deg(intrinsics, resolution),
    }
    if base.startswith("prophesee/"):
        transforms = [
            f"{calibration}/{name}"
            for name in ("T_bfs_to_prophesee", "T_to_left_bfs")
            if f"{calibration}/{name}" in handle
        ]
        result["cross_camera_transform_datasets"] = transforms
    return result


def _audit_calibration(manifest: dict[str, Any], inventory_path: Path) -> dict[str, Any]:
    inventory = {sequence.sequence_id: sequence for sequence in read_manifest(inventory_path)}
    first_by_sequence: dict[str, dict[str, Any]] = {}
    for row in manifest["rows"]:
        first_by_sequence.setdefault(row["sequence_id"], row)
    sequences = []
    for sequence_id in sorted(first_by_sequence):
        row = first_by_sequence[sequence_id]
        source = inventory[sequence_id]
        with h5py.File(row["raw_path"], "r") as handle:
            cameras = {
                name: _camera_metadata(handle, base)
                for name, base in {
                    "blackfly_left": "blackflys/left",
                    "blackfly_right": "blackflys/right",
                    "event_left": "prophesee/event_cam_left",
                    "event_right": "prophesee/event_cam_right",
                }.items()
            }
        sequences.append(
            {
                "sequence_id": sequence_id,
                "label_directory_name": source.label_dir,
                "label_camera_contract": "blackflys/left only",
                "cameras": cameras,
            }
        )
    complete = {
        camera: sum(bool(row["cameras"][camera]["available"]) for row in sequences)
        for camera in ("blackfly_left", "blackfly_right", "event_left", "event_right")
    }
    first = sequences[0]["cameras"]
    return {
        "status": "METADATA_ONLY_NO_TARGETS",
        "sequence_count": len(sequences),
        "complete_camera_calibration_counts": complete,
        "sequences": sequences,
        "first_sequence_fov_example_deg": {
            camera: first[camera].get("horizontal_fov_deg_pinhole") for camera in first
        },
        "optics_uncertainty": (
            "The stored pinhole intrinsics imply approximately 54 degrees for event-left and "
            "22 degrees for event-right in the first sequence. A claimed 16 mm lens mapping "
            "cannot be verified from focal length in pixels without authoritative sensor pitch/"
            "active-area metadata; it is recorded but not used to select queries or models."
        ),
        "right_camera_dependency": (
            "Right RGB/event calibration and media exist, but the frozen object labels and ROI "
            "contract are left-camera only. A right-camera arm needs an independently frozen "
            "right ROI correspondence and checkpoint input contract."
        ),
        "fcwd_dependency": (
            "FCWD requires a separately authorized public inventory, target/ROI/timestamp "
            "semantics and ancestry audit. It cannot be added by changing architecture, context "
            "length, or windows in this campaign."
        ),
        "forbidden_reads": ["ttc.csv", "gt.hdf5", "distance", "depth", "sealed benchmark"],
    }


def build_population(
    *,
    inventory_path: Path,
    data_root: Path,
    prior_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build all feasible dev32 queries and freeze their source lineage."""

    inventory_path = inventory_path.resolve()
    data_root = data_root.resolve()
    prior_manifest_path = prior_manifest_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="query_manifest_", suffix=".json", dir=output_dir, delete=False
    ) as handle:
        pending = Path(handle.name)
    try:
        manifest = build_manifest(
            inventory_path,
            data_root,
            pending,
            max_per_sequence=ALL_FEASIBLE_LIMIT,
        )
        manifest.update(
            protocol="evttc_historical_dev32_all_feasible_rotation_only_v1",
            selection_mode="ALL_CAUSALLY_FEASIBLE_NO_TARGETS",
            maximum_per_sequence=ALL_FEASIBLE_LIMIT,
            selected_count=len(manifest["rows"]),
            source_inventory_sha256=_digest(inventory_path),
            prior_population_sha256=_digest(prior_manifest_path),
        )
    finally:
        pending.unlink(missing_ok=True)

    prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    expanded_keys = {_semantic_key(row) for row in manifest["rows"]}
    prior_keys = [_semantic_key(row) for row in prior["rows"]]
    missing = sorted(key for key in prior_keys if key not in expanded_keys)
    snapshot = {
        "status": "PASSED" if not missing else "FAILED",
        "prior_manifest_sha256": _digest(prior_manifest_path),
        "prior_query_count": len(prior_keys),
        "contained_semantic_queries": len(prior_keys) - len(missing),
        "missing_semantic_query_sha256": missing,
        "selection_uses_targets": False,
    }
    if missing:
        raise ValueError("expanded population does not contain every frozen prior query")

    calibration = _audit_calibration(manifest, inventory_path)
    manifest_path = output_dir / "QUERY_MANIFEST.json"
    _write_immutable(manifest_path, manifest)
    _write_immutable(output_dir / "PRIOR_QUERY_SNAPSHOT.json", snapshot)
    _write_immutable(output_dir / "CALIBRATION_AUDIT.json", calibration)
    freeze = {
        "status": "FROZEN_LABEL_FREE",
        "manifest_sha256": _digest(manifest_path),
        "manifest_rows": len(manifest["rows"]),
        "inventory_path": str(inventory_path),
        "inventory_sha256": _digest(inventory_path),
        "data_root": str(data_root),
        "prior_manifest_path": str(prior_manifest_path),
        "prior_manifest_sha256": _digest(prior_manifest_path),
        "population_source_sha256": _digest(Path(__file__)),
        "dependency_source_sha256": _digest(
            Path(__file__).parents[1] / "evttc_transfer" / "inputs.py"
        ),
        "optimizer_updates": 0,
        "targets_read": False,
        "gpu_used": False,
    }
    _write_immutable(output_dir / "POPULATION_FREEZE.json", freeze)
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = build_population(
        inventory_path=arguments.inventory,
        data_root=arguments.data_root,
        prior_manifest_path=arguments.prior_manifest,
        output_dir=arguments.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ALL_FEASIBLE_LIMIT", "build_population"]
