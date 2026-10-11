"""Build auditable input-only registration receipts before TTC inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
from collections import Counter
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.sota_eval.fcwd_inputs import FCWDEventReader, _validate_row
from operational.sota_eval.fcwd_run import validate_manifest

from .geometry import SETTINGS, estimate_shift, project_box


def source_bindings() -> dict[str, str]:
    """Pin the complete additive adapter and the retained frozen input helpers."""
    root = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob("*.py"))
    files += [
        Path(name)
        for name in (
            "operational/sota_eval/fcwd_inputs.py",
            "operational/sota_eval/fcwd_run.py",
            "operational/rgb_port_revision/transfer_inputs.py",
            "src/e_jepa_ttc/rgb_port/data.py",
        )
    ]
    return {
        path.resolve().relative_to(root).as_posix(): sha256_file(path) for path in sorted(files)
    }


def extract_calibration(path: Path) -> dict[str, Any]:
    """Decode actual MATLAB MCOS camera objects, retaining only named intrinsics."""
    from matio import load_from_mat

    objects = load_from_mat(str(path), raw_data=True)
    result: dict[str, Any] = {
        "schema": "fcwd_published_intrinsics_v1",
        "source_sha256": sha256_file(path),
        "source_name": path.name,
        "decoder": "mat-io",
        "decoder_version": importlib.metadata.version("mat-io"),
        "source_url": "https://nail-hnu.github.io/EventAidedTTC/",
        "extrinsics_present": False,
    }
    for name, size in (("APScameraParams", [1200, 1920]), ("DVScameraParams", [720, 1280])):
        value = cast(Any, objects[name]).properties["any"][0, 0]
        result[name] = {
            key: value[key].tolist()
            for key in ("IntrinsicMatrix", "RadialDistortion", "TangentialDistortion", "ImageSize")
        }
        if np.asarray(value["ImageSize"]).ravel().tolist() != size:
            raise ValueError("Published calibration dimensions do not match FCWD sensors")
    return result


def counts(events: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    """Build a bounded count image from right-camera events only."""
    x, y = events["x"][mask], events["y"][mask]
    if np.any((x < 0) | (x >= 1280) | (y < 0) | (y >= 720)):
        raise ValueError("Event coordinate outside FCWD sensor")
    return np.bincount(y * 1280 + x, minlength=720 * 1280).reshape(720, 1280).astype(np.float32)


def register_query(row: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """Use only a frame and event packet available by the query timestamp."""
    path = _validate_row(row)
    query = int(row["anchor_us"])
    with h5py.File(path, "r") as handle:
        clock = np.asarray(handle["blackflys/right/ts"], np.int64)
        if np.any(np.diff(clock) <= 0):
            raise ValueError("Nonmonotonic RGB sensor clock")
        index = int(np.searchsorted(clock, query, side="right")) - 1
        if index < 0 or query - clock[index] > 51_000:
            return {"accepted": False, "reason": "NO_RECENT_CAUSAL_RGB_FRAME"}
        timestamp = int(clock[index])
        data = handle["blackflys/right/data"]
        if not isinstance(data, h5py.Dataset):
            raise ValueError("RGB data is not a dataset")
        rgb = np.asarray(data[index], np.uint8)
    with FCWDEventReader(path) as reader:
        events = reader.read_window(timestamp - SETTINGS["event_window_us"], timestamp)
    early = counts(events, events["t"] < timestamp - SETTINGS["event_window_us"] // 2)
    late = counts(events, events["t"] >= timestamp - SETTINGS["event_window_us"] // 2)
    result = estimate_shift(rgb, early, late, row["boxes_xyxy3"][-1], calibration)
    result.update(
        {
            "query_id": row["query_id"],
            "sequence_id": row["sequence_id"],
            "query_us": query,
            "rgb_frame_index": index,
            "rgb_sensor_us": timestamp,
            "event_interval_us": [timestamp - SETTINGS["event_window_us"], timestamp],
            "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "count_maps_sha256": hashlib.sha256(early.tobytes() + late.tobytes()).hexdigest(),
            "labels_read": False,
            "available_us": query,
        }
    )
    if result["accepted"]:
        shift = tuple(result["shift_px"])
        boxes = [project_box(box, calibration, shift) for box in row["boxes_xyxy3"]]
        square = common_square_from_boxes(boxes, (0, 1, 2), margin_fraction=0.25)
        result["rgb_boxes_xyxy3"] = boxes
        result["rgb_square_xyxy"] = list(square)
        # Complete ROI outside sensor cannot be made meaningful by clipping.
        if square[2] <= 0 or square[3] <= 0 or square[0] >= 1920 or square[1] >= 1200:
            result.update(accepted=False, reason="PROJECTED_ROI_OUTSIDE_RGB")
    return result


def build(
    manifest_path: Path, calibration_path: Path, output: Path, limit: int = 0
) -> dict[str, Any]:
    """Resume geometry preparation with immutable inputs and no target access."""
    manifest = read_json_shared(manifest_path)
    rows = validate_manifest(manifest)
    if limit:
        rows = sorted(rows, key=lambda row: hashlib.sha256(row["query_id"].encode()).hexdigest())[
            :limit
        ]
    calibration = extract_calibration(calibration_path)
    binding = {
        "schema": "fcwd_rgb_registration_v1",
        "manifest_sha256": sha256_file(manifest_path),
        "calibration": calibration,
        "settings": SETTINGS,
        "sources": source_bindings(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "selected_ids": [r["query_id"] for r in rows],
        "scope": "input_only_adaptive_registration_exploratory_not_metric_extrinsics",
        "labels_read": False,
        "optimizer_updates": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "BINDING.json"
    if receipt_path.exists() and read_json_shared(receipt_path) != binding:
        raise ValueError("Registration source/input changed: use a separate experiment directory")
    atomic_write_json(receipt_path, binding)
    records = []
    for index, row in enumerate(rows):
        path = output / "queries" / f"{row['query_id'].replace(':', '_')}.json"
        if path.exists():
            value = read_json_shared(path)
            if value.get("binding_sha256") != sha256_file(receipt_path):
                raise ValueError("Saved registration binding changed")
        else:
            value = register_query(row, calibration)
            value.update(query_id=row["query_id"], binding_sha256=sha256_file(receipt_path))
            atomic_write_json(path, value)
        records.append(value)
        if index % 10 == 0 or index == len(rows) - 1:
            print(
                f"registered {index + 1}/{len(rows)}; "
                f"accepted {sum(v['accepted'] for v in records)}",
                flush=True,
            )
    result = {
        "status": "COMPLETE",
        "population": len(records),
        "accepted": sum(v["accepted"] for v in records),
        "rejected_reasons": dict(Counter(v["reason"] for v in records if not v["accepted"])),
        "by_sequence": {
            seq: {
                "rows": sum(v.get("sequence_id") == seq for v in records),
                "accepted": sum(v.get("sequence_id") == seq and v["accepted"] for v in records),
            }
            for seq in sorted({r["sequence_id"] for r in rows})
        },
        "binding_sha256": sha256_file(receipt_path),
        "query_sha256": {
            p.name: sha256_file(p) for p in sorted((output / "queries").glob("*.json"))
        },
        "labels_read": False,
        "optimizer_updates": 0,
        "fallback": "keep_all_queries; event_prediction_when_RGB_registration_rejected",
        "limitations": [
            "No ground-truth pixel correspondences",
            "NCC is not a geometric error bound",
            "Input-only target-domain adaptation",
            "Query ROI reused retrospectively",
        ],
    }
    atomic_write_json(output / "REGISTRATION_SEALED.json", result)
    return result


def main() -> None:
    """CLI for label-free calibration preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    build(args.manifest, args.calibration, args.output, args.limit)


if __name__ == "__main__":
    main()
