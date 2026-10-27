"""Sensor-only EvTTC population derived from the author correspondence.

This is an explicit reconstruction, not the authors' unavailable query list.
No TTC file is read by population selection or preprocessing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.data.garl_official_preprocessing import official_resize_roi, official_square_box
from e_jepa_ttc.efficient_context.garl_input import native_feature
from operational.evttc_rgb_transfer.inputs import _normalized_rgb
from operational.evttc_transfer.inputs import (
    EvTTCEventReader,
    _calibration,
    _calibration_sha256,
    _canonical_sha256,
    _single_object_bbox,
    project_rgb_box_rotation_only,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "e-jepa-ttc/datasets/evttc"
CASES = {
    "CCRs2-medium": ("CCRs-2/medium-100/overlap-100", 9.4, 11.6),
    "CCRs2-high": ("CCRs-2/high-100/overlap-100", 15.5, 17.4),
    "CCRm-medium-overlap0": ("CCRm/medium-0/overlap-0", 21.4, 22.2),
    "CCRm-medium-overlap50": ("CCRm/medium-50/overlap-50", 21.4, 22.2),
    "CCRm-medium-overlap100": ("CCRm/medium-100/overlap-100", 21.4, 22.2),
}


def sha(path: Path) -> str:
    """Hash a bounded stream."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    """Write deterministic UTF-8 LF JSON to this study's own destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def nearest(clock: np.ndarray, target: int) -> int:
    """Nearest timestamp, with an earlier-index tie break."""
    insertion = int(np.searchsorted(clock, target))
    candidates = sorted({max(0, insertion - 1), min(len(clock) - 1, insertion)})
    return min(candidates, key=lambda index: (abs(int(clock[index]) - target), index))


def population() -> dict[str, Any]:
    """Freeze grid, nearest-center tracking, sensor support and all exclusions."""
    rows, excluded, inventory = [], [], {}
    for case, (relative, start, end) in CASES.items():
        folder = DATA / relative
        raw_files = sorted(folder.glob("20*.hdf5"))
        if len(raw_files) != 1:
            raise ValueError(f"Ambiguous raw recording: {relative}")
        raw = raw_files[0]
        stat = raw.stat()
        with h5py.File(raw, "r") as handle:
            clock = np.asarray(handle["blackflys/left/ts"], np.int64)
            calibration = _calibration(handle)
        selected = []
        previous_center = None
        for grid in np.arange(round(start * 1e6), round(end * 1e6) + 1, 100_000):
            index = nearest(clock, int(grid))
            path = folder / "leftlabel" / f"{index:04d}.json"
            failure = None
            if abs(int(clock[index]) - grid) > 1000:
                failure = "NO_RGB_FRAME_WITHIN_1MS_OF_REGISTERED_GRID"
            elif not path.exists():
                failure = "MISSING_ANNOTATION"
            if failure:
                selected.append(None)
                excluded.append(
                    {"case": case, "grid_us": int(grid), "stage": "frame", "reason": failure}
                )
                continue
            objects = read(path).get("objects", [])
            candidates = [
                (i, item, _single_object_bbox(item))
                for i, item in enumerate(objects)
                if str(item.get("category", "")).lower() == "car"
            ]
            candidates = [
                (item[0], item[1], cast(tuple[float, float, float, float], item[2]))
                for item in candidates
                if item[2] is not None
            ]
            if not candidates:
                selected.append(None)
                excluded.append(
                    {
                        "case": case,
                        "grid_us": int(grid),
                        "stage": "frame",
                        "reason": "NO_VALID_CAR_BOX",
                    }
                )
                continue
            reference = np.array([960, 600]) if previous_center is None else previous_center
            choice = min(
                candidates,
                key=lambda x: (
                    float(
                        np.square((np.array(x[2][:2]) + np.array(x[2][2:])) / 2 - reference).sum()
                    ),
                    x[0],
                ),
            )
            box = tuple(float(x) for x in choice[2])
            previous_center = (np.array(box[:2]) + np.array(box[2:])) / 2
            event_box = project_rgb_box_rotation_only(box, calibration)
            selected.append(
                {
                    "index": index,
                    "timestamp_us": int(clock[index]),
                    "rgb_box": box,
                    "event_box": event_box,
                    "source": {
                        "path": str(path.resolve()),
                        "sha256": sha(path),
                        "frame_index": index,
                        "timestamp_us": int(clock[index]),
                        "available_us": int(clock[index]),
                        "category": choice[1]["category"],
                        "group": choice[1].get("group"),
                        "object_index": choice[0],
                    },
                    "multiple_car_objects": len(candidates) > 1,
                }
            )
        count_before = len(rows)
        for index in range(2, len(selected)):
            triple = selected[index - 2 : index + 1]
            grid_anchor = round(start * 1e6) + index * 100_000
            if any(value is None for value in triple):
                excluded.append(
                    {
                        "case": case,
                        "grid_us": grid_anchor,
                        "stage": "query",
                        "reason": "INCOMPLETE_TRIPLET",
                    }
                )
                continue
            anchor = triple[-1]["timestamp_us"]
            windows = [
                [anchor + offset - 100_000, anchor + offset] for offset in (-200_000, -100_000, 0)
            ]
            boxes = [value["event_box"] for value in triple]
            row = {
                "query_id": f"{case}:{anchor}",
                "sequence_id": case,
                "anchor_us": anchor,
                "raw_path": str(raw.resolve()),
                "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                "windows_us": windows,
                "boxes_xyxy3": boxes,
                "square_xyxy": common_square_from_boxes(boxes, (1, 2), margin_fraction=0.25),
                "bbox_sources": [value["source"] for value in triple],
                "rgb_boxes": [value["rgb_box"] for value in triple],
                "rgb_frame_indices": [value["index"] for value in triple],
                "rgb_timestamps_us": [value["timestamp_us"] for value in triple],
                "calibration_sha256": _calibration_sha256(calibration),
                "multiple_car_objects": any(value["multiple_car_objects"] for value in triple),
            }
            row["metadata_sha256"] = _canonical_sha256(row)
            rows.append(row)
        inventory[case] = {
            "relative_directory": relative,
            "interval_seconds": [start, end],
            "raw_path": str(raw.resolve()),
            "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "sensor_range_us": [int(clock[0]), int(clock[-1])],
            "grid_frames": len(selected),
            "complete_queries": len(rows) - count_before,
        }
    return {
        "rows": rows,
        "excluded": excluded,
        "inventory": inventory,
        "targets_read": False,
        "optimizer_updates": 0,
    }


def garl_inputs(row: dict[str, Any]) -> dict[str, Any]:
    """Native Garl RGB normalization and factor-one crops on actual RGB segments."""
    raw = Path(row["raw_path"])
    stat = raw.stat()
    if row["raw_stat"] != {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
        raise ValueError("Raw recording changed")
    for source in row["bbox_sources"]:
        if sha(Path(source["path"])) != source["sha256"]:
            raise ValueError("Annotation changed")
    times = row["rgb_timestamps_us"]
    rgb_boxes, event_boxes = row["rgb_boxes"][-2:], row["boxes_xyxy3"][-2:]
    events = []
    with EvTTCEventReader(raw) as reader:
        for index in range(2):
            event = reader.read_window(times[index], times[index + 1])
            if not len(event["t"]):
                return {"status": "EMPTY_EVENT_SEGMENT", "sensor": None}
            events.append(native_feature(event, official_square_box(event_boxes, index), size=128))
    # The released RGB loader uses one shared endpoint-1 square for the two frames.
    rgb_square = official_square_box(rgb_boxes, 1)
    images = []
    with h5py.File(raw, "r") as handle:
        for index in row["rgb_frame_indices"][-2:]:
            dataset = handle["blackflys/left/data"]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError("Missing RGB dataset")
            frame = np.asarray(dataset[index], np.uint8)
            image = official_resize_roi(_normalized_rgb(frame), rgb_square, (128, 128))
            images.append(image)
    sensor = torch.cat([*images, *events]).numpy().astype(np.float32, copy=False)
    if sensor.shape != (46, 128, 128) or not np.isfinite(sensor).all():
        raise ValueError("Garl sensor contract failed")
    return {
        "status": "READY",
        "sensor": sensor,
        "sensor_sha256": hashlib.sha256(sensor.tobytes()).hexdigest(),
        "rgb_square": list(rgb_square),
        "event_delta_t_s": (np.diff(times) / 1e6).tolist(),
    }
