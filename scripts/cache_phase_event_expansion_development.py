"""Cache label-free causal event windows for the 77 eligible Slider-500 queries.

Uses the existing audited SensorSequence reader and event-plane ROI adapter.
The fixed policy is a trailing inclusive 100 ms window, 1.2x current-box padding,
and at most 20,000 events selected uniformly by rank in chronological order.
No other sequence is admitted; no targets, scoring, fitting or prediction occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from operational.evttc_competition.geometry import (
    input_bbox_to_event_box,
    load_calibration,
    normalize_box_xyxy,
)
from operational.evttc_competition.readers import SensorSequence

WINDOW_US = 100_000
MAX_EVENTS = 20_000
ROI_PADDING = 1.2


def trailing_window(query_us: int, event_start_us: int, event_end_us: int) -> tuple[int, int]:
    """Return clipped half-open reader bounds for inclusive [query-100ms,query]."""
    if event_end_us < event_start_us:
        raise ValueError("Sensor coverage end precedes its start")
    start = max(query_us - WINDOW_US, event_start_us)
    end = min(query_us + 1, event_end_us)
    return start, max(start, end)


def pad_event_roi(
    box: tuple[float, float, float, float], resolution: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Expand the already adapted current box by fixed 1.2x width and height."""
    raw = np.asarray(box, dtype=np.float64)
    center = (raw[:2] + raw[2:]) * 0.5
    half_extent = (raw[2:] - raw[:2]) * ROI_PADDING * 0.5
    padded = np.concatenate((center - half_extent, center + half_extent))
    return normalize_box_xyxy(padded.tolist(), resolution, clamp=True)


def uniform_indices(count: int, budget: int) -> np.ndarray:
    """Select reproducible unique chronological ranks, including both endpoints."""
    if count < 0 or budget < 1:
        raise ValueError("Count must be nonnegative and budget positive")
    if count <= budget:
        return np.arange(count, dtype=np.int64)
    if budget == 1:
        return np.array([count // 2], dtype=np.int64)
    return np.arange(budget, dtype=np.int64) * (count - 1) // (budget - 1)


def select_roi_events(
    events: np.ndarray,
    start_us: int,
    end_us: int,
    query_us: int,
    roi: tuple[float, float, float, float],
    budget: int = MAX_EVENTS,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Validate causality, crop the fixed half-open ROI and retain event values."""
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4 or events.dtype.kind not in "iu":
        raise ValueError("Expected integer [N,4] x,y,t_us,p events")
    if len(events):
        clocks = events[:, 2]
        if np.any(np.diff(clocks.astype(np.int64)) < 0):
            raise ValueError("Reader events must be chronological")
        if np.any(clocks > query_us):
            raise ValueError("Future events violate the causal query boundary")
        if np.any(clocks < start_us) or np.any(clocks >= end_us):
            raise ValueError("Reader events violate the requested window")
        if not np.isin(events[:, 3], [0, 1]).all():
            raise ValueError("Audited sensor polarity encoding must remain 0/1")
    x0, y0, x1, y1 = roi
    inside = (events[:, 0] >= x0) & (events[:, 0] < x1) & (events[:, 1] >= y0) & (events[:, 1] < y1)
    cropped = events[inside]
    indices = uniform_indices(len(cropped), budget)
    return cropped[indices].copy(), len(cropped), indices


def file_sha(path: Path) -> str:
    """Stream source hashes without loading complete raw recordings into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a cache receipt after writing complete JSON."""
    pending = path.with_suffix(".building.json")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def require_new_cache(output: Path) -> None:
    """Refuse an existing destination before any cache write or source processing."""
    if output.exists():
        raise FileExistsError("Use a new output directory; frozen caches cannot be overwritten")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("output/phase_campaign/Slider500"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/phase_event_expansion/development_cache")
    )
    args = parser.parse_args()
    require_new_cache(args.output)
    source = args.campaign / "inputs/Slider-500"
    previous_freeze = json.loads((args.campaign / "FREEZE.json").read_text())
    previous_sensors = json.loads((args.campaign / "SENSORS.json").read_text())
    if previous_freeze["topic_profile"] != "event_aided_ttc":
        raise ValueError("Slider500 topic profile differs from the frozen contract")
    if previous_sensors["sequence_id"] != "Slider-500":
        raise ValueError("Only eligible Slider500 sensor data may be cached")
    setup_start = time.perf_counter()
    # These are the only raw sensor/sidecar files allowed, never gt_ttc.csv.
    for name in ("slider_500.bag", "bbox.csv", "calibParams.mat"):
        expected = previous_sensors["files"][name]
        path = source / name
        if path.stat().st_size != expected["bytes"] or file_sha(path) != expected["sha256"]:
            raise ValueError(f"Changed audited Slider500 sensor file: {name}")
        source_name = "bounding_box.csv" if name == "bbox.csv" else name
        if expected["sha256"] != previous_freeze["sources"][source_name]:
            raise ValueError("Frozen source and sensor receipts disagree")
    calibration = load_calibration(source, "slider_candidate")
    if calibration.metadata != previous_freeze["geometry"]:
        raise ValueError("Audited calibration metadata changed")
    source_paths = [
        Path(__file__),
        Path("operational/evttc_competition/readers.py"),
        Path("operational/evttc_competition/geometry.py"),
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    with SensorSequence(source.parent, "Slider-500", slider_profile="event_aided_ttc") as reader:
        sensors = reader.source_metadata()
        if sensors != previous_sensors:
            raise ValueError(
                "Current reader's sensor clocks/provenance differ from audited SENSORS"
            )
        if calibration.event_resolution != (reader.event_width, reader.event_height):
            raise ValueError("Calibration and sensor event resolutions differ")
        queries = reader.bbox_rows()
        identity = [{"frame_index": r["frame_index"], "anchor_us": r["anchor_us"]} for r in queries]
        if len(queries) != 77 or identity != previous_freeze["queries"]:
            raise ValueError("Expected the exact frozen Slider500 77-query population")
        freeze = {
            "schema": "phase_event_expansion_development_cache_v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "sequence_id": "Slider-500",
            "queries": identity,
            "window_us": WINDOW_US,
            "max_events": MAX_EVENTS,
            "roi_padding": ROI_PADDING,
            "window_contract": (
                "Inclusive [query-100000,query], clipped to observed coverage; reader half-open"
            ),
            "sampling": (
                "Uniform chronological event ranks floor(k*(N-1)/(budget-1)); no random seed"
            ),
            "roi_contract": (
                "Current supplied bbox only, audited event-plane conversion; "
                "padded 1.2x and clipped"
            ),
            "coordinates": "Original distorted event raster pixels; no resizing or undistortion",
            "polarity_encoding": "Original0/1 unchanged",
            "coverage": {
                "event_start_us": reader.event_start_us,
                "event_end_exclusive_us": reader.event_end_us,
            },
            "sensor_provenance": sensors,
            "calibration": {
                "event_intrinsics": calibration.event_intrinsics.tolist(),
                "event_distortion": calibration.event_distortion.tolist(),
                "event_resolution": list(calibration.event_resolution),
                "source_kind": calibration.source_kind,
                "metadata": calibration.metadata,
            },
            "input_receipts_sha256": {
                str(args.campaign / name): file_sha(args.campaign / name)
                for name in ("FREEZE.json", "SENSORS.json")
            },
            "source_sha256": {str(path): file_sha(path) for path in source_paths},
            "setup_seconds": time.perf_counter() - setup_start,
            "targets_opened": False,
            "test_sequences_opened": False,
            "training_performed": False,
            "scoring_performed": False,
        }
        freeze_path = args.output / "CACHE_FREEZE.json"
        write_json(freeze_path, freeze)
        rows = []
        for index, query in enumerate(queries):
            started = time.perf_counter()
            query_us = int(query["anchor_us"])
            event_box = input_bbox_to_event_box(
                query["box_xyxy"], calibration, coordinate_plane="event"
            )
            roi = pad_event_roi(event_box, calibration.event_resolution)
            start_us, end_us = trailing_window(query_us, reader.event_start_us, reader.event_end_us)
            read_started = time.perf_counter()
            raw = reader.read_events(start_us, end_us)
            read_seconds = time.perf_counter() - read_started
            selection_started = time.perf_counter()
            selected, roi_count, ranks = select_roi_events(raw, start_us, end_us, query_us, roi)
            selection_seconds = time.perf_counter() - selection_started
            filename = f"query_{index:05d}.npz"
            destination = args.output / filename
            pending = args.output / f".{filename}.building.npz"
            save_started = time.perf_counter()
            np.savez_compressed(
                pending,
                x=selected[:, 0].astype(np.int32),
                y=selected[:, 1].astype(np.int32),
                t_us=selected[:, 2].astype(np.int64),
                p=selected[:, 3].astype(np.int8),
                query_us=np.array(query_us, np.int64),
                roi_xyxy=np.asarray(roi, np.float64),
            )
            pending.replace(destination)
            save_seconds = time.perf_counter() - save_started
            rows.append(
                {
                    "query_id": f"Slider-500:{index}",
                    "row_index": index,
                    "frame_index": int(query["frame_index"]),
                    "query_us": query_us,
                    "timestamp_text": query["timestamp_text"],
                    "file": filename,
                    "sha256": file_sha(destination),
                    "current_bbox_raw_xyxy": list(query["box_xyxy"]),
                    "current_bbox_event_xyxy": list(event_box),
                    "roi_xyxy": list(roi),
                    "window_start_us": start_us,
                    "window_end_exclusive_us": end_us,
                    "start_clipped_to_coverage": start_us > query_us - WINDOW_US,
                    "end_clipped_to_coverage": end_us < query_us + 1,
                    "raw_window_count": len(raw),
                    "roi_count_before_sampling": roi_count,
                    "selected_count": len(selected),
                    "positive_count": int((selected[:, 3] == 1).sum()),
                    "negative_count": int((selected[:, 3] == 0).sum()),
                    "first_selected_us": int(selected[0, 2]) if len(selected) else None,
                    "last_selected_us": int(selected[-1, 2]) if len(selected) else None,
                    "query_minus_last_selected_us": query_us - int(selected[-1, 2])
                    if len(selected)
                    else None,
                    "selection_ranks_sha256": hashlib.sha256(ranks.tobytes()).hexdigest(),
                    "sensor_clock_origin_ns": sensors["clock_origin_ns"],
                    "calibration_sha256": sensors["files"]["calibParams.mat"]["sha256"],
                    "read_seconds": read_seconds,
                    "selection_seconds": selection_seconds,
                    "save_seconds": save_seconds,
                    "total_cache_seconds": time.perf_counter() - started,
                }
            )
            if index % 10 == 0 or index == len(queries) - 1:
                print(
                    f"Slider500 {index + 1}/77 raw={len(raw)} "
                    f"ROI={roi_count} selected={len(selected)}",
                    flush=True,
                )
        manifest = {
            "schema": "phase_event_expansion_development_manifest_v1",
            "status": "complete",
            "cache_freeze_sha256": file_sha(freeze_path),
            "sequence_id": "Slider-500",
            "rows": rows,
            "queries": 77,
            "total_selected_events": sum(r["selected_count"] for r in rows),
            "labels_in_cache": False,
            "raw_recordings_copied": False,
            "coordinates": freeze["coordinates"],
            "calibration": freeze["calibration"],
            "calibration_file_sha256": sensors["files"]["calibParams.mat"]["sha256"],
            "timing_scope": (
                "Cache read/crop/sample/save/per-window hash; "
                "excludes expert inference and reader setup/source-file hashing"
            ),
        }
        write_json(args.output / "MANIFEST.json", manifest)
        print(
            "COMPLETE: 77 causal Slider500 queries, "
            f"{manifest['total_selected_events']} cached events"
        )


if __name__ == "__main__":
    main()
