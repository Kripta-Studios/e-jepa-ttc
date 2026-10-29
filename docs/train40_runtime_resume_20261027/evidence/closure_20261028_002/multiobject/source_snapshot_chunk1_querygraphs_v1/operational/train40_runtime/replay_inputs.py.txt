"""Construct new causal sensor-clock queries without touching frozen cohorts.

This is offline replay of one public annotated scenario target, not a detector
or measured sensor-arrival trace. Annotation availability is an explicit clock
assumption. No RGB pixels, TTC, depth, velocity or ground-truth assets are read
while selecting queries. Future clock metadata defines the replay range; only
annotations available by each query are parsed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from operational.evttc_transfer.inputs import (
    _calibration,
    _calibration_sha256,
    _canonical_sha256,
    _single_object_bbox,
    project_rgb_box_rotation_only,
)
from operational.rgb_port_revision.transfer_inputs import sensor_triplets

SCHEMA = "train40_sensor_clock_replay_queries_v1"


def _label_inventory(folder: Path) -> dict[int, list[Path]]:
    """Inspect filenames only; reading future annotation contents is forbidden."""
    result: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(folder.glob("*.json")):
        if path.stem.isdigit():
            result[int(path.stem)].append(path)
    return result


def _annotation(
    index: int,
    timestamp_us: int,
    *,
    available_us: int,
    query_us: int,
    inventory: Mapping[int, Sequence[Path]],
    target_type: str,
    calibration: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any] | None, str]:
    if timestamp_us > query_us or available_us > query_us:
        raise ValueError("Attempted to read an annotation before its declared availability")
    paths = inventory.get(index, [])
    if not paths:
        return None, "MISSING_CAUSAL_TARGET_ANNOTATION"
    if len(paths) != 1:
        return None, "AMBIGUOUS_ANNOTATION_FILENAME"
    path = paths[0]
    payload = path.read_bytes()
    try:
        decoded = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None, "INVALID_ANNOTATION_JSON"
    if not isinstance(decoded, dict) or not isinstance(decoded.get("objects"), list):
        return None, "INVALID_ANNOTATION_OBJECTS"
    candidates = [
        (object_index, item)
        for object_index, item in enumerate(decoded["objects"])
        if isinstance(item, dict) and str(item.get("category", "")).lower() == target_type.lower()
    ]
    if len(candidates) != 1 or candidates[0][1].get("group") != 1:
        return None, "AMBIGUOUS_OR_NONPRIMARY_SCENARIO_TARGET"
    object_index, obj = candidates[0]
    try:
        box = _single_object_bbox(obj)
    except (TypeError, ValueError):
        return None, "INVALID_CAUSAL_TARGET_ROI"
    if box is None:
        return None, "INVALID_CAUSAL_TARGET_ROI"
    event_box = project_rgb_box_rotation_only(box, calibration)
    return {
        "index": index,
        "rgb_box": list(box),
        "event_box": list(event_box),
        "source": {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "frame_index": index,
            "timestamp_us": timestamp_us,
            "available_us": available_us,
            "category": obj["category"],
            "group": obj.get("group"),
            "object_index": object_index,
            "availability_basis": "sensor_timestamp_plus_declared_replay_delay",
        },
    }, ""


def generate_sensor_clock_replay(
    base_manifest: Mapping[str, Any],
    *,
    step_us: int = 20_000,
    sequence_ids: Sequence[str] | None = None,
    bounds_us: Mapping[str, tuple[int, int]] | None = None,
    max_updates_per_sequence: int | None = None,
    annotation_delay_us: int = 0,
) -> dict[str, Any]:
    """Return a distinct replay manifest, exclusions and assumption provenance.

    Bounds are inclusive. By default each physical sequence spans the first and
    last registered base anchor. ``max_updates_per_sequence`` limits scheduled
    attempts, including exclusions, so missing inputs cannot improve coverage.
    Explicit ``bounds_us`` may extend the replay within available sensor support.
    Nothing is written and no original manifest row is modified.

    The latest available three-frame native-100ms annotation triplet defines
    the ROI. Its last two boxes are projected using the same rotation-only
    calibration and 25% shared-square rule as the frozen transfer adapter.
    Event windows end at the new query timestamp. The adapter may therefore
    crop later RGB observations using these causally available, older boxes.
    """
    if isinstance(step_us, bool) or not isinstance(step_us, int) or step_us <= 0:
        raise ValueError("Replay step_us must be a positive integer")
    if (
        isinstance(annotation_delay_us, bool)
        or not isinstance(annotation_delay_us, int)
        or annotation_delay_us < 0
    ):
        raise ValueError("annotation_delay_us must be a nonnegative integer")
    if max_updates_per_sequence is not None and (
        isinstance(max_updates_per_sequence, bool)
        or not isinstance(max_updates_per_sequence, int)
        or max_updates_per_sequence < 1
    ):
        raise ValueError("max_updates_per_sequence must be a positive integer")
    base_rows = base_manifest.get("rows")
    if not isinstance(base_rows, list) or not base_rows:
        raise ValueError("A nonempty registered base population is required")
    by_sequence: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in base_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Base manifest rows must be objects")
        sequence = str(row.get("physical_sequence_id", row["sequence_id"]))
        by_sequence[sequence].append(row)
    selected = sorted(by_sequence) if sequence_ids is None else list(sequence_ids)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(by_sequence):
        raise ValueError("Requested physical sequence IDs must be unique and present in the base")
    if bounds_us is not None and set(bounds_us) - set(selected):
        raise ValueError("Replay bounds reference an unselected sequence")
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    scheduled = 0
    for sequence in sorted(selected):
        population = sorted(by_sequence[sequence], key=lambda row: int(row["anchor_us"]))
        base = population[0]
        raw_path = Path(str(base["raw_path"])).resolve(strict=True)
        label_dirs = {
            Path(str(source["path"])).resolve().parent
            for row in population
            for source in row["bbox_sources"]
        }
        if len(label_dirs) != 1:
            raise ValueError(f"{sequence}: expected exactly one public annotation directory")
        label_dir = label_dirs.pop()
        if any(
            Path(str(row["raw_path"])).resolve() != raw_path
            or row["target_type"] != base["target_type"]
            or row["raw_stat"]["size_bytes"] != base["raw_stat"]["size_bytes"]
            or row["raw_stat"]["mtime_ns"] != base["raw_stat"]["mtime_ns"]
            for row in population
        ):
            raise ValueError(f"{sequence}: base rows disagree on physical source identity")
        stat = raw_path.stat()
        raw_stat = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if any(int(base["raw_stat"][key]) != value for key, value in raw_stat.items()):
            raise ValueError(f"{sequence}: raw source differs from the registered base")
        with h5py.File(raw_path, "r") as handle:
            clock = np.asarray(handle["blackflys/left/ts"], dtype=np.int64)
            event_clock = handle["prophesee/event_cam_left/t"]
            if not isinstance(event_clock, h5py.Dataset) or len(event_clock) == 0:
                raise ValueError(f"{sequence}: event clock is missing or empty")
            event_first, event_last = int(event_clock[0]), int(event_clock[-1])
            calibration = _calibration(handle)
        if clock.ndim != 1 or len(clock) < 3 or np.any(np.diff(clock) <= 0):
            raise ValueError(f"{sequence}: invalid actual RGB sensor clock")
        first, last = (
            bounds_us[sequence]
            if bounds_us is not None and sequence in bounds_us
            else (int(population[0]["anchor_us"]), int(population[-1]["anchor_us"]))
        )
        if (
            isinstance(first, bool)
            or isinstance(last, bool)
            or not isinstance(first, int)
            or not isinstance(last, int)
            or first < 0
            or last < first
        ):
            raise ValueError(f"{sequence}: replay bounds must be ordered nonnegative integers")
        label_paths = _label_inventory(label_dir)
        annotations: dict[int, tuple[dict[str, Any] | None, str]] = {}
        scheduled_sequence = 0
        accepted_sequence = 0
        for query in range(first, last + 1, step_us):
            if (
                max_updates_per_sequence is not None
                and scheduled_sequence >= max_updates_per_sequence
            ):
                break
            scheduled += 1
            scheduled_sequence += 1
            identity = f"SENSOR_REPLAY:{sequence}:{query}"
            reason = ""
            if query - 600_000 < event_first or query > event_last:
                reason = "INSUFFICIENT_FULL_600MS_EVENT_SUPPORT"
            groups = sensor_triplets(clock, query - annotation_delay_us) if not reason else []
            ids = groups[-1] if groups else np.empty(0, dtype=np.int64)
            if not reason and (
                len(ids) != 3
                or query - int(clock[ids[0]]) > 650_000
                or np.any(clock[ids] + annotation_delay_us > query)
            ):
                reason = "NO_AVAILABLE_NATIVE_100MS_RGB_ANNOTATION_TRIPLET"
            triple: list[dict[str, Any]] = []
            if not reason:
                for index_value in ids:
                    index = int(index_value)
                    if index not in annotations:
                        annotations[index] = _annotation(
                            index,
                            int(clock[index]),
                            available_us=int(clock[index]) + annotation_delay_us,
                            query_us=query,
                            inventory=label_paths,
                            target_type=str(base["target_type"]),
                            calibration=calibration,
                        )
                    item, reason = annotations[index]
                    if item is None:
                        break
                    triple.append(item)
            if reason:
                excluded.append(
                    {
                        "query_id": identity,
                        "physical_sequence_id": sequence,
                        "anchor_us": query,
                        "reason": reason,
                    }
                )
                continue
            boxes = [item["event_box"] for item in triple]
            sources = [dict(item["source"]) for item in triple]
            row = {
                "query_id": identity,
                "sequence_id": sequence,
                "physical_sequence_id": sequence,
                "anchor_us": query,
                "raw_path": str(raw_path),
                "raw_stat": dict(raw_stat),
                "windows_us": [
                    [query + lag - 100_000, query + lag] for lag in (-200_000, -100_000, 0)
                ],
                "boxes_xyxy3": boxes,
                "square_xyxy": common_square_from_boxes(boxes, (1, 2), margin_fraction=0.25),
                "bbox_sources": sources,
                "rgb_boxes": [item["rgb_box"] for item in triple],
                "rgb_frame_indices": [int(index) for index in ids],
                "rgb_timestamps_us": [int(clock[index]) for index in ids],
                "calibration_sha256": _calibration_sha256(calibration),
                "multiple_car_objects": False,
                "target_type": str(base["target_type"]),
                "scenario_family": str(base.get("scenario_family", "unknown")),
                "speed_bucket": str(base.get("speed_bucket", "unknown")),
                "strata": ["SENSOR_CLOCK_REPLAY_SINGLE_ANNOTATED_TARGET"],
                "replay_step_us": step_us,
                "annotation_delay_us": annotation_delay_us,
                "roi_sensor_anchor_us": sources[-1]["timestamp_us"],
                "roi_available_us": sources[-1]["available_us"],
                "roi_age_us": query - sources[-1]["timestamp_us"],
            }
            row["metadata_sha256"] = _canonical_sha256(row)
            rows.append(row)
            accepted_sequence += 1
        inventory.append(
            {
                "physical_sequence_id": sequence,
                "raw_path": str(raw_path),
                "raw_stat": raw_stat,
                "annotation_directory": str(label_dir),
                "event_sensor_range_us": [event_first, event_last],
                "requested_bounds_us": [first, last],
                "scheduled_updates": scheduled_sequence,
                "accepted_updates": accepted_sequence,
                "annotation_frames_inspected": len(annotations),
            }
        )
    if (
        len({row["query_id"] for row in rows}) != len(rows)
        or len(rows) + len(excluded) != scheduled
    ):
        raise RuntimeError("Replay coverage accounting or query identity is inconsistent")
    return {
        "schema": SCHEMA,
        "rows": rows,
        "excluded": excluded,
        "inventory": inventory,
        "scheduled_updates": scheduled,
        "accepted_updates": len(rows),
        "exclusion_reasons": dict(sorted(Counter(item["reason"] for item in excluded).items())),
        "targets_read": False,
        "optimizer_updates": 0,
        "provenance": {
            "base_manifest_canonical_sha256": _canonical_sha256(dict(base_manifest)),
            "step_us": step_us,
            "frequency_hz": 1_000_000 / step_us,
            "annotation_delay_us": annotation_delay_us,
            "scope": "sensor-clock replay of one annotated primary scenario target",
            "hardware_arrivals_measured": False,
            "annotation_availability_measured": False,
            "annotation_availability": "actual RGB timestamp plus declared replay delay",
            "roi_policy": "latest available native triplet; last-two-box shared square, margin0.25",
            "event_roi_projection": "same calibrated rotation-only RGB-to-event mapping; no depth",
            "event_history_us": 600_000,
            "maximum_rgb_history_us": 650_000,
            "query_schedule": "fixed sensor-clock intervals; rejected updates remain in coverage",
            "future_annotations_read": False,
            "original_manifest_modified": False,
            "rgb_pixels_read": False,
            "event_coordinates_or_polarities_read": False,
            "sequence_endpoint_clock_metadata_read": True,
            "outputs_written": False,
        },
    }
