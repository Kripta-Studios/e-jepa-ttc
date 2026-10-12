"""Build target-free queries from the three author-specified EvTTC intervals."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from e_jepa_ttc.data.garl_official_preprocessing import official_resize_roi, official_square_box
from e_jepa_ttc.efficient_context.garl_input import native_feature
from e_jepa_ttc.rgb_port.data import crop_uint8
from operational.evttc_rgb_transfer.inputs import _normalized_rgb
from operational.evttc_transfer.inputs import (
    EvTTCEventReader,
    _calibration,
    _calibration_sha256,
    _canonical_sha256,
    _single_object_bbox,
    project_rgb_box_rotation_only,
)
from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.rgb_port_revision.transfer_inputs import sensor_triplets

SCENARIOS = (
    ("CCRs2-medium", "CCRs-2-medium-100-overlap-100", 9_400_000, 11_600_000),
    ("CCRs2-high", "CCRs-2-high-100-overlap-100", 15_500_000, 17_400_000),
    ("CCRm-medium", "CCRm-medium-100-overlap-100", 21_400_000, 22_200_000),
)
PROTOCOL = {
    "schema": "author_evttc_reconstruction_v1",
    "scope": "reconstruction_from_author_clarification_not_verified_Table_VI_queries",
    "frame_sampling": (
        "first_RGB_frame_in_interval_then_nearest_100ms_grid; ties_earlier; tolerance_1ms"
    ),
    "triplets": "three_consecutive_grid_frames_inside_interval; last_frame_is_query",
    "target_selection": (
        "first_valid_annotation_object_initially_then_nearest_previous_selected_center"
    ),
    "annotations": "oracle_leftlabel_available_at_frame_timestamp; no_artificial_50ms_delay",
    "mapping": "calibrated_undistorted_rays_rotation_only; translation_requires_unavailable_depth",
    "mapping_limitation": "author_email_does_not_specify_projection_depth_or_transform_convention",
    "crop": "release_max_edge_square_factor_1.0; RGB_shared_at_last_endpoint; event_per_endpoint",
    "Garl_event_segments": "actual_RGB_intervals_t0_to_t1_and_t1_to_t2_half_open",
    "Garl_delta_t": "release_fixed_0.1s; measured_intervals_logged",
    "V13_context": "native_four_event_observations_and_causal_RGB_history_at_same_query",
    "V13_ROI": "same_unexpanded_query_square_across_history; model_specific_representation",
    "V12_context": "native_H8_at_50ms; same_unexpanded_event_query_square",
    "labels": (
        "read_only_after_all_predictions_sealed; nearest_timestamp_max_0.05s; no_interpolation"
    ),
    "invalid_policy": "retain_every_candidate_and_each_model_failure_with_reason",
    "metric_primary": (
        "RTE_percent_native_outputs; pooled_and_macro_sequence; coverage_and_common_rows"
    ),
    "seeds_or_models_selected_from_scores": False,
    "optimizer_updates": 0,
}


def sampled_frames(clock: np.ndarray, start: int, end: int) -> list[int]:
    """Use a fixed timestamp grid, without bridging missing sensor observations."""
    if clock.ndim != 1 or np.any(np.diff(clock) <= 0):
        raise ValueError("RGB clock must be strictly increasing")
    inside = np.flatnonzero((clock >= start) & (clock <= end))
    if not len(inside):
        return []
    result = []
    for target in range(int(clock[inside[0]]), end + 1, 100_000):
        pos = int(np.searchsorted(clock, target))
        candidates = [i for i in (pos - 1, pos) if 0 <= i < len(clock)]
        index = min(candidates, key=lambda i: (abs(int(clock[i]) - target), i))
        result.append(index if abs(int(clock[index]) - target) <= 1_000 else -1)
    return result


def choose_box(
    objects: Sequence[dict[str, Any]], previous: tuple[float, float] | None
) -> tuple[int, tuple[float, float, float, float]]:
    """Track by previous box center, preserving annotation order for ties."""
    candidates = [(i, _single_object_bbox(obj)) for i, obj in enumerate(objects)]
    valid = [(i, box) for i, box in candidates if box is not None]
    if not valid:
        raise ValueError("NO_VALID_ANNOTATED_OBJECT")
    if previous is None:
        return valid[0]
    return min(
        valid,
        key=lambda item: (
            ((item[1][0] + item[1][2]) / 2 - previous[0]) ** 2
            + ((item[1][1] + item[1][3]) / 2 - previous[1]) ** 2,
            item[0],
        ),
    )


def build_manifest(previous_manifest: Path, output: Path) -> dict[str, Any]:
    """Use historical manifest only to locate sensors, never its queries or labels."""
    inventory = json.loads(previous_manifest.read_text(encoding="utf-8"))
    paths = {r["sequence_id"]: Path(r["raw_path"]) for r in inventory["rows"]}
    rows, audit, source_files = [], [], {}
    for name, sequence, start, end in SCENARIOS:
        path = paths[sequence]
        with h5py.File(path, "r") as handle:
            clock = np.asarray(handle["blackflys/left/ts"], np.int64)
            calibration = _calibration(handle)
            indices = sampled_frames(clock, start, end)
        previous = None
        observations = []
        for ordinal, index in enumerate(indices):
            reason = ""
            item = None
            if index < 0:
                reason = "FRAME_GRID_MISMATCH"
            else:
                label = path.parent / "leftlabel" / f"{index:04d}.json"
                if not label.is_file():
                    reason = "MISSING_ANNOTATION"
                else:
                    source_files[str(label)] = sha256_file(label)
                    payload = json.loads(label.read_text(encoding="utf-8"))
                    try:
                        object_index, box = choose_box(payload.get("objects", []), previous)
                        previous = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        event_box = project_rgb_box_rotation_only(box, calibration)
                        item = {
                            "index": index,
                            "timestamp_us": int(clock[index]),
                            "rgb_box": box,
                            "event_box": event_box,
                            "annotation_path": str(label),
                            "annotation_sha256": source_files[str(label)],
                            "selected_object_index": object_index,
                            "object_count": len(payload.get("objects", [])),
                        }
                    except ValueError as exc:
                        reason = str(exc)
            observations.append(item)
            query_id = f"{name}:{ordinal:03d}"
            candidate = {
                "query_id": query_id,
                "scenario": name,
                "sequence_id": sequence,
                "frame_index": index,
                "anchor_us": int(clock[index]) if index >= 0 else None,
            }
            if ordinal < 2:
                reason = reason or "TRIPLET_WARMUP"
            elif any(x is None for x in observations[-3:]):
                reason = reason or "TRIPLET_HAS_INVALID_FRAME_OR_ANNOTATION"
            if reason:
                audit.append({**candidate, "status": "INPUT_REJECTED", "reason": reason})
                continue
            triplet = observations[-3:]
            anchor = int(clock[index])
            boxes = [x["event_box"] for x in triplet]
            stat = path.stat()
            row = {
                **candidate,
                "raw_path": str(path),
                "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
                "windows_us": [
                    [anchor - 300_000 + k * 100_000, anchor - 200_000 + k * 100_000]
                    for k in range(3)
                ],
                "boxes_xyxy3": boxes,
                "rgb_boxes": [x["rgb_box"] for x in triplet],
                "rgb_frame_indices": [x["index"] for x in triplet],
                "rgb_timestamps_us": [x["timestamp_us"] for x in triplet],
                "bbox_sources": triplet,
                "square_xyxy": official_square_box(boxes[-2:], 1),
                "rgb_square_xyxy": official_square_box([x["rgb_box"] for x in triplet[-2:]], 1),
                "calibration_sha256": _calibration_sha256(calibration),
            }
            row["metadata_sha256"] = _canonical_sha256(row)
            rows.append(row)
            audit.append({**candidate, "status": "INPUT_ADMITTED", "reason": ""})
    result = {
        "status": "LABEL_FREE_MANIFEST",
        "protocol": PROTOCOL,
        "rows": rows,
        "rows_metadata_sha256": _canonical_sha256(rows),
        "candidates": audit,
        "annotation_sources": source_files,
        "locator_manifest_sha256": sha256_file(previous_manifest),
    }
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != json.loads(json.dumps(result)):
            raise ValueError("Existing prospective manifest differs")
    else:
        atomic_write_json(output, result)
    return result


def validate_row(row: dict[str, Any]) -> None:
    """Verify bound files and sensor stat before preparing any model's inputs."""
    metadata = dict(row)
    expected = metadata.pop("metadata_sha256")
    if _canonical_sha256(metadata) != expected:
        raise ValueError("Query metadata changed")
    stat = Path(row["raw_path"]).stat()
    if row["raw_stat"] != {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
        raise ValueError("Raw sensor file changed")
    for source in row["bbox_sources"]:
        if sha256_file(Path(source["annotation_path"])) != source["annotation_sha256"]:
            raise ValueError("Annotation changed")


def garl_inputs(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Prepare actual consecutive event segments and the release RGB normalization."""
    validate_row(row)
    boxes = row["boxes_xyxy3"][-2:]
    timestamps = row["rgb_timestamps_us"]
    parts, counts = [], []
    with EvTTCEventReader(Path(row["raw_path"])) as reader:
        for k in range(2):
            raw = reader.read_window(timestamps[k], timestamps[k + 1])
            square = official_square_box(boxes, k)
            roi_count = int(
                (
                    (raw["x"] >= square[0])
                    & (raw["x"] < square[2])
                    & (raw["y"] >= square[1])
                    & (raw["y"] < square[3])
                ).sum()
            )
            counts.append({"total": len(raw["t"]), "roi": roi_count})
            if not len(raw["t"]) or roi_count == 0:
                raise ValueError("EMPTY_EVENT_SEGMENT_OR_ROI")
            parts.append(native_feature(raw, square, size=128).numpy())
    with h5py.File(row["raw_path"], "r") as handle:
        data = handle["blackflys/left/data"]
        if not isinstance(data, h5py.Dataset):
            raise ValueError("RGB data must be a dataset")
        rgb = [
            official_resize_roi(
                _normalized_rgb(np.asarray(data[i])),
                tuple(row["rgb_square_xyxy"]),
                (128, 128),
            ).numpy()
            for i in row["rgb_frame_indices"][-2:]
        ]
    return (
        np.stack(rgb),
        np.stack(parts),
        {"events": counts, "actual_deltas_us": np.diff(timestamps).tolist()},
    )


def rgb_observations(row: dict[str, Any]) -> tuple[list, str]:
    """Native V13 history, using the predeclared unexpanded query ROI."""
    validate_row(row)
    result = []
    query = row["anchor_us"]
    with h5py.File(row["raw_path"], "r") as handle:
        clock = np.asarray(handle["blackflys/left/ts"], np.int64)
        data = handle["blackflys/left/data"]
        if not isinstance(data, h5py.Dataset):
            raise ValueError("RGB data must be a dataset")
        groups = sensor_triplets(clock, query)
        if not groups:
            return [], "INSUFFICIENT_CAUSAL_HISTORY"
        frames = {
            int(i): crop_uint8(np.asarray(data[int(i)]), row["rgb_square_xyxy"])
            for i in np.unique(np.concatenate(groups))
        }
        for ids in groups:
            ts = clock[ids]
            sensor = np.ascontiguousarray(
                np.stack([frames[int(i)] for i in ids]).astype(np.float32) / 255
            )
            result.append(
                (
                    sensor,
                    {
                        "first_sensor_ts_us": int(ts[0]),
                        "anchor_us": int(ts[-1]),
                        "available_us": query,
                        "delta_t_s": (np.diff(ts) / 1e6).tolist(),
                    },
                )
            )
    return result, ""
