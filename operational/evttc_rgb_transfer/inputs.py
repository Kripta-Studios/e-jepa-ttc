"""Prepare the frozen EvTTC cohort for published full RGB+event Garl.

The query population comes unchanged from ``evttc_transfer_20261008``.  This
module never prepares H8 inputs and never opens TTC, distance, depth,
navigation, or GT assets.  RGB pixels come from the Blackfly-left frames
nearest to the two fixed event endpoints, matching native ``sync=front``.  The
already pinned causal boxes still define the crop.  A bounded acquisition wait
of at most 1 ms is explicit when RGB lands just after an event endpoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from e_jepa_ttc.data.garl_official_preprocessing import (
    official_resize_roi,
    official_square_box,
)
from e_jepa_ttc.efficient_context.garl_input import native_feature
from operational.evttc_transfer.inputs import (
    EvTTCEventReader,
    _canonical_sha256,
    _sha256_file,
    _single_object_bbox,
)

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
RGB_INTERVAL_TARGET_US = 100_000
RGB_INTERVAL_TOLERANCE_US = 1_000

CONTRACT: dict[str, Any] = {
    "protocol": "evttc_historical_dev32_garl_full_fixed_queries_v1",
    "query_population": "unchanged QUERY_MANIFEST.json, 1024 rows",
    "camera": {"rgb": "blackflys/left", "events": "prophesee/event_cam_left"},
    "sensor_shape": [46, 128, 128],
    "sensor_order": ["rgb_endpoint_0", "rgb_endpoint_1", "event_endpoint_0", "event_endpoint_1"],
    "rgb_shape": [2, 3, 128, 128],
    "event_shape": [40, 128, 128],
    "rgb_normalization": {
        "scale": "uint8/255 in float32",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "order": "normalize full RGB frame before shared crop",
    },
    "rgb_sync": (
        "nearest blackflys/left/ts to each fixed event endpoint within 1 ms; "
        "ties select the earlier frame index"
    ),
    "rgb_crop": (
        "upstream shared max-edge square centered on endpoint 1; grid_sample bilinear "
        "align_corners=True"
    ),
    "event_crop": (
        "native per-endpoint square with shared max edge; 20-plane timevolume and upstream resize"
    ),
    "fixed_dt_s": 0.1,
    "rgb_interval_target_us": RGB_INTERVAL_TARGET_US,
    "rgb_interval_tolerance_us": RGB_INTERVAL_TOLERANCE_US,
    "availability": (
        "prediction_available_us=max(anchor_us, RGB endpoint timestamps); positive bounded "
        "sync wait is recorded"
    ),
    "unavailable_policy": (
        "retain query; no window/box reselection, interpolation, or extrapolation; preserve "
        "event-only comparator"
    ),
    "forbidden_assets": ["ttc.csv", "gt.hdf5", "distance", "depth", "navigation"],
}


def _validate_frozen_row(row: Mapping[str, Any]) -> Path:
    required = {
        "query_id",
        "raw_path",
        "raw_stat",
        "windows_us",
        "boxes_xyxy3",
        "bbox_sources",
        "metadata_sha256",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"frozen query lacks fields: {missing}")
    metadata = dict(row)
    expected_metadata_sha = str(metadata.pop("metadata_sha256"))
    if _canonical_sha256(metadata) != expected_metadata_sha:
        raise ValueError("frozen query metadata SHA-256 mismatch")
    path = Path(str(row["raw_path"]))
    stat = path.stat()
    expected_stat = row["raw_stat"]
    if stat.st_size != int(expected_stat["size_bytes"]) or stat.st_mtime_ns != int(
        expected_stat["mtime_ns"]
    ):
        raise ValueError("EvTTC raw HDF5 stat differs from the frozen query")
    return path


def _rgb_box(source: Mapping[str, Any]) -> tuple[float, float, float, float]:
    path = Path(str(source["path"]))
    if _sha256_file(path) != source["sha256"]:
        raise ValueError("frozen RGB bbox source SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects", [])
    compatible = [
        item
        for item in objects
        if isinstance(item, dict)
        and str(item.get("category")) == str(source["category"])
        and item.get("group") == source.get("group")
    ]
    if len(compatible) != 1:
        raise ValueError("frozen RGB bbox identity is no longer unique")
    bbox = _single_object_bbox(compatible[0])
    if bbox is None:
        raise ValueError("frozen RGB bbox became invalid")
    return bbox


def _normalized_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Blackfly-left frame must be uint8 RGB [H,W,3]")
    tensor = frame.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _nearest_frame_indices(
    frame_timestamps: np.ndarray,
    endpoints_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Bind fixed endpoints to nearest RGB frames with deterministic ties."""

    timestamps = np.asarray(frame_timestamps, dtype=np.int64)
    endpoints = np.asarray(endpoints_us, dtype=np.int64)
    if timestamps.ndim != 1 or endpoints.shape != (2,) or len(timestamps) == 0:
        raise ValueError("RGB synchronization requires 1D timestamps and two endpoints")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("Blackfly-left timestamps are not monotonic")
    indices = []
    offsets = []
    for endpoint in endpoints:
        insertion = int(np.searchsorted(timestamps, endpoint, side="left"))
        candidates = sorted({max(0, insertion - 1), min(len(timestamps) - 1, insertion)})
        selected = min(
            candidates,
            key=lambda index: (abs(int(timestamps[index]) - int(endpoint)), index),
        )
        offset = int(timestamps[selected]) - int(endpoint)
        if abs(offset) > RGB_INTERVAL_TOLERANCE_US:
            return None
        indices.append(selected)
        offsets.append(offset)
    return np.asarray(indices, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


def _event_input(row: Mapping[str, Any], raw_path: Path) -> tuple[np.ndarray, list[list[int]]]:
    windows = np.asarray(row["windows_us"], dtype=np.int64)
    if windows.shape != (3, 2):
        raise ValueError("frozen query event windows must have shape [3,2]")
    boxes: list[tuple[float, float, float, float]] = []
    for raw_box in row["boxes_xyxy3"][-2:]:
        if len(raw_box) != 4:
            raise ValueError("frozen event boxes must be xyxy")
        boxes.append((float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3])))
    parts = []
    squares: list[list[int]] = []
    with EvTTCEventReader(raw_path) as reader:
        for local_index, window_index in enumerate((1, 2)):
            start, end = (int(value) for value in windows[window_index])
            if end - start != 100_000:
                raise ValueError("full Garl requires exact 100 ms event endpoints")
            raw = reader.read_window(start, end)
            if not len(raw["t"]):
                raise ValueError("full Garl event endpoint is empty")
            square = official_square_box(boxes, local_index)
            squares.append(list(square))
            parts.append(native_feature(raw, square, size=128).numpy())
    result = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if result.shape != (40, 128, 128):
        raise ValueError("full Garl event input violates [40,128,128]")
    return result, squares


def prepare(row: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare full Garl input or retain an explicit RGB-timing unavailability."""

    raw_path = _validate_frozen_row(row)
    sources = list(row["bbox_sources"][-2:])
    if len(sources) != 2:
        raise ValueError("full Garl requires the final two frozen causal bbox sources")
    event_windows = np.asarray(row["windows_us"], dtype=np.int64)[-2:]
    event_endpoint_timestamps = event_windows[:, 1].astype(np.int64)
    garl_events, event_squares = _event_input(row, raw_path)
    event_sha = hashlib.sha256(np.ascontiguousarray(garl_events).tobytes()).hexdigest()
    metadata: dict[str, Any] = {
        "query_id": row["query_id"],
        "source_bbox_frame_indices": [int(source["frame_index"]) for source in sources],
        "source_bbox_timestamps_us": [int(source["timestamp_us"]) for source in sources],
        "event_endpoint_timestamps_us": event_endpoint_timestamps.tolist(),
        "event_endpoint_interval_us": int(
            event_endpoint_timestamps[1] - event_endpoint_timestamps[0]
        ),
        "event_squares_xyxy": event_squares,
        "garl_events_sha256": event_sha,
        "alignment_rule": (
            "native sync-front: nearest Blackfly-left frame to each fixed event endpoint within "
            "1 ms; causal frozen boxes define crops; no interpolation, extrapolation, window/box "
            "replacement, or query reselection"
        ),
        "limitation": (
            "RGB may arrive up to 1 ms after an event endpoint; this bounded acquisition wait is "
            "included in prediction availability"
        ),
    }
    rgb_boxes = [_rgb_box(source) for source in sources]
    rgb_square = official_square_box(rgb_boxes, 1)
    with h5py.File(raw_path, "r") as handle:
        data = handle.get("blackflys/left/data")
        timestamps = handle.get("blackflys/left/ts")
        if not isinstance(data, h5py.Dataset) or not isinstance(timestamps, h5py.Dataset):
            raise ValueError("EvTTC HDF5 lacks explicit Blackfly-left RGB datasets")
        frame_timestamps = np.asarray(timestamps, dtype=np.int64)
        synchronization = _nearest_frame_indices(frame_timestamps, event_endpoint_timestamps)
        if synchronization is None:
            metadata["available"] = False
            return {
                "sensor": None,
                "garl_events": garl_events,
                "unavailable_reason": "RGB_ENDPOINT_WITHOUT_FRAME_WITHIN_1MS",
                "metadata": metadata,
            }
        rgb_frame_indices, rgb_offsets_us = synchronization
        rgb_timestamps = frame_timestamps[rgb_frame_indices]
        rgb_interval_us = int(rgb_timestamps[1] - rgb_timestamps[0])
        metadata.update(
            rgb_frame_indices=rgb_frame_indices.tolist(),
            rgb_timestamps_us=rgb_timestamps.tolist(),
            rgb_event_endpoint_offsets_us=rgb_offsets_us.tolist(),
            rgb_interval_us=rgb_interval_us,
        )
        if abs(rgb_interval_us - RGB_INTERVAL_TARGET_US) > RGB_INTERVAL_TOLERANCE_US:
            metadata["available"] = False
            return {
                "sensor": None,
                "garl_events": garl_events,
                "unavailable_reason": (
                    "RGB_ENDPOINT_INTERVAL_OUTSIDE_FIXED_DT:"
                    f"actual_us={rgb_interval_us},target_us={RGB_INTERVAL_TARGET_US},"
                    f"tolerance_us={RGB_INTERVAL_TOLERANCE_US}"
                ),
                "metadata": metadata,
            }
        frames = []
        for frame_index in rgb_frame_indices:
            frames.append(_normalized_rgb(np.asarray(data[frame_index])))
    full_rgb = np.concatenate(frames, axis=0)
    resized = official_resize_roi(full_rgb, rgb_square, (128, 128)).numpy()
    rgb_endpoints = resized.reshape(2, 3, 128, 128).astype(np.float32, copy=False)
    sensor = np.concatenate((rgb_endpoints.reshape(6, 128, 128), garl_events), axis=0)
    sensor = sensor.astype(np.float32, copy=False)
    if sensor.shape != (46, 128, 128) or not np.isfinite(sensor).all():
        raise ValueError("full Garl sensor violates finite [46,128,128] FP32 contract")
    metadata.update(
        available=True,
        prediction_available_us=max(int(row["anchor_us"]), int(rgb_timestamps.max())),
        additional_sync_wait_us=max(0, int(rgb_timestamps.max()) - int(row["anchor_us"])),
        rgb_square_xyxy=list(rgb_square),
        rgb_endpoints_sha256=hashlib.sha256(
            np.ascontiguousarray(rgb_endpoints).tobytes()
        ).hexdigest(),
        sensor_sha256=hashlib.sha256(np.ascontiguousarray(sensor).tobytes()).hexdigest(),
    )
    return {
        "sensor": sensor,
        "garl_events": garl_events,
        "unavailable_reason": None,
        "metadata": metadata,
    }


__all__ = ["CONTRACT", "prepare"]
