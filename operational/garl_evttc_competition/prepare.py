"""Label-blind EvTTC preparation for the three released Garl-TTC LHR models."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from e_jepa_ttc.data.garl_official_preprocessing import (
    official_resize_feature,
    official_resize_roi,
    official_square_box,
    official_timevolume_roi_np,
)
from operational.evttc_competition.geometry import (
    Calibration,
    input_bbox_to_event_box,
    project_event_box_to_rgb,
)

Variant = Literal["full", "event", "visual"]
VARIANTS: tuple[Variant, ...] = ("full", "event", "visual")
PERIOD_US = 100_000
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), np.float32)[:, None, None]
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), np.float32)[:, None, None]
POLICY: dict[str, object] = {
    "version": "garl_evttc_competition_transfer_v1",
    "variants": list(VARIANTS),
    "query_population": "same 434 supplied EvTTC bbox rows in original order",
    "event_windows": "exact half-open microseconds inside recorded coverage",
    "event_endpoints": (
        "two real endpoint windows; mature event-only 2x100ms, startup observed prefix split in two"
    ),
    "rgb_endpoints": (
        "two distinct causal real frames; latest <= query and predecessor nearest 100ms earlier"
    ),
    "interpolation_or_frame_duplication": False,
    "event_representation": "polarity-agnostic native 20-plane timevolume per endpoint",
    "native_timevolume_range_s": 0.1,
    "native_timevolume_bins": 20,
    "native_bin_width_s": 0.005,
    "native_bin_boundary_behavior": (
        "each endpoint uses at most 100000us of sensor support; native timevolume retains its "
        "released first-event-relative 0.1s boundary mask"
    ),
    "full_event_support": (
        "min(actual RGB endpoint spacing,100000us) ending independently at each RGB endpoint"
    ),
    "short_window_relabeling": False,
    "eap_x_offset_pixels": 0,
    "eap_x_plus_5_not_transferred": True,
    "camera": "road right event/right RGB; Slider supplied event/RGB pair",
    "rgb_projection": "calibrated event rays and rotation; translation omitted without depth",
    "bbox_selection": (
        "latest supplied causal bbox at or before each sensor endpoint; earliest query-causal "
        "fallback only when none exists"
    ),
    "bbox_fallback_availability": "query timestamp",
    "native_training_delta_t_s": 0.1,
    "ttc_conversion_delta": "actual endpoint spacing in FP32; native model dT remains 0.1",
    "raw_event_empty": "variant unavailable/null",
    "roi_event_empty": "valid all-zero native timevolume",
    "missing_rgb_pair": "variant unavailable/null",
    "signed_ttc": "preserve finite signed output; no clipping or absolute value",
    "nonfinite_or_singular_height_ratio_output": "variant unavailable/null",
    "targets_used": False,
    "comparison_scope": (
        "local adapted predictions and coverage only; no accuracy or paper-protocol equivalence"
    ),
}


class SensorReader(Protocol):
    """Minimal causal sensor-reader interface required by this transfer adapter."""

    @property
    def event_start_us(self) -> int: ...

    @property
    def event_end_us(self) -> int: ...

    @property
    def rgb_timestamps_us(self) -> np.ndarray: ...

    def read_events(self, start_us: int, end_us: int) -> np.ndarray: ...

    def read_rgb(self, index: int) -> np.ndarray: ...


@dataclass(frozen=True)
class PreparedGarlQuery:
    variant: Variant
    sensor: np.ndarray | None
    delta_t_s: float | None
    reason: str
    details: dict[str, Any]


def _endpoint_boxes(
    rows: Sequence[Mapping[str, Any]],
    endpoints_us: Sequence[int],
    query_us: int,
    calibration: Calibration,
) -> tuple[
    list[tuple[float, float, float, float]],
    list[tuple[float, float, float, float]],
    list[dict[str, Any]],
]:
    query_causal = sorted(
        (row for row in rows if int(row["anchor_us"]) <= query_us),
        key=lambda row: int(row["anchor_us"]),
    )
    if not query_causal:
        raise ValueError("query has no causal supplied box")
    selected: list[tuple[Mapping[str, Any], bool]] = []
    for endpoint in endpoints_us:
        eligible = [row for row in query_causal if int(row["anchor_us"]) <= endpoint]
        selected.append((eligible[-1], False) if eligible else (query_causal[0], True))
    event = [
        input_bbox_to_event_box(item[0]["box_xyxy"], calibration, coordinate_plane="event")
        for item in selected
    ]
    rgb = [project_event_box_to_rgb(box, calibration) for box in event]
    sources = [
        {
            "endpoint_us": int(endpoint),
            "row_index": int(row["row_index"]),
            "index": int(row["index"]),
            "anchor_us": int(row["anchor_us"]),
            "selection": "earliest_query_causal_fallback"
            if fallback
            else "latest_at_or_before_endpoint",
            "fallback": fallback,
            "available_us": query_us if fallback else int(row["anchor_us"]),
            "raw_box": list(row["box_xyxy"]),
            "event_box": list(event_box),
            "rgb_box": list(rgb_box),
        }
        for endpoint, (row, fallback), event_box, rgb_box in zip(
            endpoints_us, selected, event, rgb, strict=True
        )
    ]
    return event, rgb, sources


def _rgb_indices(clock: np.ndarray, query_us: int) -> tuple[int, int] | None:
    current = int(np.searchsorted(clock, query_us, side="right")) - 1
    if current < 1:
        return None
    target = int(clock[current]) - PERIOD_US
    insertion = int(np.searchsorted(clock[:current], target))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < current]
    if not candidates:
        return None
    previous = min(candidates, key=lambda index: (abs(int(clock[index]) - target), index))
    return previous, current


def _event_windows(query_us: int, start_us: int) -> tuple[list[tuple[int, int]], bool] | None:
    available = query_us - start_us
    if available < 2:
        return None
    if available >= 2 * PERIOD_US:
        return [
            (query_us - 2 * PERIOD_US, query_us - PERIOD_US),
            (query_us - PERIOD_US, query_us),
        ], False
    middle = start_us + available // 2
    if not start_us < middle < query_us:
        return None
    return [(start_us, middle), (middle, query_us)], True


def _events(
    reader: SensorReader,
    windows: Sequence[tuple[int, int]],
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[np.ndarray | None, list[dict[str, Any]], str | None]:
    tensors: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(windows):
        if start < reader.event_start_us or end > reader.event_end_us or end <= start:
            return None, details, "event_window_out_of_recorded_coverage"
        raw = reader.read_events(start, end)
        if len(raw) == 0:
            return None, details, "raw_event_window_empty"
        square = official_square_box(list(boxes), index)
        feature, counts = official_timevolume_roi_np(
            square, raw[:, 0], raw[:, 1], raw[:, 2], time_window_s=0.1, number_of_planes=20
        )
        tensor = official_resize_feature(feature, (128, 128)).numpy().astype(np.float32)
        tensors.append(tensor)
        details.append(
            {
                "window_us": [start, end],
                "duration_us": end - start,
                "square_xyxy": list(square),
                "raw_event_count": len(raw),
                "roi_event_count": int(counts.sum()),
                "roi_empty": int(counts.sum()) == 0,
            }
        )
    return np.stack(tensors), details, None


def _rgb(
    reader: SensorReader,
    indices: tuple[int, int],
    boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[np.ndarray, dict[str, Any]]:
    frames = [reader.read_rgb(index) for index in indices]
    normalized = np.concatenate(
        [
            (frame.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0) - IMAGENET_MEAN)
            / IMAGENET_STD
            for frame in frames
        ],
        axis=0,
    )
    square = official_square_box(list(boxes), 1)
    tensor = official_resize_roi(normalized, square, (128, 128)).numpy()
    result = tensor.reshape(2, 3, 128, 128).astype(np.float32, copy=False)
    clock = reader.rgb_timestamps_us
    return result, {
        "frame_indices": list(indices),
        "timestamps_us": [int(clock[index]) for index in indices],
        "square_xyxy": list(square),
    }


def _unavailable(variant: Variant, reason: str, details: dict[str, Any]) -> PreparedGarlQuery:
    return PreparedGarlQuery(variant, None, None, reason, details)


def prepare_query(
    reader: SensorReader,
    rows: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    calibration: Calibration,
    variant: Variant,
) -> PreparedGarlQuery:
    """Prepare one comparator independently without labels, interpolation or cadence relabeling."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown Garl variant: {variant}")
    query_us = int(row["anchor_us"])
    details: dict[str, Any] = {
        "variant": variant,
        "query_us": query_us,
        "available_us": query_us,
        "native_timevolume_range_s": 0.1,
        "native_timevolume_bins": 20,
        "native_bin_width_s": 0.005,
        "polarity_consumed": False,
    }

    if variant == "event":
        plan = _event_windows(query_us, reader.event_start_us)
        if plan is None or query_us > reader.event_end_us:
            return _unavailable(variant, "insufficient_real_event_history", details)
        windows, prefix = plan
        endpoints = [windows[0][1], windows[1][1]]
        event_boxes, _, sources = _endpoint_boxes(rows, endpoints, query_us, calibration)
        details.update(
            bbox_sources=sources,
            same_box_used_for_both_endpoints=sources[0]["row_index"] == sources[1]["row_index"],
        )
        event, event_details, reason = _events(reader, windows, event_boxes)
        details.update(event_windows=event_details, observed_prefix_adaptation=prefix)
        if reason is not None or event is None:
            return _unavailable(variant, reason or "event_preparation_unavailable", details)
        delta = (windows[1][1] - windows[0][1]) / 1e6
        sensor = event.reshape(40, 128, 128)
    else:
        indices = _rgb_indices(reader.rgb_timestamps_us, query_us)
        if indices is None:
            return _unavailable(variant, "fewer_than_two_real_causal_rgb_frames", details)
        clock = reader.rgb_timestamps_us
        endpoints = [int(clock[index]) for index in indices]
        event_boxes, rgb_boxes, sources = _endpoint_boxes(rows, endpoints, query_us, calibration)
        details.update(
            bbox_sources=sources,
            same_box_used_for_both_endpoints=sources[0]["row_index"] == sources[1]["row_index"],
        )
        rgb, rgb_details = _rgb(reader, indices, rgb_boxes)
        delta_us = int(endpoints[1]) - int(endpoints[0])
        if delta_us <= 0:
            return _unavailable(variant, "nonpositive_real_rgb_interval", details)
        delta = delta_us / 1e6
        details.update(rgb=rgb_details, observed_prefix_adaptation=False)
        if variant == "visual":
            sensor = rgb.reshape(6, 128, 128)
        else:
            support_us = min(delta_us, PERIOD_US)
            windows = [
                (max(reader.event_start_us, int(endpoints[0]) - support_us), int(endpoints[0])),
                (max(reader.event_start_us, int(endpoints[1]) - support_us), int(endpoints[1])),
            ]
            prefix = any(end - start != support_us for start, end in windows)
            event, event_details, reason = _events(reader, windows, event_boxes)
            details.update(
                event_windows=event_details,
                observed_prefix_adaptation=prefix,
                event_support_cap_us=PERIOD_US,
                event_support_us=support_us,
                actual_rgb_delta_minus_event_support_us=delta_us - support_us,
                event_support_ends_at_each_rgb_endpoint=True,
            )
            if reason is not None or event is None:
                return _unavailable(variant, reason or "event_preparation_unavailable", details)
            sensor = np.concatenate((rgb.reshape(6, 128, 128), event.reshape(40, 128, 128)))

    if delta <= 0 or sensor.dtype != np.float32 or not np.isfinite(sensor).all():
        raise ValueError("prepared Garl tensor violates finite positive-delta FP32 contract")
    details.update(
        endpoint_timestamps_us=[int(value) for value in endpoints],
        conversion_delta_t_s=delta,
        native_training_delta_t_s=0.1,
        variable_cadence_transfer=not np.isclose(delta, 0.1, rtol=0.0, atol=1e-9),
        sensor_shape=list(sensor.shape),
        sensor_sha256=hashlib.sha256(np.ascontiguousarray(sensor).tobytes()).hexdigest(),
    )
    return PreparedGarlQuery(variant, sensor, delta, "available", details)


__all__ = ["POLICY", "PreparedGarlQuery", "VARIANTS", "prepare_query"]
