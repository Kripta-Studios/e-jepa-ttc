"""Causal sensor preparation for the explicitly adapted competition protocol.

Weights and the 100 ms phase output mapping remain fixed. Short startup windows
and nearest actual RGB cadence are declared input adaptations, not resampling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from e_jepa_ttc.rgb_port.data import crop_uint8
from operational.evttc_competition.geometry import (
    Calibration,
    input_bbox_to_event_box,
    project_event_box_to_rgb,
)
from operational.evttc_competition.readers import SensorSequence
from operational.evttc_competition.runtime import PreparedObservation

PERIOD_US = 100_000
SPAN_US = 650_000
POLICY: dict[str, object] = {
    "version": "evttc_competition_causal_adapter_v2",
    "query_population": "every_supplied_bbox_csv_row_in_original_order",
    "output_clock": "original_bbox_csv_timestamp_text",
    "road_camera": "right_event_and_right_RGB",
    "bbox_plane": "event_raster_inferred_from_clock_and_sensor_overlays",
    "bbox_bounds": "zero_based_xyxy_half_open_explicit_clipping",
    "bbox_availability": "provided_annotation_available_at_its_RGB_frame_timestamp",
    "rgb_availability": "provided_RGB_frame_available_at_its_dataset_timestamp",
    "crop": "last_two_causal_boxes_common_square_margin_0.25_at_query",
    "single_box_startup": "reuse_same_box_for_crop_only_never_duplicate_sensor_frames",
    "event_to_rgb": "calibrated_rays_rotation_only_translation_omitted_no_depth",
    "event_mature": "three_100ms_windows_per_observation_up_to_four_100ms_history_anchors",
    "event_history_anchors": "fixed_query_minus_300_200_100_0_ms_independent_of_outputs",
    "event_startup": (
        "at_each_anchor_partition_available_prefix_if_under_300ms_into_three_real_windows"
    ),
    "event_support": "positive_duration_windows_inside_recorded_sensor_coverage",
    "event_empty_windows": "retain_observed_empty_windows_with_explicit_zero_support_diagnostics",
    "rgb_cadence": "nearest_real_predecessor_to_100ms_earlier_tie_prefers_earlier_index",
    "rgb_observations": "T3_sliding_triplets_or_T2_with_two_selected_real_startup_frames",
    "rgb_history": "at_most_eight_observations_whose_first_frame_is_within_650ms",
    "rgb_delta": "measured_frame_intervals_no_interpolation_or_nominal_relabeling",
    "event_representation": "canonical_float64_ROI_mapping_five_bins_per_polarity_plus_count_rate",
    "rgb_representation": "canonical_PIL_bilinear_ROI_uint8_then_float32_div255",
    "model": "unchanged_sealed_TRAIN40_nine_components_and_normalizers",
    "model_phase_anchor_seconds": 0.1,
    "information_span_us_max": SPAN_US,
    "outputs": "preserve_signed_TTC_no_new_clamp_or_absolute_value",
    "unrepresentable_observation": (
        "reject_whole_observation_only_on_exact_frozen_phase_domain_error_and_invalid_expert_TTC"
    ),
    "observation_admission": (
        "mathematical_phase_representability_independent_of_known_mask_or_accuracy"
    ),
    "infinite_expert_ttc": (
        "phase_limit_but_native_feature_nonfinite_fail_fast_not_observation_filtered"
    ),
    "missing_support": "retain_query_with_null_prediction_and_reason_no_constant_fill",
    "targets_used": False,
    "training_updates": 0,
}


@dataclass(frozen=True)
class PreparedQuery:
    """Real model observations and a serializable input audit for one CSV row."""

    events: tuple[PreparedObservation, ...]
    rgb: tuple[PreparedObservation, ...]
    details: dict[str, Any]


def event_window_plan(query_us: int, event_start_us: int) -> list[tuple[tuple[int, int], ...]]:
    """Plan observed half-open intervals without crossing the recording start."""
    result = []
    for lag in (300_000, 200_000, 100_000, 0):
        anchor = query_us - lag
        available = anchor - event_start_us
        if available < 3:
            continue
        if available < 3 * PERIOD_US:
            edges = [event_start_us + available * index // 3 for index in range(4)]
            result.append(tuple(zip(edges[:-1], edges[1:], strict=True)))
        else:
            result.append(
                tuple(
                    (anchor - (3 - index) * PERIOD_US, anchor - (2 - index) * PERIOD_US)
                    for index in range(3)
                )
            )
    return result


def rgb_frame_plan(clock: np.ndarray, query_us: int) -> list[tuple[int, ...]]:
    """Select actual T2/T3 frames; measured intervals may differ from 100 ms."""
    if clock.ndim != 1 or np.any(np.diff(clock) <= 0):
        raise ValueError("RGB timestamps must be a strictly increasing vector")
    last = int(np.searchsorted(clock, query_us, side="right")) - 1
    if last < 1:
        return []
    chain = [last]
    while len(chain) < 10 and chain[-1] > 0:
        target = int(clock[chain[-1]]) - PERIOD_US
        insertion = int(np.searchsorted(clock, target))
        candidates = [value for value in (insertion - 1, insertion) if 0 <= value < chain[-1]]
        if not candidates:
            break
        previous = min(candidates, key=lambda value: (abs(int(clock[value]) - target), value))
        if query_us - int(clock[previous]) > SPAN_US:
            break
        chain.append(previous)
    chain.reverse()
    if len(chain) == 2:
        return [tuple(chain)]
    return [tuple(chain[end - 2 : end + 1]) for end in range(2, len(chain))][-8:]


def _squares(
    rows: Sequence[Mapping[str, Any]],
    query_us: int,
    calibration: Calibration,
) -> tuple[tuple[float, ...], tuple[float, ...], list[dict[str, object]]]:
    anchors = [int(row["anchor_us"]) for row in rows]
    if len(anchors) != len(set(anchors)):
        raise ValueError("Duplicate bbox anchors require explicit track identity")
    causal = sorted(
        (row for row in rows if int(row["anchor_us"]) <= query_us),
        key=lambda row: int(row["anchor_us"]),
    )[-2:]
    if not causal:
        raise ValueError("No causal provided bbox at this query")
    boxes = [
        input_bbox_to_event_box(row["box_xyxy"], calibration, coordinate_plane="event")
        for row in causal
    ]
    rgb_boxes = [project_event_box_to_rgb(box, calibration) for box in boxes]
    indices = (0, len(boxes) - 1)
    event_square = common_square_from_boxes(boxes, indices, margin_fraction=0.25)
    rgb_square = common_square_from_boxes(rgb_boxes, indices, margin_fraction=0.25)
    sources: list[dict[str, object]] = [
        {
            "index": int(row["index"]),
            "anchor_us": int(row["anchor_us"]),
            "raw_box": list(row["box_xyxy"]),
            "event_box": list(box),
            "rgb_box": list(rgb_box),
            "bounds_clipped": tuple(row["box_xyxy"]) != tuple(box),
        }
        for row, box, rgb_box in zip(causal, boxes, rgb_boxes, strict=True)
    ]
    return event_square, rgb_square, sources


def prepare_query(
    reader: SensorSequence,
    rows: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    calibration: Calibration,
) -> PreparedQuery:
    """Decode causal inputs, retaining complete temporal and support provenance."""
    query_us = int(row["anchor_us"])
    event_square, rgb_square, bbox_sources = _squares(rows, query_us, calibration)
    windows = event_window_plan(query_us, reader.event_start_us)
    coverage_reason = "available"
    if query_us > reader.event_end_us:
        windows = []
        coverage_reason = "query_after_recorded_event_coverage"
    elif not windows:
        coverage_reason = "insufficient_recorded_event_duration"
    cached: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
    events = []
    event_details: list[dict[str, object]] = []
    for observation_windows in windows:
        for start, end in observation_windows:
            if (start, end) in cached:
                continue
            raw = reader.read_events(start, end)
            if len(raw) and (int(raw[:, 2].min()) < start or int(raw[:, 2].max()) >= end):
                raise ValueError("Reader violated half-open event interval")
            mapped = map_roi(
                dict(zip(("x", "y", "t", "p"), raw.T, strict=True)), event_square, 128, 0.0
            )
            pixels = encode_mapped(mapped, start, end, 128, reader.sequence_id).numpy()
            cached[start, end] = pixels, len(raw), len(mapped["t"])
        pixels = np.stack([cached[key][0] for key in observation_windows])
        counts = [cached[key][2] for key in observation_windows]
        zero_support = sum(counts) == 0
        endpoints = tuple(end for _, end in observation_windows)
        first = observation_windows[0][0]
        if (
            first < reader.event_start_us
            or endpoints[-1] > reader.event_end_us
            or query_us - first > SPAN_US
        ):
            raise ValueError("Event plan exceeds available causal history")
        events.append(
            PreparedObservation("event", pixels, endpoints, first, endpoints[-1], query_us)
        )
        event_details.append(
            {
                "windows_us": [list(key) for key in observation_windows],
                "raw_counts": [cached[key][1] for key in observation_windows],
                "roi_counts": counts,
                "delta_t_s": (np.diff(endpoints) / 1e6).tolist(),
                "first_sensor_ts_us": first,
                "anchor_us": endpoints[-1],
                "available_us": query_us,
                "admitted": True,
                "preparation_mode": (
                    "adaptive_observed_prefix"
                    if endpoints[-1] - first < 3 * PERIOD_US
                    else "native_100ms_windows"
                ),
                "zero_support": zero_support,
                "reason": "observed_empty_ROI" if zero_support else "observed_events_in_ROI",
            }
        )
    clock = reader.rgb_timestamps_us
    rgb_groups = rgb_frame_plan(clock, query_us)
    frame_cache = {
        index: crop_uint8(reader.read_rgb(index), rgb_square)
        for index in sorted({index for group in rgb_groups for index in group})
    }
    rgb = []
    rgb_details = []
    for group in rgb_groups:
        endpoints = tuple(int(clock[index]) for index in group)
        if endpoints[-1] > query_us or query_us - endpoints[0] > SPAN_US:
            raise ValueError("RGB plan exceeds available causal history")
        pixels = np.stack([frame_cache[index] for index in group]).astype(np.float32) / 255.0
        rgb.append(
            PreparedObservation("rgb", pixels, endpoints, endpoints[0], endpoints[-1], query_us)
        )
        rgb_details.append(
            {
                "frame_indices": list(group),
                "timestamps_us": list(endpoints),
                "delta_t_s": (np.diff(endpoints) / 1e6).tolist(),
                "first_sensor_ts_us": endpoints[0],
                "anchor_us": endpoints[-1],
                "available_us": query_us,
            }
        )
    return PreparedQuery(
        tuple(events),
        tuple(rgb),
        {
            "query_us": query_us,
            "event_recording_start_us": reader.event_start_us,
            "event_recording_end_us": reader.event_end_us,
            "event_coverage_reason": coverage_reason,
            "startup_mode": (
                "adaptive_observed_prefix"
                if any(
                    item["preparation_mode"] == "adaptive_observed_prefix" for item in event_details
                )
                else "native_100ms_windows"
            ),
            "event_square": list(event_square),
            "rgb_square": list(rgb_square),
            "bbox_sources": bbox_sources,
            "event_observations": event_details,
            "rgb_observations": rgb_details,
        },
    )
