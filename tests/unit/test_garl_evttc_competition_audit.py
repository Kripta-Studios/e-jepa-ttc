"""Sensor-independent adversarial checks for the post-run audit itself."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "garl_audit",
    Path(__file__).resolve().parents[2] / "scripts/verify_garl_evttc_competition_run.py",
)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def full_fixture() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    clock = {"rgb_us": [0, 100001, 200002], "event_start_us": 0, "event_end_us": 300000}
    identities = [
        {
            "sequence_id": "fixture",
            "row_index": i,
            "index": i,
            "timestamp_text": str(t / 1e6),
            "anchor_us": t,
            "box_text": ["0", "0", "10", "10"],
        }
        for i, t in enumerate(clock["rgb_us"])
    ]
    endpoints = [100001, 200002]
    delta = 0.100001
    prediction = {
        "variant": "full",
        "status": "ok",
        "model_ttc": float(np.float32(delta) / np.float32(0.5)),
        "heights": [1.0, 2.0],
        "reason": "available",
        "native_delta_t_s": 0.1,
        "conversion_delta_t_s": delta,
        "diagnostics": {"height_t0": 1.0, "height_t1": 2.0, "height_ratio_denominator": 0.5},
    }
    sources = [
        {
            "endpoint_us": t,
            "row_index": i,
            "index": i,
            "anchor_us": t,
            "selection": "latest_at_or_before_endpoint",
            "fallback": False,
            "available_us": t,
            "raw_box": [0.0, 0.0, 10.0, 10.0],
            "event_box": [0.0, 0.0, 10.0, 10.0],
            "rgb_box": [0.0, 0.0, 10.0, 10.0],
        }
        for i, t in enumerate(endpoints, 1)
    ]
    detail = {
        "variant": "full",
        "query_us": 200002,
        "available_us": 200002,
        "native_timevolume_range_s": 0.1,
        "native_timevolume_bins": 20,
        "native_bin_width_s": 0.005,
        "polarity_consumed": False,
        "rgb": {"frame_indices": [1, 2], "timestamps_us": endpoints, "square_xyxy": [0, 0, 10, 10]},
        "event_support_cap_us": 100000,
        "event_support_us": 100000,
        "actual_rgb_delta_minus_event_support_us": 1,
        "event_support_ends_at_each_rgb_endpoint": True,
        "bbox_sources": sources,
        "same_box_used_for_both_endpoints": False,
        "observed_prefix_adaptation": False,
        "endpoint_timestamps_us": endpoints,
        "conversion_delta_t_s": delta,
        "native_training_delta_t_s": 0.1,
        "variable_cadence_transfer": True,
        "sensor_shape": [46, 128, 128],
        "sensor_sha256": "0" * 64,
        "event_windows": [
            {
                "window_us": [t - 100000, t],
                "duration_us": 100000,
                "square_xyxy": [0, 0, 10, 10],
                "raw_event_count": 1,
                "roi_event_count": 0,
                "roi_empty": True,
            }
            for t in endpoints
        ],
    }
    row = {
        "variant": "full",
        "identity": identities[-1],
        "prediction": prediction,
        "input_details": detail,
        "status": "ok",
        "reason": "available",
        "prediction_ttc_seconds": prediction["model_ttc"],
        "preprocess_seconds": 0.01,
        "model_wall_seconds": 0.02,
        "cost_time_seconds": 0.03,
    }
    return row, clock, identities, {"event_resolution": [1280, 720]}


@pytest.mark.parametrize(
    ("query", "start", "expected"),
    [
        (1, 0, None),
        (2, 0, [[0, 1], [1, 2]]),
        (5, 0, [[0, 2], [2, 5]]),
        (300000, 0, [[100000, 200000], [200000, 300000]]),
    ],
)
def test_real_event_plan(query: int, start: int, expected: list[list[int]] | None) -> None:
    assert audit.event_plan(query, start) == expected


def test_rgb_pair_and_tie_are_causal() -> None:
    assert audit.rgb_plan([0, 40082, 80164, 120246], 120246) == [1, 3]
    assert audit.rgb_plan([0, 50000, 150000], 149999) == [0, 1]
    assert audit.rgb_plan([0, 100000], 0) is None


def test_actual_full_gap_and_empty_roi_valid() -> None:
    result = audit.inspect_temporal(*full_fixture())
    assert result["full_gap_us"] == 1 and result["max_span_us"] == 200001
    assert result["zero_roi_windows"] == 2


@pytest.mark.parametrize(
    "kind",
    ["future_rgb", "wrong_rgb", "future_bbox", "false_fallback", "long_support", "wrong_gap"],
)
def test_temporal_tampering_rejected(kind: str) -> None:
    args = full_fixture()
    d = args[0]["input_details"]
    if kind == "future_rgb":
        d["rgb"]["timestamps_us"][1] += 1
    if kind == "wrong_rgb":
        d["rgb"]["frame_indices"][0] = 0
    if kind == "future_bbox":
        d["bbox_sources"][0]["anchor_us"] = 200002
    if kind == "false_fallback":
        d["bbox_sources"][0]["fallback"] = True
    if kind == "long_support":
        d["event_windows"][0]["window_us"][0] -= 1
    if kind == "wrong_gap":
        d["actual_rgb_delta_minus_event_support_us"] = 0
    with pytest.raises(audit.helpers.AuditError):
        audit.inspect_temporal(*args)


def test_negative_prediction_must_keep_sign() -> None:
    row, *_ = full_fixture()
    p = row["prediction"]
    p["heights"] = [3.0, 2.0]
    p["diagnostics"].update(height_t0=3.0, height_ratio_denominator=-0.5)
    p["model_ttc"] = -p["model_ttc"]
    row["prediction_ttc_seconds"] = p["model_ttc"]
    audit.inspect_prediction(row, 0.100001)
    row["prediction_ttc_seconds"] = abs(row["prediction_ttc_seconds"])
    with pytest.raises(audit.helpers.AuditError):
        audit.inspect_prediction(row, 0.100001)


def test_singular_prediction_remains_null() -> None:
    row, *_ = full_fixture()
    p = row["prediction"]
    p.update(
        heights=[2.0, 2.0],
        status="unavailable",
        model_ttc=None,
        reason="nonfinite_height_ratio_ttc",
    )
    p["diagnostics"].update(height_t0=2.0, height_ratio_denominator=0.0, raw_ttc="+inf")
    row.update(status="unavailable", prediction_ttc_seconds=None, reason=p["reason"])
    audit.inspect_prediction(row, 0.100001)
    row["prediction_ttc_seconds"] = 1.0
    with pytest.raises(audit.helpers.AuditError):
        audit.inspect_prediction(row, 0.100001)


def test_native_point_one_cannot_replace_actual_cadence() -> None:
    row, *_ = full_fixture()
    row["prediction"]["conversion_delta_t_s"] = 0.1
    with pytest.raises(audit.helpers.AuditError):
        audit.inspect_prediction(row, 0.100001)


def test_legitimate_bbox_fallback_is_marked_query_available() -> None:
    row, clock, identities, calibration = full_fixture()
    identities = identities[-1:]
    identities[0]["row_index"] = 0
    d = row["input_details"]
    d["same_box_used_for_both_endpoints"] = True
    for i, source in enumerate(d["bbox_sources"]):
        source.update(row_index=0, index=2, anchor_us=200002, available_us=200002)
        if i == 0:
            source.update(fallback=True, selection="earliest_query_causal_fallback")
    assert audit.inspect_temporal(row, clock, identities, calibration)["bbox_fallbacks"] == 1


def test_unavailable_preparation_never_gets_finite_prediction() -> None:
    row, *_ = full_fixture()
    row.update(
        prediction=None, status="unavailable", prediction_ttc_seconds=None, model_wall_seconds=0.0
    )
    audit.inspect_prediction(row, None)
    row["prediction_ttc_seconds"] = 1.0
    with pytest.raises(audit.helpers.AuditError):
        audit.inspect_prediction(row, None)
