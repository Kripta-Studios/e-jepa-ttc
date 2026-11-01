"""Producer replay preserves batch grouping and explicit observation identity."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from operational.phase_timewarp.capture import bind_calls, capture_calls
from operational.phase_timewarp.pilot import finite_max_difference, replay_calls
from operational.rgb_port import transfer as engine
from scripts.pilot_phase_producer_timewarp import verify_sensor_identity


def metadata(anchor):
    return {
        "anchor_us": anchor,
        "available_us": 1_000_000,
        "first_sensor_ts_us": anchor - 200_000,
        "timestamps_us": [anchor - 100_000, anchor],
        "windows_us": [[anchor - 200_000, anchor - 100_000], [anchor - 100_000, anchor]],
    }


def test_nonfinite_stage_difference_remains_explicit_unknown():
    assert finite_max_difference([float("nan")], [1.0]) is None
    assert finite_max_difference([2.0], [1.0]) == 1.0


def test_sensor_identity_accepts_json_tuple_roundtrip_but_rejects_changed_box():
    from decimal import Decimal

    row = {"anchor_us": 123, "box_xyxy": (1, 2, 3, 4), "timestamp_s": Decimal("0.000123")}
    saved = {"anchor_us": 123, "box_xyxy": [1, 2, 3, 4], "timestamp_s": "0.000123"}
    verify_sensor_identity(row, saved)
    with pytest.raises(ValueError, match="identity"):
        verify_sensor_identity(row, {**saved, "box_xyxy": [1, 2, 3, 5]})


def test_capture_owns_pixels_and_explicit_batch_map():
    sensor = torch.ones(2, 2, 12, 128, 128)
    delta = torch.full((2, 1), 0.1)

    def fake(models, prefix, pixels, dt):
        return torch.ones(len(pixels), 17), torch.ones(len(pixels), 3)

    with patch.object(engine, "_expert_block", fake):
        with capture_calls({}, "cpu") as (calls, arrays):
            engine._expert_block({}, "E", sensor, delta)
    sensor.zero_()
    assert arrays["call0.pixels"].sum() > 0
    source = {"event": [metadata(900_000), metadata(1_000_000)], "rgb": []}
    bind_calls(calls, arrays, source, source)
    assert [v["batch_index"] for v in calls[0]["observations"]] == [0, 1]
    assert [v["observation_id"] for v in calls[0]["observations"]] == [
        "event:900000",
        "event:1000000",
    ]
    with pytest.raises(ValueError):
        bind_calls(calls, arrays, {"event": [metadata(900_000)], "rgb": []}, source)


def test_identity_replay_preserves_event_batch_and_original_dt():
    sensor = torch.ones(2, 2, 12, 128, 128)
    delta = torch.full((2, 1), 0.1)

    def fake(models, prefix, pixels, dt):
        raw = torch.ones(len(pixels), 17) * len(pixels)
        return raw, torch.ones(len(pixels), 3)

    source = {"event": [metadata(900_000), metadata(1_000_000)], "rgb": []}
    with patch.object(engine, "_expert_block", fake):
        with capture_calls({}, "cpu") as (calls, arrays):
            engine._expert_block({}, "E", sensor, delta)
        bind_calls(calls, arrays, source, source)
        result = replay_calls({}, calls, arrays, 1_000_000, 1.0, "cpu")
    np.testing.assert_array_equal(result["arrays"]["call0.pixels"], arrays["call0.pixels"])
    np.testing.assert_array_equal(result["arrays"]["call0.delta"], arrays["call0.delta"])
    np.testing.assert_array_equal(result["arrays"]["call0.raw17"], arrays["call0.raw17"])
    assert result["calls"][0]["sensor_shape"][0] == 2
    assert len(result["observations"]["event"]) == 2


def test_matched_subset_control_reexecutes_retained_batch_instead_of_slicing():
    sensor = torch.ones(2, 2, 12, 128, 128)
    delta = torch.full((2, 1), 0.1)

    def fake(models, prefix, pixels, dt):
        return torch.ones(len(pixels), 17) * len(pixels), torch.ones(len(pixels), 3)

    source = {"event": [metadata(800_000), metadata(1_000_000)], "rgb": []}
    with patch.object(engine, "_expert_block", fake):
        with capture_calls({}, "cpu") as (calls, arrays):
            engine._expert_block({}, "E", sensor, delta)
        bind_calls(calls, arrays, source, source)
        warped = replay_calls({}, calls, arrays, 1_000_000, 0.5, "cpu")
        control = replay_calls(
            {}, calls, arrays, 1_000_000, 1.0, "cpu", only_ids=set(warped["selected_ids"])
        )
    assert warped["selected_ids"] == control["selected_ids"] == ["event:1000000"]
    assert control["calls"][0]["sensor_shape"][0] == 1
    assert np.all(control["arrays"]["call0.raw17"] == 1)
    assert np.all(arrays["call0.raw17"] == 2)


def test_originally_rejected_observation_is_rerun_under_warp():
    sensor = torch.ones(1, 2, 12, 128, 128)
    delta = torch.full((1, 1), 0.1)

    def fake(models, prefix, pixels, dt):
        if dt[0, 0] > 0.075:
            raise ValueError("TTC lies outside the signed benchmark-phase domain")
        return torch.ones(len(pixels), 17), torch.ones(len(pixels), 3)

    source = {"event": [metadata(1_000_000)], "rgb": []}
    with patch.object(engine, "_expert_block", fake):
        with pytest.raises(ValueError), capture_calls({}, "cpu") as (calls, arrays):
            engine._expert_block({}, "E", sensor, delta)
        bind_calls(calls, arrays, source, {"event": [], "rgb": []})
        result = replay_calls({}, calls, arrays, 1_000_000, 2.0, "cpu")
    assert not calls[0]["observations"][0]["source_accepted"]
    assert result["accepted_ids"] == ["event:1000000"]
