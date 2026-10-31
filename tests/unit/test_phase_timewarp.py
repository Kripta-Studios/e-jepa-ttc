"""Fixed CPU time-warp math and raw-event checks; tolerances precede execution."""

from dataclasses import replace

import numpy as np
import pytest

from e_jepa_ttc.data.eap_representation import event_voxel_with_scalars
from e_jepa_ttc.data.types import EventBatch
from operational.phase_timewarp.warp import (
    filter_retained,
    warp_clock,
    warp_metadata,
    warp_pixels,
)

# Preregistered before fixture execution: aligned voxel/count exact; FP32 rate2e-6.
# Rounded fixture voxel1e-3 absolute, count exact, FP32 rate2e-6 absolute.
ALIGNED_ATOL = np.array([0.0] * 11 + [2e-6])
ROUNDED_ATOL = np.array([1e-3] * 10 + [0.0, 2e-6])


def observation():
    return {
        "timestamps_us": [800000, 900000, 1000000],
        "windows_us": [[700000, 800000], [800000, 900000], [900000, 1000000]],
        "first_sensor_ts_us": 700000,
        "anchor_us": 1000000,
        "available_us": 1000100,
        "delta_t_s": [0.1, 0.1],
        "identity": "unchanged",
    }


def test_nearest_even_relative_to_query_and_large_integer_precision():
    assert [warp_clock(t, 10, 2.0) for t in (5, 7, 9, 10)] == [8, 8, 10, 10]
    q = 2**60
    assert warp_clock(q - 3, q, 2.0) == q - 2
    assert warp_clock(q - 3, q, 0.5) == q - 6
    for bad in (True, 1.5, np.nan):
        with pytest.raises(ValueError):
            warp_clock(bad, 10, 1.0)  # type: ignore[arg-type] -- intentional invalid input
    for bad in (True, 0.0, 3.0, np.nan):
        with pytest.raises(ValueError):
            warp_clock(5, 10, bad)
    with pytest.raises(ValueError, match="causal"):
        warp_clock(11, 10, 1.0)


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_metadata_and_target_ratio_invariance_without_real_targets(speed):
    old = observation()
    new = warp_metadata(old, 1100000, speed)
    assert new is not old and new["identity"] == old["identity"]
    np.testing.assert_array_equal(new["delta_t_s"], np.diff(new["timestamps_us"]) / 1e6)
    np.testing.assert_allclose(
        np.array(new["delta_t_s"]) / (2.0 / speed), np.array(old["delta_t_s"]) / 2.0, rtol=0, atol=0
    )
    assert new["windows_us"][0][0] == new["first_sensor_ts_us"]
    if speed == 1:
        assert new == old
    assert old == observation()


def test_metadata_rejects_collapsed_timestamps_and_misaligned_windows():
    old = observation()
    old["windows_us"][1][1] += 1
    with pytest.raises(ValueError):
        warp_metadata(old, 1100000, 2.0)
    old = {"timestamps_us": [6, 7], "first_sensor_ts_us": 6, "anchor_us": 7, "available_us": 10}
    with pytest.raises(ValueError, match="collapsed"):
        warp_metadata(old, 11, 2.0)


def test_filter_reports_removed_identities_and_never_invents_history():
    values = [warp_metadata(observation(), 1100000, 0.5)]
    retained, rejected = filter_retained(values, 1100000)
    assert retained == [] and rejected[0]["index"] == 0
    assert rejected[0]["reason"] == "history_budget"
    retained, rejected = filter_retained([observation()], 1100000)
    assert retained == [observation()] and rejected == []


def test_retention_cap_and_invalid_causal_metadata_are_explicit():
    items = [
        {
            "timestamps_us": [100 + i * 10, 105 + i * 10],
            "first_sensor_ts_us": 100 + i * 10,
            "anchor_us": 105 + i * 10,
            "available_us": 200,
            "identity": str(i),
        }
        for i in range(9)
    ]
    retained, discarded = filter_retained(items, 200)
    assert [item["identity"] for item in retained] == [str(i) for i in range(1, 9)]
    assert discarded[0]["index"] == 0 and discarded[0]["reason"] == "observation_limit"
    items[0]["available_us"] = 201
    with pytest.raises(ValueError, match="causal"):
        filter_retained(items, 200)
    with pytest.raises(ValueError):
        warp_clock(-(2**63), 1, 0.5)


def test_rgb_and_identity_rate_windows_are_bitwise_copies():
    rgb = np.arange(2 * 3 * 128 * 128, dtype=np.float32).reshape(2, 3, 128, 128)
    out = warp_pixels(rgb, "rgb", None, None)
    assert np.array_equal(rgb, out) and not np.shares_memory(rgb, out)
    event = np.zeros((3, 12, 128, 128), np.float32)
    event[:, 11] = 12.345
    windows = observation()["windows_us"]
    out = warp_pixels(event, "event", windows, windows)
    assert np.array_equal(event, out) and not np.shares_memory(event, out)
    with pytest.raises(ValueError):
        warp_pixels(event.astype(np.int32), "event", windows, windows)
    with pytest.raises(ValueError):
        warp_pixels(event[:, :11], "event", windows, windows)


def test_rate_uses_each_integer_window_duration_not_nominal_speed_or_pair_dt():
    pixels = np.zeros((2, 12, 128, 128), np.float32)
    pixels[:, 11] = np.log1p(100.0)
    old = [[0, 3], [4, 9]]
    new = [[5, 6], [7, 10]]
    value = warp_pixels(pixels, "event", old, new)
    np.testing.assert_allclose(np.expm1(value[:, 11, 0, 0]), [300.0, 100.0 * 5 / 3], rtol=1e-6)
    np.testing.assert_array_equal(value[:, :11], pixels[:, :11])


def test_quantization_can_change_half_open_membership_without_claiming_reencoding():
    # Event at old end-1 rounds onto new end. Real half-open re-encoding removes it;
    # the cached representation intentionally preserves channel10 and its membership.
    old = EventBatch(
        x=np.array([1, 2], np.int32),
        y=np.array([1, 2], np.int32),
        t_us=np.array([1000, 99999], np.int64),
        polarity=np.ones(2, np.int8),
        width=128,
        height=128,
        sequence_id="boundary",
        t_start_us=0,
        t_end_us=100000,
    )
    times = np.array([warp_clock(int(t), 200000, 2.0) for t in old.t_us])
    start, end = warp_clock(0, 200000, 2.0), warp_clock(100000, 200000, 2.0)
    keep = (times >= start) & (times < end)
    assert keep.tolist() == [True, False]
    new = replace(
        old,
        x=old.x[keep],
        y=old.y[keep],
        t_us=times[keep],
        polarity=old.polarity[keep],
        t_start_us=start,
        t_end_us=end,
    )
    pixels = event_voxel_with_scalars(old, bins_per_polarity=5).numpy()
    transformed = warp_pixels(pixels[None], "event", [[0, 100000]], [[start, end]])[0]
    direct = event_voxel_with_scalars(new, bins_per_polarity=5).numpy()
    assert transformed[10, 0, 0] == pixels[10, 0, 0] and transformed[10, 0, 0] != direct[10, 0, 0]
    print(
        {
            "boundary_dropped_events": int((~keep).sum()),
            "max_abs_error_by_channel": np.max(np.abs(transformed - direct), axis=(1, 2)).tolist(),
        }
    )


@pytest.mark.parametrize("rounded", [False, True])
@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_raw_event_encoder_alignment_and_quantization(rounded, speed):
    duration = 100001 if rounded else 100000
    times = np.arange(1000, 99001, 2000, dtype=np.int64) + (1 if rounded else 0)
    events = EventBatch(
        x=np.arange(len(times), dtype=np.int32),
        y=np.full(len(times), 64, np.int32),
        t_us=times,
        polarity=np.where(np.arange(len(times)) % 2, -1, 1).astype(np.int8),
        width=128,
        height=128,
        sequence_id="synthetic_timewarp",
        t_start_us=0,
        t_end_us=duration,
    )
    q = 200000
    start, end = warp_clock(0, q, speed), warp_clock(duration, q, speed)
    warped = replace(
        events,
        t_us=np.array([warp_clock(int(t), q, speed) for t in times]),
        t_start_us=start,
        t_end_us=end,
    )
    original = event_voxel_with_scalars(events, bins_per_polarity=5).numpy()
    direct = event_voxel_with_scalars(warped, bins_per_polarity=5).numpy()
    transformed = warp_pixels(original[None], "event", [[0, duration]], [[start, end]])[0]
    error = np.max(np.abs(transformed.astype(np.float64) - direct), axis=(1, 2))
    print({"rounded": rounded, "speed": speed, "max_abs_error_by_channel": error.tolist()})
    assert np.all(error <= (ROUNDED_ATOL if rounded else ALIGNED_ATOL))
    np.testing.assert_array_equal(transformed[:11], original[:11])
