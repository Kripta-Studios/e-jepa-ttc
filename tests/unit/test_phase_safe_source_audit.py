"""Causal source histories retain observed geometry and reject ambiguous clocks."""

import numpy as np
import pytest

from scripts.build_phase_safe_source_audit import causal_history


def row():
    return {
        "query_time_us": 1_000_000,
        "cutoff_us": 1_000_000,
        "producer_times_us": [900_000, 1_000_000],
        "producer_available_us": [900_000, 1_000_000],
        "producer_boxes_xyxy": [[0, 0, 10, 10], [0, 0, 11, 11]],
        "observation_times_us": [[300_000, 800_000, 900_000, 1_100_000]],
        "observation_available_us": [[300_000, 1_000_001, 900_000, 1_100_000]],
        "observation_boxes_xyxy": [[[0, 0, 8, 8], [0, 0, 9, 9], [0, 0, 10, 10], [0, 0, 12, 12]]],
    }


def test_history_filters_future_old_and_unavailable_and_deduplicates():
    boxes, clocks = causal_history(row())
    np.testing.assert_array_equal(clocks, [900_000, 1_000_000])
    np.testing.assert_array_equal(boxes, [[0, 0, 10, 10], [0, 0, 11, 11]])


def test_conflicting_same_timestamp_boxes_fail():
    value = row()
    value["observation_boxes_xyxy"][0][2] = [0, 0, 99, 99]
    with pytest.raises(ValueError, match="Conflicting"):
        causal_history(value)


def test_clock_shift_preserves_geometry():
    from operational.phase_startup.geometry import startup_geometry

    boxes, clocks = causal_history(row())
    original = startup_geometry(boxes, clocks, 1_000_000)
    shifted = startup_geometry(boxes, clocks + 12_345_678, 13_345_678)
    assert original == shifted
    assert original["valid"]
    assert original["ttc"] == pytest.approx(1.0)


def test_availability_cannot_precede_observation():
    value = row()
    value["producer_available_us"][0] = 899_999
    with pytest.raises(ValueError, match="Availability"):
        causal_history(value)
