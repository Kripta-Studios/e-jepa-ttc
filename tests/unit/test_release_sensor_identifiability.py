"""Stream comparisons must distinguish event content from simultaneous packet ordering."""

import numpy as np

from scripts.release_sensor_identifiability import stream_comparison


def test_simultaneous_event_order_does_not_create_false_physical_difference():
    left = np.array([[0, 0.1, 1], [1, 0.1, -1], [0, 0.2, 1]], float)
    right = left[[1, 0, 2]].copy()
    right[0, 1] += 2e-15
    assert stream_comparison(left, right)["same_stream_at_1ps"]


def test_equal_count_does_not_hide_different_timing_or_polarity():
    left = np.array([[0, 0.1, 1], [1, 0.1, -1], [0, 0.2, 1]], float)
    later = left.copy()
    later[0, 1] += 0.001
    assert not stream_comparison(left, later)["same_stream_at_1ps"]
    changed = left.copy()
    changed[0, 2] = -1
    assert not stream_comparison(left, changed)["same_stream_at_1ps"]
    assert not stream_comparison(left, left[:1])["same_stream_at_1ps"]
