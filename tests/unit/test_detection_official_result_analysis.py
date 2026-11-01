"""Boundary checks for interpreting organizer-returned precision curves."""

from __future__ import annotations

import math

import pytest

from scripts.analyze_detection_official_result import reconstruct_ap


def test_recall_confined_to_first_ten_percent_has_zero_official_ap():
    assert reconstruct_ap([1.0] * 11 + [0.0] * 90, 0.1, 0.1) == 0


def test_perfect_curve_is_one_but_precision_floor_has_zero_ap():
    assert reconstruct_ap([1.0] * 101, 0.1, 0.1) == pytest.approx(1)
    assert reconstruct_ap([0.1] * 101, 0.1, 0.1) == 0


@pytest.mark.parametrize("curve", [[1.0] * 100, [math.nan] * 101, [1.1] * 101])
def test_invalid_returned_curves_are_rejected(curve):
    with pytest.raises(ValueError):
        reconstruct_ap(curve, 0.1, 0.1)
