"""Prevent real-data scoring before synthetic recovery and preserve sensor clocks."""

import numpy as np
import pytest

from scripts.evaluate_phase_event_expansion_development import (
    require_synthetic_gates,
    undistort_inputs,
)


def test_failed_or_empty_synthetic_confirmation_cannot_unlock_real_scoring():
    for summary in (
        {},
        {"all_synthetic_gates_pass": True, "evaluation_total": 0},
        {
            "all_synthetic_gates_pass": True,
            "evaluation_total": 12,
            "evaluation_passed": 11,
            "real_data_authorized_by_this_result": True,
        },
    ):
        with pytest.raises(ValueError):
            require_synthetic_gates(summary)
    require_synthetic_gates(
        {
            "all_synthetic_gates_pass": True,
            "evaluation_total": 12,
            "evaluation_passed": 12,
            "real_data_authorized_by_this_result": True,
        }
    )
    # Permission comes from the user's ongoing research instruction, not this
    # provenance field, which the synthetic-only runner deliberately keeps false.
    require_synthetic_gates(
        {
            "all_synthetic_gates_pass": True,
            "evaluation_total": 12,
            "evaluation_passed": 12,
            "real_data_authorized_by_this_result": False,
        }
    )


def test_zero_distortion_preserves_event_positions_times_and_polarity():
    arrays = {
        "x": np.array([10, 20, 30]),
        "y": np.array([12, 20, 28]),
        "t_us": np.array([1, 2, 3]),
        "p": np.array([0, 1, 0]),
        "roi_xyxy": np.array([0.0, 0.0, 40.0, 40.0]),
    }
    calibration = {"event_intrinsics": [100, 100, 20, 20], "event_distortion": [0, 0, 0, 0]}
    events, roi = undistort_inputs(arrays, calibration)
    np.testing.assert_allclose(events[:, :2], np.column_stack((arrays["x"], arrays["y"])))
    np.testing.assert_array_equal(events[:, 2], arrays["t_us"])
    np.testing.assert_array_equal(events[:, 3], [-1, 1, -1])
    np.testing.assert_allclose(roi, arrays["roi_xyxy"])


def test_wide_slider_calibration_converges_without_relaxing_residual():
    from operational.evttc_competition.geometry import _distort_radtan

    calibration = {
        "event_intrinsics": [1846.74, 1845.35, 320.79, 237.97],
        "event_distortion": [0.991775, -46.198745, 0.0, 0.0],
    }
    arrays = {
        "x": np.array([0, 639]),
        "y": np.array([0, 479]),
        "t_us": np.array([1, 2]),
        "p": np.array([0, 1]),
        "roi_xyxy": np.array([0.0, 0.0, 640.0, 480.0]),
    }
    events, roi = undistort_inputs(arrays, calibration)
    assert np.isfinite(events).all() and np.isfinite(roi).all()
    focal = np.array(calibration["event_intrinsics"][:2])
    principal = np.array(calibration["event_intrinsics"][2:])
    projected = (
        _distort_radtan(
            (events[:, :2] - principal) / focal, np.array(calibration["event_distortion"])
        )
        * focal
        + principal
    )
    assert np.max(np.abs(projected - np.column_stack((arrays["x"], arrays["y"])))) <= 1e-3
