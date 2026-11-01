"""Physical units and masking for the exploratory available-data sensor proxy."""

import numpy as np
import pytest

from scripts.release_sensor_proxy import (
    count_cells,
    fit_nonnegative_line,
    project_event_points,
    sample_bilinear,
    texture_motion,
)
from scripts.release_sensor_reference import observed_change


def calibration(rotation=None):
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = rotation
    return {
        "event_intrinsics": np.array([10.0, 10.0, 20.0, 20.0]),
        "rgb_intrinsics": np.array([10.0, 10.0, 20.0, 20.0]),
        "event_distortion": np.zeros(5),
        "rgb_distortion": np.zeros(5),
        "event_from_rgb": transform,
    }


def test_projection_uses_inverse_rotation_and_ignores_depth_translation():
    angle = 0.1
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    calib = calibration(rotation)
    calib["event_from_rgb"][:3, 3] = [1, 2, 3]
    point, valid = project_event_points(np.array([[20.0, 20.0]]), calib)
    assert valid.tolist() == [True]
    assert point[0, 0] == pytest.approx(20 - 10 * np.tan(angle))
    assert point[0, 1] == pytest.approx(20)


def test_bilinear_validity_does_not_invent_out_of_frame_samples():
    yy, xx = np.mgrid[:4, :5]
    value, valid = sample_bilinear(
        2 * xx + 3 * yy, np.array([[1.5, 1.25], [-1.0, 1.0], [np.nan, 0.0]])
    )
    assert value[0] == pytest.approx(6.75)
    assert valid.tolist() == [True, False, False]


def test_log_texture_radial_term_has_sensor_pixel_units_and_sign():
    yy, xx = np.mgrid[:64, :64]
    frame = 40 * np.exp(0.01 * xx + 0.005 * yy)
    edges = np.array([10, 20, 30])
    motion, coverage = texture_motion(frame, calibration(), edges, edges)
    # Each cell mean is (-5,-5), (+5,-5), (-5,+5), (+5,+5) about the ROI center.
    np.testing.assert_allclose(motion, [0.075, -0.025, 0.025, -0.075], atol=1e-10)
    np.testing.assert_array_equal(coverage, np.ones(4))


def test_signed_counts_keep_half_open_space_and_time():
    events = np.array([[0, 0, 10, 1], [1, 1, 11, 0], [2, 1, 12, 1], [4, 0, 13, 1]])
    positive, negative = count_cells(events, np.array([0, 2, 4]), np.array([0, 2]), 10, 20)
    np.testing.assert_array_equal(positive, [1, 1])
    np.testing.assert_array_equal(negative, [1, 0])
    with pytest.raises(ValueError, match="Noncausal"):
        count_cells(events, np.array([0, 4]), np.array([0, 2]), 10, 13)


def test_weighted_photometric_fit_recovers_scale_and_rejects_negative_contrast():
    x = np.array([-2.0, 0.0, 1.0, 5.0])
    weight = np.array([1.0, 3.0, 5.0, 7.0])
    assert fit_nonnegative_line(x, 3 * x + 2, weight) == pytest.approx((3, 2))
    slope, intercept = fit_nonnegative_line(x, -3 * x + 2, weight)
    assert slope == 0
    assert intercept == pytest.approx(np.average(-3 * x + 2, weights=weight))
    with pytest.raises(ValueError):
        fit_nonnegative_line(x, x, np.zeros(4))


def test_observed_brightness_control_uses_actual_interval_and_common_mask():
    yy, xx = np.mgrid[:64, :64]
    reference = 20 * np.exp(0.01 * xx + 0.005 * yy)
    edges = np.array([10, 20, 30])
    rate, old_coverage, common = observed_change(
        reference, 2 * reference, calibration(), edges, edges, 0.2
    )
    np.testing.assert_allclose(rate, np.full(4, np.log(2) / 0.2), atol=1e-12)
    np.testing.assert_array_equal(old_coverage, np.ones(4))
    np.testing.assert_array_equal(common, np.ones(4))
    _, old_coverage, common = observed_change(
        reference, np.full_like(reference, 255), calibration(), edges, edges, 0.2
    )
    np.testing.assert_array_equal(old_coverage, np.ones(4))
    np.testing.assert_array_equal(common, np.zeros(4))
    with pytest.raises(ValueError):
        observed_change(reference, reference, calibration(), edges, edges, 0)
