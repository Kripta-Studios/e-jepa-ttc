"""Expansion identifiability, nuisance separation and robust photometric readout."""

import numpy as np
import pytest

from scripts.release_photometric_q import fit_functional, photometric_design
from scripts.release_sensor_proxy import texture_motion

TRACE = np.array([0.5, 0, 0, 0.5, 0, 0, 0])


def test_full_affine_readout_separates_translation_and_light():
    design = np.random.default_rng(7).normal(size=(100, 7))
    design[:, -1] = 1
    coefficient = np.array([0.3, -0.2, 0.1, 0.5, 2, -3, 4])
    result = fit_functional(design, design @ coefficient, TRACE)
    assert result["rank"] == 7
    assert result["q"] == pytest.approx(0.4, abs=1e-12)


def test_trace_is_estimable_without_identifying_every_affine_parameter():
    x, y = np.random.default_rng(8).normal(size=(2, 100))
    design = np.column_stack((-x * x, -x * y, -y * x, -y * y, -x, -y, np.ones(100)))
    coefficient = np.array([0.3, -0.2, 0.1, 0.5, 2, -3, 4])
    result = fit_functional(design, design @ coefficient, TRACE)
    assert result["rank"] == 6
    assert result["q"] == pytest.approx(0.4, abs=1e-12)


def test_one_gradient_direction_cannot_identify_expansion_trace():
    x, y = np.random.default_rng(8).normal(size=(2, 100))
    design = np.column_stack(
        (-x, -y, np.zeros((100, 2)), -np.ones(100), np.zeros(100), np.ones(100))
    )
    result = fit_functional(design, 0.4 * x + 2, TRACE)
    assert result["q"] is None
    assert result["reason"] == "q_not_observable"


def test_robust_readout_rejects_sparse_large_residuals():
    design = np.random.default_rng(9).normal(size=(200, 7))
    design[:, -1] = 1
    coefficient = np.array([0.3, -0.2, 0.1, 0.5, 2, -3, 4])
    target = design @ coefficient
    target[:5] += 100
    ordinary = TRACE @ np.linalg.lstsq(design, target, rcond=None)[0]
    robust = fit_functional(design, target, TRACE)["q"]
    assert abs(robust - 0.4) < 0.01 * abs(ordinary - 0.4)


def test_small_support_and_zero_functional_are_explicit():
    assert fit_functional(np.ones((31, 2)), np.ones(31), np.array([1.0, 0.0]))["q"] is None
    with pytest.raises(ValueError):
        fit_functional(np.ones((32, 2)), np.ones(32), np.zeros(2))


def test_design_preserves_cached_radial_term_coverage_and_sensor_units():
    yy, xx = np.mgrid[:64, :64]
    frame = 40 * np.exp(0.01 * xx + 0.005 * yy)
    calibration = {
        "event_intrinsics": np.array([10.0, 10.0, 20.0, 20.0]),
        "rgb_intrinsics": np.array([10.0, 10.0, 20.0, 20.0]),
        "event_distortion": np.zeros(5),
        "rgb_distortion": np.zeros(5),
        "event_from_rgb": np.eye(4),
    }
    edges = np.array([10, 20, 30])
    design, coverage = photometric_design(frame, calibration, edges, edges)
    radial, old_coverage = texture_motion(frame, calibration, edges, edges)
    np.testing.assert_array_equal(coverage, old_coverage)
    np.testing.assert_allclose(design[:, 0] + design[:, 3], radial, atol=1e-12)
    np.testing.assert_allclose(design[:, 4], -0.2, atol=1e-12)
    np.testing.assert_allclose(design[:, 5], -0.1, atol=1e-12)
