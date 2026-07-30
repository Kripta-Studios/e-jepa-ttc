from __future__ import annotations

import numpy as np

from e_jepa_ttc.geometry.rgb_foe import (
    affine_foe_xy,
    fit_affine_flow,
    ttc_from_affine_fit,
)


def _synthetic_affine(
    *,
    delta_t_s: float,
    ttc_s: float,
    translation: tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:80:4, 0:120:4]
    points = np.column_stack((x.reshape(-1), y.reshape(-1))).astype(np.float64)
    expansion = delta_t_s / ttc_s
    centered = points - np.asarray((60.0, 40.0))
    flow = np.column_stack(
        (
            translation[0] + expansion * centered[:, 0] - rotation * centered[:, 1],
            translation[1] + rotation * centered[:, 0] + expansion * centered[:, 1],
        )
    )
    return points, flow


def test_affine_divergence_recovers_known_ttc_with_translation_and_rotation() -> None:
    points, flow = _synthetic_affine(
        delta_t_s=0.1,
        ttc_s=2.0,
        translation=(3.0, -2.0),
        rotation=0.03,
    )
    fit = fit_affine_flow(points, flow)
    result = ttc_from_affine_fit(fit, delta_t_s=0.1)
    assert result.valid
    assert np.isclose(result.ttc_seconds, 2.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(fit.divergence_per_frame, 0.1, rtol=1e-6, atol=1e-6)
    assert fit.residual_rmse_px < 1e-9


def test_robust_affine_fit_rejects_large_outliers() -> None:
    points, flow = _synthetic_affine(delta_t_s=0.1, ttc_s=4.0)
    contaminated = flow.copy()
    contaminated[::17] += np.asarray((80.0, -60.0))
    fit = fit_affine_flow(points, contaminated, robust_iterations=8)
    result = ttc_from_affine_fit(fit, delta_t_s=0.1)
    assert result.valid
    assert np.isclose(result.ttc_seconds, 4.0, rtol=0.03)
    assert fit.inlier_fraction > 0.85


def test_contraction_does_not_produce_positive_collision_ttc() -> None:
    points, flow = _synthetic_affine(delta_t_s=0.1, ttc_s=-3.0)
    fit = fit_affine_flow(points, flow)
    result = ttc_from_affine_fit(fit, delta_t_s=0.1)
    assert not result.valid
    assert result.reason == "non_approaching_or_zero_divergence"
    assert np.isnan(result.ttc_seconds)


def test_affine_foe_recovers_expansion_center() -> None:
    points, flow = _synthetic_affine(delta_t_s=0.1, ttc_s=2.0)
    fit = fit_affine_flow(points, flow)
    foe = affine_foe_xy(fit.coefficients)
    assert foe is not None
    assert np.allclose(foe, (60.0, 40.0), atol=1e-6)
