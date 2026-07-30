from __future__ import annotations

import numpy as np

from e_jepa_ttc.evaluation.geometry_fusion import (
    GeometryFusionConfig,
    fuse_inverse_ttc,
    geometry_reliability,
)


def test_invalid_or_unstable_geometry_gets_zero_weight() -> None:
    confidence = geometry_reliability(
        valid=np.asarray([True, False, True]),
        inlier_fraction=np.asarray([0.8, 0.9, 0.8]),
        residual_rmse_px=np.asarray([0.0, 0.0, np.inf]),
        condition_number=np.asarray([1.0, 1.0, 1.0]),
        config=GeometryFusionConfig(),
    )
    assert np.allclose(confidence, [0.8, 0.0, 0.0])


def test_inverse_ttc_fusion_respects_endpoints_and_invalid_fallback() -> None:
    config = GeometryFusionConfig()
    neural = np.asarray([4.0, 4.0, 4.0])
    geometry = np.asarray([2.0, 2.0, np.nan])
    fused, weight = fuse_inverse_ttc(
        neural,
        geometry,
        np.asarray([0.0, 1.0, 1.0]),
        config=config,
    )
    assert np.allclose(fused, [4.0, 8.0 / 3.0, 4.0])
    # Full reliability is attenuated by disagreement, as declared by the gate.
    assert np.allclose(weight, [0.0, 0.5, 0.0])


def test_identical_estimates_preserve_ttc() -> None:
    neural = np.asarray([1.5, 3.0, 8.0])
    fused, _ = fuse_inverse_ttc(
        neural,
        neural,
        np.ones(3),
        config=GeometryFusionConfig(),
    )
    assert np.allclose(fused, neural)
