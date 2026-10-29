from __future__ import annotations

import numpy as np

from e_jepa_ttc.geometry.strttc import (
    construct_strttc_system,
    inverse_ttc_at_endpoint,
    robust_linear_strttc,
    warp_strttc_events,
)


def test_strttc_linear_system_recovers_known_inverse_ttc() -> None:
    rng = np.random.default_rng(4)
    intrinsics = (500.0, 480.0, 320.0, 240.0)
    inverse_ttc = 0.4
    translation = np.array([0.02, -0.01])
    normalized = rng.uniform(-0.4, 0.4, size=(200, 2))
    pixels = np.column_stack(
        (
            normalized[:, 0] * intrinsics[0] + intrinsics[2],
            normalized[:, 1] * intrinsics[1] + intrinsics[3],
        )
    )
    normalized_flow = inverse_ttc * normalized + translation
    flow = normalized_flow * np.array(intrinsics[:2])
    events = np.column_stack((np.zeros(200), pixels))
    design, target = construct_strttc_system(events, 0.0, flow, intrinsics)

    result = robust_linear_strttc(
        design,
        target,
        iterations=32,
        squared_residual_threshold=1e-10,
        minimum_inlier_fraction=0.8,
        seed=2,
    )

    assert np.allclose(
        result.parameters,
        np.array([inverse_ttc, *translation]),
        atol=1e-8,
    )
    assert result.inlier_ratio == 1.0


def test_strttc_warp_matches_official_affine_model() -> None:
    intrinsics = (100.0, 100.0, 50.0, 50.0)
    coordinates = np.array([[60.0, 40.0]])
    warped = warp_strttc_events(
        coordinates,
        np.array([-0.1]),
        0.0,
        np.array([0.5, 0.0, 0.0]),
        intrinsics,
    )

    assert np.allclose(warped, np.array([[60.5, 39.5]]))


def test_inverse_ttc_transport_reaches_current_endpoint() -> None:
    # TTC=2 s at the reference becomes 1.8 s two tenths later.
    current = inverse_ttc_at_endpoint(0.5, 0.2)
    assert np.isclose(current, 1.0 / 1.8)
