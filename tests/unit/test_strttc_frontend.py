from __future__ import annotations

import numpy as np

from e_jepa_ttc.geometry.strttc_frontend import (
    STRTTCFrontendConfig,
    estimate_plane_normal_flow,
    nearest_linear_time_surface,
)


def test_nearest_linear_time_surface_selects_event_closest_to_midpoint() -> None:
    events = np.array(
        [
            [0.0, 2, 3, -1],
            [0.4, 2, 3, -1],
            [0.19, 2, 3, -1],
            [0.2, 4, 5, 1],
        ],
        dtype=np.float64,
    )
    surface, valid, reference = nearest_linear_time_surface(
        events,
        width=8,
        height=8,
    )

    assert np.isclose(reference, 0.2)
    assert np.isclose(surface[3, 2], -0.01)
    assert valid[3, 2]
    assert not valid[5, 4]


def test_local_plane_fit_recovers_normal_flow() -> None:
    rows = []
    a = 0.02
    b = -0.01
    for y in range(4, 13):
        for x in range(4, 13):
            rows.append([a * x + b * y, x, y, -1])
    events = np.asarray(rows, dtype=np.float64)
    contour = np.array([[a * 8 + b * 8, 8, 8]], dtype=np.float64)
    config = STRTTCFrontendConfig(
        spatial_window_size=8,
        minimum_neighbour_fraction=0.2,
        plane_ransac_iterations=16,
        plane_squared_residual_threshold=1e-10,
        plane_minimum_inlier_fraction=0.8,
        flow_squared_threshold=1e-8,
    )

    points, flow = estimate_plane_normal_flow(
        events,
        contour,
        width=20,
        height=20,
        config=config,
    )

    expected = np.array([a, b]) / (a * a + b * b)
    assert points.shape == (1, 3)
    assert np.allclose(flow[0], expected)
