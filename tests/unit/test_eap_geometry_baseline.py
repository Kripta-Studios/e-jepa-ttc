from __future__ import annotations

import numpy as np

from e_jepa_ttc.baselines.eap_geometry import (
    depth_velocity_ttc,
    height_ratio_ttc,
    inverse_ttc_geometry_fusion,
)


def test_height_ratio_and_depth_velocity_recover_known_ttc() -> None:
    boxes = np.asarray(
        [
            [
                [[0.0, 0.0, 1.0, 0.1]],
                [[0.0, 0.0, 1.0, 0.15]],
                [[0.0, 0.0, 1.0, 0.2]],
            ]
        ]
    )
    timestamps = np.asarray([[100_000, 200_000, 300_000]])
    depth = np.asarray([[[11.0], [10.0], [9.0]]])

    height_ttc = height_ratio_ttc(boxes, timestamps)
    depth_ttc = depth_velocity_ttc(depth, timestamps)

    np.testing.assert_allclose(height_ttc, [[0.4]])
    np.testing.assert_allclose(depth_ttc, [[0.9]])


def test_geometry_fusion_uses_available_inverse_ttc_estimates() -> None:
    fused = inverse_ttc_geometry_fusion(
        np.asarray([2.0, np.nan, -4.0]),
        np.asarray([4.0, 3.0, -4.0]),
    )

    np.testing.assert_allclose(fused, [1.0 / 0.375, 3.0, -4.0])
