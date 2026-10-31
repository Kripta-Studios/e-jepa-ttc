import numpy as np
import pytest

from scripts.audit_phase_startup import pinhole_residual


def test_known_translation_pixel_residual_scales_with_inverse_depth():
    rays = np.array([[0.0, 0.0, 1.0], [0.3, 0.2, 1.0]])
    rotation = np.eye(3)
    translation = np.array([0.02, 0.0, 0.0])
    focal = np.array([1000.0, 1000.0])
    np.testing.assert_allclose(
        pinhole_residual(rays, rotation, translation, focal, 1), [[20, 0], [20, 0]]
    )
    np.testing.assert_allclose(
        pinhole_residual(rays, rotation, translation, focal, 2), [[10, 0], [10, 0]]
    )
    np.testing.assert_array_equal(
        pinhole_residual(rays, rotation, np.zeros(3), focal, 1), np.zeros((2, 2))
    )
    with pytest.raises(ValueError):
        pinhole_residual(rays, rotation, translation, focal, -1)
