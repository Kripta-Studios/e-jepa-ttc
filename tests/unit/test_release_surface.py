"""Geometry controls independent of TTC labels or released checkpoints."""

import numpy as np
import pytest

from operational.release_research.surface_correspondence import (
    affine_surface,
    match_patch,
    persistent_surface,
)


def test_constant_images_do_not_fabricate_surface_motion():
    result = persistent_surface(np.zeros((3, 128, 128)), np.array([0, 100000, 200000]))
    assert result["persistent_patches"] == 0
    assert result["q_at_last_rgb"] is None


def test_translation_has_independent_composition_and_zero_expansion():
    image = np.random.default_rng(7).uniform(size=(128, 128))
    frames = np.stack([np.roll(image, (i, 2 * i), axis=(0, 1)) for i in range(3)])
    result = persistent_surface(frames, np.array([0, 100000, 200000]))
    assert result["persistent_patches"] == 49
    assert abs(result["q_at_last_rgb"]) < 1e-10
    assert abs(result["composition_logscale_residual"]) < 1e-10
    assert result["routes"]["ab"]["translation_pixels"] == pytest.approx([2, 1])
    assert all(r["c_direct"] == r["c_composed"] for r in result["matches"])


def test_affine_scale_uses_area_and_rejects_rank_deficient_geometry():
    points = np.array([(x, y) for x in (16, 32, 64) for y in (16, 64)])
    result = affine_surface(points, 1.1 * points + np.array([2, 5]))
    assert result["log_scale_change"] == pytest.approx(np.log(1.1))
    assert not affine_surface(np.ones((6, 2)), np.ones((6, 2)))["valid"]
    assert not affine_surface(points, points * np.array([-1, 1]))["valid"]


def test_occlusion_removes_correspondences_and_invalid_clock_rejected():
    image = np.random.default_rng(4).uniform(size=(128, 128))
    covered = image.copy()
    covered[:, 64:] = 0
    result = persistent_surface(np.stack([image, image, covered]), np.array([0, 40000, 80000]))
    assert 0 < result["persistent_patches"] < 49
    match = match_patch(image, covered, (112, 112), 8)
    assert match is not None
    assert match[1] == 0
    with pytest.raises(ValueError):
        persistent_surface(np.stack([image] * 3), np.array([0, 0, 80000]))
