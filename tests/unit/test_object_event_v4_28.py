import math

import pytest
import torch

from e_jepa_ttc.models.object_event_v4_28 import ObjectEventTTCV428, ObjectEventV428Config
from e_jepa_ttc.training.object_event_v4_28 import gaussian_scale_target, posterior_kl_loss


def test_config_rejects_unknown_matcher():
    with pytest.raises(ValueError):
        ObjectEventV428Config(matcher="free_regression")


def test_candidate_grid_identity_at_zero_scale_rotation_and_centers():
    center = torch.zeros(1, 2)
    grid = ObjectEventTTCV428._candidate_source_grid(
        height=3,
        width=3,
        previous_center=center,
        current_center=center,
        log_scales=torch.tensor([0.0]),
        rotations=torch.tensor([0.0]),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    expected_y, expected_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 3), torch.linspace(-1.0, 1.0, 3), indexing="ij"
    )
    expected = torch.stack((expected_x, expected_y), dim=-1)[None, None]
    torch.testing.assert_close(grid, expected)


def test_candidate_grid_scale_matches_height_ratio_convention():
    center = torch.zeros(1, 2)
    log_eta = math.log(0.5)
    grid = ObjectEventTTCV428._candidate_source_grid(
        height=3,
        width=3,
        previous_center=center,
        current_center=center,
        log_scales=torch.tensor([log_eta]),
        rotations=torch.tensor([0.0]),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    # The target bottom-right (+1,+1) samples source (+0.5,+0.5), matching
    # eta=h_prev/h_curr when the current object has doubled in image scale.
    torch.testing.assert_close(grid[0, 0, -1, -1], torch.tensor([0.5, 0.5]), atol=1.0e-6, rtol=0.0)


def test_gaussian_scale_target_is_normalized_and_peaks_near_target():
    candidates = torch.linspace(-0.2, 0.2, 41)
    target = torch.tensor([-0.073, 0.041])
    distribution = gaussian_scale_target(target, candidates, sigma=0.015)
    torch.testing.assert_close(distribution.sum(dim=-1), torch.ones(2), atol=1.0e-6, rtol=0.0)
    peaks = candidates[distribution.argmax(dim=-1)]
    assert torch.all((peaks - target).abs() <= 0.01)


def test_posterior_kl_prefers_correct_scale_peak():
    candidates = torch.linspace(-0.2, 0.2, 41)
    target = torch.tensor([-0.08])
    correct = -((candidates[None] - target[:, None]) / 0.015).square()
    wrong = -((candidates[None] - 0.08) / 0.015).square()
    correct_loss = posterior_kl_loss(correct, target, candidates, sigma=0.015, epsilon=1.0e-6)
    wrong_loss = posterior_kl_loss(wrong, target, candidates, sigma=0.015, epsilon=1.0e-6)
    assert correct_loss.item() < wrong_loss.item()


def test_event_weight_backward_is_finite_at_zero_activity():
    foreground = torch.tensor([[[0.0, 0.5], [1.0, 0.0]]], requires_grad=True)
    activity = torch.zeros_like(foreground, requires_grad=True)
    weight = ObjectEventTTCV428._event_weight(
        foreground,
        activity,
        foreground_floor=0.02,
        activity_floor=0.05,
        epsilon=1.0e-6,
    )
    assert torch.isfinite(weight).all()
    weight.sum().backward()
    assert torch.isfinite(foreground.grad).all()
    assert torch.isfinite(activity.grad).all()
