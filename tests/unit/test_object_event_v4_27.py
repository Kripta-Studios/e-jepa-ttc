import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_27 import ObjectEventTTCV427, ObjectEventV427Config
from e_jepa_ttc.training.object_event_v4_27 import balanced_sign_weights, target_log_height_ratio


def test_target_log_height_ratio_sign_matches_approach_geometry():
    heights = torch.tensor([[20.0, 25.0], [25.0, 20.0]])
    target = target_log_height_ratio(heights)
    assert target[0] < 0.0  # approaching: current object is taller
    assert target[1] > 0.0  # receding: current object is shorter


def test_balanced_sign_weights_equalize_total_mass():
    target = torch.tensor([-0.2, 0.1, 0.2, 0.3, 0.4])
    weights = balanced_sign_weights(target)
    negative_mass = weights[target < 0].sum()
    positive_mass = weights[target >= 0].sum()
    torch.testing.assert_close(negative_mass, positive_mass)


def test_balanced_sign_weights_are_finite_for_single_class_batch():
    weights = balanced_sign_weights(torch.tensor([0.1, 0.2, 0.3]))
    assert torch.isfinite(weights).all()
    assert weights.min() > 0.0


def test_scale_grid_is_odd_and_straddles_zero():
    cfg = ObjectEventV427Config(scale_bins=45)
    grid = torch.linspace(cfg.log_scale_min, cfg.log_scale_max, cfg.scale_bins)
    assert grid.shape == (45,)
    assert torch.isclose(grid[22], torch.tensor(0.0), atol=1.0e-7)


def test_vertical_warp_log_eta_convention_matches_height_ratio():
    height = 101
    y = torch.linspace(-1.0, 1.0, height)
    source = torch.exp(-((y / 0.25) ** 2))[None, None]
    source_weight = torch.ones(1, height)
    center = torch.zeros(1)
    log_eta = torch.tensor([-0.12])

    # Construct the current profile from the physical convention
    # eta = h_previous / h_current.  Negative log_eta means approach.
    ratio = torch.exp(log_eta)
    grid_y = ratio[:, None] * y[None]
    grid = torch.stack((torch.zeros_like(grid_y), grid_y), dim=-1).unsqueeze(2)
    current = functional.grid_sample(
        source.unsqueeze(-1), grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(-1)

    warped, _ = ObjectEventTTCV427._warp_profile(
        None, source, source_weight, center, center, log_eta  # type: ignore[arg-type]
    )
    torch.testing.assert_close(warped, current, atol=1.0e-6, rtol=1.0e-5)


def test_stable_overlap_has_finite_backward_at_zero_padding():
    warped = torch.tensor([[0.0, 0.25, 1.0]], requires_grad=True)
    current = torch.tensor([[0.5, 0.5, 0.5]], requires_grad=True)
    overlap = ObjectEventTTCV427._stable_overlap(warped, current)
    assert overlap[0, 0].item() == 0.0
    overlap.sum().backward()
    assert torch.isfinite(warped.grad).all()
    assert torch.isfinite(current.grad).all()


def test_legacy_sqrt_overlap_would_have_infinite_gradient_at_zero():
    product = torch.tensor([0.0, 0.25], requires_grad=True)
    legacy = torch.sqrt(product.clamp_min(0.0))
    legacy.sum().backward()
    assert torch.isinf(product.grad[0])
