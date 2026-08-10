from __future__ import annotations

import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.models.local_transport import (
    LocalTransportMatch,
    local_correlation_match,
    transport_physical_features,
)


def test_local_correlation_recovers_known_integer_translation() -> None:
    torch.manual_seed(7)
    previous = torch.randn(1, 32, 12, 12)
    current = torch.zeros_like(previous)
    current[..., :, 1:] = previous[..., :, :-1]
    match = local_correlation_match(previous, current, radius=2, temperature=0.01)
    interior = match.dx[:, 2:-2, 2:-3]
    assert float(interior.mean()) > 0.9
    assert float(match.dy[:, 2:-2, 2:-2].abs().mean()) < 0.15
    assert torch.isfinite(match.entropy).all()
    assert bool(((match.entropy >= 0.0) & (match.entropy <= 1.0)).all())


def test_transport_summary_extracts_positive_axis_expansion() -> None:
    height = width = 15
    x = torch.linspace(-1.0, 1.0, width)
    y = torch.linspace(-1.0, 1.0, height)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    dx = 1.5 * xx[None]
    dy = 1.0 * yy[None]
    zeros = torch.zeros_like(dx)
    confidence = torch.ones_like(dx) * 0.2
    entropy = torch.ones_like(dx) * 0.2
    valid = torch.ones_like(dx, dtype=torch.bool)
    forward = LocalTransportMatch(dx, dy, confidence, entropy, valid)
    reverse = LocalTransportMatch(-dx, -dy, confidence, entropy, valid)
    features = transport_physical_features(
        forward,
        reverse,
        foreground_weight=torch.ones(1, 1, height, width),
        radius=4,
    )
    assert features.shape == (1, 18)
    assert float(features[0, 2]) > 0.05
    assert float(features[0, 3]) > 0.05
    assert float(features[0, 4]) > 0.05


def test_a5_transport_keeps_residual_bound_and_reversal_antisymmetry() -> None:
    torch.manual_seed(11)
    model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=2,
            hidden_dim=16,
            geometry_dim=24,
            residual_depth=1,
            dropout=0.0,
            transport_enabled=True,
            transport_radius=2,
            transport_temperature=0.05,
        )
    ).eval()
    inputs = torch.randn(2, 2, 2, 32, 32)
    delta = torch.full((2, 1), 0.1)
    with torch.inference_mode():
        forward = model(inputs, delta)
        reverse = model(inputs.flip(1), delta)
    assert forward.transport_tokens is not None
    assert forward.transport_raw_features is not None
    assert forward.transport_raw_features.shape == (2, 1, 18)
    assert torch.allclose(
        forward.residual_log_height_ratio,
        -reverse.residual_log_height_ratio,
        atol=1.0e-6,
        rtol=0.0,
    )
    assert bool(
        (forward.residual_log_height_ratio.abs() <= model.config.max_abs_log_ratio_residual).all()
    )


def test_transport_disabled_preserves_a4_parameter_contract() -> None:
    model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=12,
            hidden_dim=64,
            geometry_dim=128,
            residual_depth=2,
            dropout=0.05,
            foreground_decoder="resize_conv",
            foreground_fullres_dim=24,
            foreground_temporal_smoothing=0.15,
            temporal_inverse_ttc_blend=0.75,
        )
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 355118


def test_local_transport_is_differentiable_through_both_endpoints() -> None:
    torch.manual_seed(19)
    previous = torch.randn(2, 8, 10, 10, requires_grad=True)
    current = torch.randn(2, 8, 10, 10, requires_grad=True)
    match = local_correlation_match(previous, current, radius=2, temperature=0.07)
    objective = (
        match.dx.square().mean()
        + match.dy.square().mean()
        + 0.1 * match.entropy.mean()
        - 0.1 * match.confidence_margin.mean()
    )
    objective.backward()
    assert previous.grad is not None and torch.isfinite(previous.grad).all()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert float(previous.grad.abs().sum()) > 0.0
    assert float(current.grad.abs().sum()) > 0.0
