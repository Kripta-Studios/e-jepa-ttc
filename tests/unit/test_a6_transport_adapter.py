from __future__ import annotations
import torch
import pytest
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig


def _config(**updates):
    values = dict(
        in_channels=12,
        hidden_dim=64,
        geometry_dim=128,
        residual_depth=2,
        dropout=0.05,
        foreground_decoder="resize_conv",
        foreground_fullres_dim=24,
        foreground_temporal_smoothing=0.15,
        transport_enabled=True,
        transport_radius=1,
        transport_temperature=0.02,
        transport_adapter_enabled=True,
        transport_adapter_depth=1,
    )
    values.update(updates)
    return CausalScaleTTCConfig(**values)


def test_adapter_requires_transport() -> None:
    with pytest.raises(ValueError, match="requires transport_enabled"):
        _config(transport_enabled=False)


def test_transport_adapter_is_exact_identity_at_initialization() -> None:
    model = CausalScaleTTC(_config())
    assert model.transport_adapter is not None
    features = torch.randn(2, 64, 32, 32)
    with torch.no_grad():
        adapted = model.transport_adapter(features)
    torch.testing.assert_close(adapted, features, rtol=0.0, atol=0.0)


def test_frozen_geometry_leaves_transport_adapter_trainable() -> None:
    model = CausalScaleTTC(_config())
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert model.transport_adapter is not None
    assert all(p.requires_grad for p in model.transport_adapter.parameters())


def test_adapter_receives_gradient_without_geometry_gradient() -> None:
    model = CausalScaleTTC(_config())
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.train()
    x = torch.randn(2, 3, 12, 128, 128)
    dt = torch.full((2, 2), 0.05)
    out = model(x, dt)
    loss = out.pair_log_height_ratio.square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.transport_adapter is not None
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.transport_adapter.parameters())
