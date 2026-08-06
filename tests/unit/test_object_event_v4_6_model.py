from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_6 import ObjectEventTTCV46, ObjectEventV46Config


def _model() -> ObjectEventTTCV46:
    base = ObjectEventV41Config(
        input_size=32,
        stem_dim=16,
        embed_dim=16,
        spatial_grid=2,
        encoded_hidden_dim=32,
        activity_hidden_dim=16,
        dropout=0.0,
    )
    geometry = ObjectEventV46Config(
        foreground_hidden_dim=16,
        scale_hidden_dim=16,
        maximum_blend=0.75,
    )
    model = ObjectEventTTCV46(base, geometry)
    model.freeze_base()
    return model


def test_v46_forward_is_event_only_and_finite() -> None:
    model = _model()
    events = torch.randn(4, 3, 12, 32, 32)
    output = model(events)
    assert output.expansion.shape == (4,)
    assert output.foreground_logits.shape[:2] == (4, 3)
    assert output.predicted_log_heights.shape == (4, 3)
    assert output.blend.shape == (4,)
    assert torch.isfinite(output.expansion).all()
    assert torch.isfinite(output.height_log_eta).all()
    assert bool((output.blend >= 0).all())
    assert bool((output.blend <= 0.75).all())


def test_v46_base_is_frozen_but_geometry_receives_gradients() -> None:
    model = _model()
    events = torch.randn(2, 3, 12, 32, 32)
    output = model(events)
    (output.expansion.square().mean() + output.height_log_eta.square().mean()).backward()
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert any(parameter.grad is not None for parameter in model.geometry_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.foreground_head.parameters())
