from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_2 import ObjectEventTTCV42
from e_jepa_ttc.training.object_event_v4_5 import (
    ObjectEventV45LossConfig,
    object_event_v4_5_loss,
)


def test_v45_real_model_step_reaches_encoder_and_head() -> None:
    torch.manual_seed(45)
    model = ObjectEventTTCV42(
        ObjectEventV41Config(
            input_size=16,
            stem_dim=8,
            embed_dim=8,
            spatial_grid=2,
            encoded_hidden_dim=32,
            activity_hidden_dim=8,
            dropout=0.0,
        )
    )
    events = torch.randn(8, 3, 12, 16, 16)
    delta_t = torch.full((8,), 0.1)
    target_ttc = torch.tensor([2.0, 2.5, 4.0, 5.5, 7.0, 9.0, -3.0, -7.0])
    output = model(events)
    loss = object_event_v4_5_loss(
        output,
        delta_t,
        target_ttc,
        config=ObjectEventV45LossConfig(),
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    encoder_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    head_gradient = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.encoded_head.parameters()
        if parameter.grad is not None
    )
    assert encoder_gradient > 0.0
    assert head_gradient > 0.0
    assert all(parameter.grad is None for parameter in model.activity_head.parameters())
