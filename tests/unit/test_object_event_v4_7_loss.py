from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_7 import ObjectEventTTCV47, ObjectEventV47Config
from e_jepa_ttc.training.object_event_v4_7 import (
    ObjectEventV47LossConfig,
    object_event_v4_7_loss,
)


def test_v47_loss_is_finite_and_backpropagates() -> None:
    model = ObjectEventTTCV47(
        ObjectEventV41Config(dropout=0.0),
        ObjectEventV47Config(mask_size=64),
    )
    events = torch.randn(4, 3, 12, 64, 64)
    output = model(events)
    delta_t = torch.full((4,), 0.1)
    ttc = torch.tensor([1.0, 2.0, -2.0, -4.0])
    heights = torch.tensor([[20.0, 22.0], [30.0, 31.0], [25.0, 24.0], [40.0, 39.0]])
    boxes = torch.tensor(
        [
            [[20.0, 20.0, 80.0, 80.0], [20.0, 20.0, 80.0, 80.0], [20.0, 18.0, 80.0, 82.0]],
            [[16.0, 16.0, 90.0, 90.0], [16.0, 16.0, 90.0, 90.0], [16.0, 15.0, 90.0, 91.0]],
            [[24.0, 24.0, 76.0, 76.0], [24.0, 24.0, 76.0, 76.0], [24.0, 25.0, 76.0, 75.0]],
            [[12.0, 12.0, 96.0, 96.0], [12.0, 12.0, 96.0, 96.0], [12.0, 13.0, 96.0, 95.0]],
        ]
    )
    loss = object_event_v4_7_loss(
        output,
        delta_t,
        ttc,
        heights,
        boxes,
        source_height=128,
        source_width=128,
        config=ObjectEventV47LossConfig(),
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
