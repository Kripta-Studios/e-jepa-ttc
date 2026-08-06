from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_7 import ObjectEventV47Config
from e_jepa_ttc.models.object_event_v4_8 import ObjectEventTTCV48, ObjectEventV48Config
from e_jepa_ttc.training.object_event_v4_8 import ObjectEventV48LossConfig, object_event_v4_8_loss


def test_v48_loss_is_finite_and_reaches_motion_head() -> None:
    model = ObjectEventTTCV48(
        ObjectEventV41Config(dropout=0.0),
        ObjectEventV47Config(mask_size=32, decoder_hidden_dim=32),
        ObjectEventV48Config(
            motion_hidden_dim=32,
            motion_refine_dim=32,
            field_size=32,
            freeze_foreground=True,
        ),
    )
    events = torch.randn(4, 3, 12, 64, 64)
    output = model(events)
    boxes = torch.tensor(
        [
            [[31.0, 21.0, 89.0, 99.0], [30.0, 20.0, 90.0, 100.0], [28.0, 18.0, 92.0, 104.0]],
            [[24.0, 24.0, 81.0, 91.0], [25.0, 25.0, 80.0, 90.0], [27.0, 27.0, 78.0, 88.0]],
            [[21.0, 19.0, 69.0, 90.0], [20.0, 18.0, 70.0, 92.0], [18.0, 16.0, 72.0, 96.0]],
            [[34.0, 23.0, 86.0, 95.0], [35.0, 24.0, 85.0, 94.0], [36.0, 25.0, 84.0, 92.0]],
        ]
    )
    heights = boxes[:, 1:3, 3] - boxes[:, 1:3, 1]
    loss = object_event_v4_8_loss(
        output,
        torch.full((4,), 0.05),
        torch.tensor([2.0, -2.5, 1.5, -3.0]),
        heights,
        boxes,
        source_height=128,
        source_width=128,
        config=ObjectEventV48LossConfig(),
    )
    assert torch.isfinite(loss.total)
    assert set(loss.components) == {
        "pooled_log_eta",
        "dense_log_eta",
        "expansion",
        "correlation",
        "sign",
        "confidence",
        "background_zero",
        "total_variation",
    }
    loss.total.backward()
    gradient = model.field_head[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0
