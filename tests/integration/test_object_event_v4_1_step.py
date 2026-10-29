from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventTTCV41, ObjectEventV41Config
from e_jepa_ttc.training.object_event_v4_1 import (
    ObjectEventV41LossConfig,
    object_event_v4_1_loss,
)


def test_v41_one_optimizer_step_changes_event_prediction() -> None:
    torch.manual_seed(23)
    model = ObjectEventTTCV41(
        ObjectEventV41Config(
            input_size=16,
            stem_dim=16,
            embed_dim=24,
            spatial_grid=2,
            encoded_hidden_dim=48,
            activity_hidden_dim=24,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    events = torch.randn(8, 3, 12, 16, 16)
    delta_t = torch.full((8,), 0.1)
    target_ttc = torch.tensor((0.6, 0.8, 1.2, 2.0, -0.7, -1.0, -1.5, -3.0))
    before = model(events).expansion.detach().clone()
    output = model(events)
    loss = object_event_v4_1_loss(
        output,
        delta_t,
        target_ttc,
        step=1,
        config=ObjectEventV41LossConfig(sign_temperature=0.5),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    optimizer.step()
    after = model(events).expansion.detach()
    assert torch.isfinite(loss.total)
    assert not torch.allclose(before, after)
