from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventTTCV41, ObjectEventV41Config
from e_jepa_ttc.training.object_event_v4_1 import (
    ObjectEventV41LossConfig,
    object_event_v4_1_loss,
)


def test_v41_learns_a_pure_temporal_event_signal() -> None:
    """Control: no boxes/motion exist and only temporal event activity carries sign."""

    torch.manual_seed(41)
    target = torch.tensor(
        (
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            -0.02,
            -0.03,
            -0.04,
            -0.05,
            -0.06,
            -0.07,
            -0.08,
            -0.09,
        )
    )
    events = torch.zeros(target.shape[0], 3, 12, 16, 16)
    for index, expansion in enumerate(target):
        magnitude = abs(float(expansion)) * 8.0
        direction = 1.0 if expansion > 0 else -1.0
        levels = (1.0 - direction * magnitude, 1.0, 1.0 + direction * magnitude)
        for step, level in enumerate(levels):
            events[index, step, 0] = level
            events[index, step, 5] = 2.0 - level
            events[index, step, 10] = 0.5 * level
            events[index, step, 11] = 0.25 * level

    delta_t = torch.full_like(target, 0.1)
    target_ttc = delta_t / target
    model = ObjectEventTTCV41(
        ObjectEventV41Config(
            input_size=16,
            stem_dim=8,
            embed_dim=8,
            spatial_grid=1,
            encoded_hidden_dim=32,
            activity_hidden_dim=16,
            dropout=0.0,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-3, weight_decay=0.0)
    loss_config = ObjectEventV41LossConfig(
        expansion_weight=8.0,
        encoded_aux_weight=0.1,
        activity_aux_weight=0.5,
        correlation_weight=0.5,
        ranking_weight=0.1,
        sign_weight=0.2,
        variance_weight=0.1,
        reversal_weight=0.0,
        sign_temperature=0.5,
    )

    for step in range(1, 31):
        output = model(events)
        loss = object_event_v4_1_loss(
            output,
            delta_t,
            target_ttc,
            step=step,
            config=loss_config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        optimizer.step()

    with torch.no_grad():
        prediction = model(events).expansion
    pearson = float(np.corrcoef(target.numpy(), prediction.numpy())[0, 1])
    positive_accuracy = float((prediction[target > 0] >= 0).float().mean())
    negative_accuracy = float((prediction[target < 0] < 0).float().mean())
    balanced_sign = 0.5 * (positive_accuracy + negative_accuracy)
    expansion_mae = float((prediction - target).abs().mean())

    assert pearson > 0.98
    assert balanced_sign == 1.0
    assert expansion_mae < 0.005
