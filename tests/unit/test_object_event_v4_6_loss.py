from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_6 import ObjectEventTTCV46, ObjectEventV46Config
from e_jepa_ttc.training.object_event_v4_6 import (
    ObjectEventV46LossConfig,
    boxes_to_feature_masks,
    object_event_v4_6_loss,
    official_range_weights,
)


def _model() -> ObjectEventTTCV46:
    return ObjectEventTTCV46(
        ObjectEventV41Config(
            input_size=32,
            stem_dim=16,
            embed_dim=16,
            spatial_grid=2,
            encoded_hidden_dim=32,
            activity_hidden_dim=16,
        ),
        ObjectEventV46Config(foreground_hidden_dim=16, scale_hidden_dim=16),
    )


def test_boxes_to_feature_masks_clips_and_rasterises() -> None:
    boxes = torch.tensor(
        [[[0.0, 0.0, 8.0, 8.0], [8.0, 8.0, 24.0, 24.0], [-4.0, 4.0, 12.0, 20.0]]]
    )
    masks = boxes_to_feature_masks(
        boxes,
        source_height=32,
        source_width=32,
        target_height=8,
        target_width=8,
    )
    assert masks.shape == (1, 2, 8, 8)
    assert masks[0, 0].sum().item() == 16.0
    assert masks[0, 1].sum().item() == 12.0


def test_official_range_weights_cover_all_ranges() -> None:
    config = ObjectEventV46LossConfig()
    target = torch.tensor([0.5, 2.0, 5.0, -2.0])
    weights = official_range_weights(target, config)
    assert weights.shape == target.shape
    assert torch.isfinite(weights).all()
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert weights[0] > weights[1] > weights[2]
    assert torch.isclose(weights[2], weights[3])


def test_v46_loss_is_finite_and_backpropagates() -> None:
    model = _model()
    model.freeze_base()
    events = torch.randn(4, 3, 12, 32, 32)
    output = model(events)
    loss = object_event_v4_6_loss(
        output,
        delta_t_s=torch.full((4,), 0.1),
        target_ttc_s=torch.tensor([1.0, 2.0, -3.0, 5.0]),
        visible_heights_px=torch.tensor(
            [[30.0, 33.0], [25.0, 26.0], [28.0, 27.0], [20.0, 21.0]]
        ),
        boxes_xyxy=torch.tensor(
            [
                [[4.0, 4.0, 20.0, 20.0], [5.0, 5.0, 21.0, 22.0], [6.0, 4.0, 23.0, 24.0]],
                [[3.0, 6.0, 19.0, 22.0], [4.0, 6.0, 20.0, 23.0], [5.0, 5.0, 21.0, 24.0]],
                [[5.0, 4.0, 22.0, 23.0], [6.0, 5.0, 23.0, 24.0], [5.0, 6.0, 22.0, 23.0]],
                [[7.0, 7.0, 20.0, 20.0], [7.0, 7.0, 21.0, 21.0], [8.0, 8.0, 22.0, 22.0]],
            ]
        ),
        source_height=32,
        source_width=32,
        config=ObjectEventV46LossConfig(),
    )
    assert torch.isfinite(loss.total)
    assert set(loss.components) == {
        "fused_mid",
        "fused_expansion",
        "fused_correlation",
        "height_ratio",
        "height_expansion",
        "foreground_bce",
        "foreground_dice",
        "sign",
        "blend",
    }
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.foreground_head.parameters())
