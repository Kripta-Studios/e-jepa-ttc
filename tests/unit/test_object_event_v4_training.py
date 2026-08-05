from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4
from e_jepa_ttc.models.object_event_v4 import ObjectEventTTCV4, ObjectEventV4Config
from e_jepa_ttc.training.object_event_v4 import (
    ObjectEventV4LossConfig,
    ObjectEventV4ModalityConfig,
    apply_modality_dropout,
    object_event_v4_loss,
)


def _batch():
    records = []
    for index, ttc in enumerate((0.8, -1.5)):
        records.append(
            {
                "event_v4_common_roi": np.random.default_rng(index).normal(
                    size=(3, 12, 16, 16)
                ).astype(np.float32),
                "garl_delta_t_s": np.float32(0.1),
                "observable_motion": np.random.default_rng(index + 3).normal(
                    size=18
                ).astype(np.float32),
                "garl_visible_heights_px": np.asarray(
                    [20.0, 20.0 / (1.0 - 0.1 / ttc)], dtype=np.float32
                ),
                "ttc_s": np.float32(ttc),
                "event_v4_boxes_xyxy": np.ones((3, 4), dtype=np.float32),
                "event_v4_common_square_xyxy": np.asarray(
                    [0, 0, 16, 16], dtype=np.float32
                ),
                "event_v4_precontext_valid": True,
                "sequence_id": "s",
                "sample_token": str(index),
                "track_id": "t",
            }
        )
    return collate_object_event_v4(records)


def _model() -> ObjectEventTTCV4:
    return ObjectEventTTCV4(
        ObjectEventV4Config(
            embed_dim=24,
            patch_size=4,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            query_count=2,
            predictor_hidden_dim=32,
            event_head_hidden_dim=32,
            motion_hidden_dim=16,
            fusion_hidden_dim=16,
            dropout=0.0,
        )
    )


def test_modality_dropout_never_drops_both_and_warmup_is_event_only() -> None:
    batch = _batch()
    warmup = apply_modality_dropout(
        batch.events,
        batch.observable_motion,
        epoch=1,
        config=ObjectEventV4ModalityConfig(event_only_warmup_epochs=3),
    )
    assert bool(warmup.motion_dropped.all())
    assert not bool(warmup.events_dropped.any())
    later = apply_modality_dropout(
        batch.events,
        batch.observable_motion,
        epoch=5,
        config=ObjectEventV4ModalityConfig(
            event_only_warmup_epochs=0,
            motion_drop_probability=0.5,
            event_drop_probability=0.4,
        ),
        generator=torch.Generator().manual_seed(7),
    )
    assert not bool((later.motion_dropped & later.events_dropped).any())


def test_event_supervision_backpropagates_into_encoder() -> None:
    torch.manual_seed(7)
    batch = _batch()
    model = _model().train()
    output = model(**batch.model_inputs())
    loss = object_event_v4_loss(output, batch, ObjectEventV4LossConfig())
    loss.total.backward()
    gradient = model.encoder.patch_embed.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0
