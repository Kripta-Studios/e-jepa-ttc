from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4
from e_jepa_ttc.models.object_event_v4 import ObjectEventTTCV4, ObjectEventV4Config
from e_jepa_ttc.training.object_event_v4 import ObjectEventV4LossConfig, object_event_v4_loss


def test_one_optimizer_step_is_finite() -> None:
    records = []
    for index, ttc in enumerate((0.7, 1.4, -1.2, 3.0)):
        ratio = 1.0 - 0.1 / ttc
        records.append(
            {
                "event_v4_common_roi": np.random.default_rng(index).normal(
                    size=(3, 12, 16, 16)
                ).astype(np.float32),
                "garl_delta_t_s": np.float32(0.1),
                "observable_motion": np.random.default_rng(index + 10).normal(
                    size=18
                ).astype(np.float32),
                "garl_visible_heights_px": np.asarray([20.0, 20.0 / ratio], dtype=np.float32),
                "ttc_s": np.float32(ttc),
                "event_v4_boxes_xyxy": np.ones((3, 4), dtype=np.float32),
                "event_v4_common_square_xyxy": np.asarray([0, 0, 16, 16], dtype=np.float32),
                "event_v4_precontext_valid": True,
                "sequence_id": "s",
                "sample_token": str(index),
                "track_id": "t",
            }
        )
    batch = collate_object_event_v4(records)
    model = ObjectEventTTCV4(
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    output = model(**batch.model_inputs())
    loss = object_event_v4_loss(output, batch, ObjectEventV4LossConfig())
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    optimizer.step()
    assert torch.isfinite(loss.total)
    assert output.signed_expansion.shape == (4,)
    assert output.predicted_future_tokens.shape == output.target_future_tokens.shape
