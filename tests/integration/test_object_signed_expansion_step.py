from __future__ import annotations

import torch

from e_jepa_ttc.data.object_signed_expansion import collate_object_signed_expansion
from e_jepa_ttc.models.object_signed_expansion import (
    ObjectCentricSignedExpansionTTC,
    ObjectSignedExpansionConfig,
)
from e_jepa_ttc.training.object_signed_expansion import (
    ObjectSignedExpansionLossConfig,
    object_signed_expansion_loss,
)


def test_signed_expansion_optimizer_step_changes_prediction() -> None:
    records = []
    for index, ttc in enumerate((2.0, -2.0, 4.0, -4.0)):
        motion = torch.zeros(18)
        motion[7] = 0.05 if ttc > 0 else -0.05
        records.append(
            {
                "jepa_event_roi": torch.randn(2, 3, 32, 32),
                "garl_delta_t_s": 0.1,
                "observable_motion": motion,
                "jepa_context_motion": torch.zeros(18),
                "precontext_motion_valid": index % 2 == 0,
                    "jepa_pair_valid": True,
                "garl_visible_heights_px": torch.tensor(
                    [90.0, 94.0] if ttc > 0 else [94.0, 90.0]
                ),
                "ttc_s": ttc,
                "sequence_id": f"s{index % 2}",
                "sample_token": str(index),
                "track_id": str(index),
            }
        )
    batch = collate_object_signed_expansion(records, event_field="jepa_event_roi")
    model = ObjectCentricSignedExpansionTTC(
        ObjectSignedExpansionConfig(
            in_channels=3,
            embed_dim=24,
            patch_size=8,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            query_count=2,
            adapter_hidden_dim=24,
            motion_hidden_dim=16,
            predictor_hidden_dim=32,
            pair_hidden_dim=32,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    before = model(**batch.model_inputs()).signed_expansion.detach().clone()
    output = model(**batch.model_inputs())
    loss = object_signed_expansion_loss(
        output,
        batch,
        epoch=4,
        config=ObjectSignedExpansionLossConfig(),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    optimizer.step()
    after = model(**batch.model_inputs()).signed_expansion.detach()
    assert torch.isfinite(after).all()
    assert not torch.allclose(before, after)
