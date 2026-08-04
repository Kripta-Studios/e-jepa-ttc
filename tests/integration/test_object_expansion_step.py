from __future__ import annotations

import torch

from e_jepa_ttc.data.garlttc_object_lhr import ObjectLHRBatch
from e_jepa_ttc.models.object_expansion import (
    ObjectCentricExpansionTTC,
    ObjectExpansionConfig,
)
from e_jepa_ttc.training.object_expansion import (
    ObjectExpansionLossConfig,
    object_expansion_loss,
)


def test_object_expansion_one_optimizer_step() -> None:
    model = ObjectCentricExpansionTTC(
        ObjectExpansionConfig(
            embed_dim=32,
            heads=4,
            spatial_window=2,
            spatial_depth=1,
            temporal_depth=1,
            query_count=2,
            adapter_hidden_dim=32,
            pair_hidden_dim=32,
            memory_budget_gb=1.0,
        )
    )
    batch = ObjectLHRBatch(
        events=torch.randn(2, 2, 21, 64, 64),
        delta_t_s=torch.tensor([0.1, 0.1]),
        visible_heights_px=torch.tensor([[50.0, 52.0], [50.0, 49.0]]),
        target_ttc_s=torch.tensor([2.0, -4.0]),
        masks=torch.zeros(2, 2, 1, 64, 64),
        mask_valid=torch.zeros(2, 2, dtype=torch.bool),
        sequence_ids=["a", "b"],
        sample_tokens=["s1", "s2"],
        track_ids=["t1", "t2"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    before = model.log_magnitude_head.weight.detach().clone()
    output = model(**batch.model_inputs())
    loss = object_expansion_loss(
        output,
        batch,
        epoch=6,
        config=ObjectExpansionLossConfig(),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    optimizer.step()
    after = model.log_magnitude_head.weight.detach()
    assert torch.isfinite(loss.total)
    assert not torch.equal(before, after)
