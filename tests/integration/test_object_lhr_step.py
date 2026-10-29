from __future__ import annotations

import torch

from e_jepa_ttc.data.garlttc_object_lhr import ObjectLHRBatch
from e_jepa_ttc.models.object_lhr import ObjectCentricLHR, ObjectCentricLHRConfig
from e_jepa_ttc.training.object_lhr import ObjectLHRCurriculumConfig, object_lhr_loss


def test_object_lhr_optimizer_step_is_finite_and_updates_height_head() -> None:
    torch.manual_seed(29)
    model = ObjectCentricLHR(
        ObjectCentricLHRConfig(
            in_channels=2,
            embed_dim=8,
            patch_size=4,
            spatial_window=2,
            heads=2,
            spatial_depth=1,
            temporal_depth=1,
            query_count=2,
            head_hidden_dim=8,
            mask_decoder=False,
            mask_size=16,
        )
    )
    batch = ObjectLHRBatch(
        events=torch.randn(4, 2, 2, 16, 16),
        delta_t_s=torch.full((4,), 0.1),
        visible_heights_px=torch.tensor(
            [[80.0, 84.0], [90.0, 99.0], [100.0, 95.0], [110.0, 100.0]]
        ),
        target_ttc_s=torch.tensor([2.0, 1.1, -2.0, -1.0]),
        masks=torch.zeros(4, 2, 1, 16, 16),
        mask_valid=torch.zeros(4, 2, dtype=torch.bool),
        sequence_ids=["s"] * 4,
        sample_tokens=[f"x{index}" for index in range(4)],
        track_ids=["t"] * 4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.height_head[-1].weight.detach().clone()
    output = model(**batch.model_inputs())
    loss = object_lhr_loss(
        output,
        batch,
        epoch=1,
        config=ObjectLHRCurriculumConfig(height_only_epochs=5, ratio_warmup_epochs=10),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    optimizer.step()
    assert torch.isfinite(loss.total)
    assert not torch.equal(before, model.height_head[-1].weight.detach())
