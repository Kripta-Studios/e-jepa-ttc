from __future__ import annotations

import torch

from e_jepa_ttc.data.garlttc_object_lhr import ObjectLHRBatch
from e_jepa_ttc.models.object_lhr import ObjectCentricLHROutput
from e_jepa_ttc.models.highres_factorized import HighResFeatures, make_patch_geometry
from e_jepa_ttc.training.object_lhr import (
    ObjectLHRCurriculumConfig,
    curriculum_phase,
    object_lhr_loss,
    target_log_ratio_from_ttc,
)


def _batch() -> ObjectLHRBatch:
    return ObjectLHRBatch(
        events=torch.zeros(2, 2, 3, 8, 8),
        delta_t_s=torch.tensor([0.1, 0.1]),
        visible_heights_px=torch.tensor([[90.0, 94.73684], [100.0, 90.90909]]),
        target_ttc_s=torch.tensor([2.0, -1.0]),
        masks=torch.zeros(2, 2, 1, 8, 8),
        mask_valid=torch.zeros(2, 2, dtype=torch.bool),
        sequence_ids=["a", "b"],
        sample_tokens=["a0", "b0"],
        track_ids=["ta", "tb"],
    )


def _output(batch: ObjectLHRBatch) -> ObjectCentricLHROutput:
    log_heights = torch.log(batch.visible_heights_px).clone().requires_grad_()
    log_ratio = log_heights[:, 0] - log_heights[:, 1]
    geometry = make_patch_geometry(8, 8, 4)
    features = HighResFeatures(
        tokens=torch.zeros(2, 2, 4, 4),
        valid_patch_mask=torch.ones(2, 2, 4, dtype=torch.bool),
        geometry=geometry,
        diagnostics={},
        encoded_grid_height=2,
        encoded_grid_width=2,
        post_merge_patch_coordinates=torch.zeros(4, 2),
    )
    return ObjectCentricLHROutput(
        log_visible_heights=log_heights,
        visible_heights_px=torch.exp(log_heights),
        log_height_ratio=log_ratio,
        height_ratio=torch.exp(log_ratio),
        ttc_mean_seconds=batch.target_ttc_s.clone(),
        direction_logits=torch.tensor([[5.0, -5.0], [-5.0, 5.0]], requires_grad=True),
        endpoint_embeddings=torch.zeros(2, 2, 4),
        pair_embedding=torch.zeros(2, 4),
        mask_logits=None,
        features=features,
    )


def test_curriculum_phase_boundaries() -> None:
    config = ObjectLHRCurriculumConfig(height_only_epochs=5, ratio_warmup_epochs=10)
    assert curriculum_phase(5, config) == "height_only"
    assert curriculum_phase(6, config) == "height_ratio"
    assert curriculum_phase(10, config) == "height_ratio"
    assert curriculum_phase(11, config) == "full_mid"


def test_full_loss_uses_ttc_aligned_log_ratio() -> None:
    batch = _batch()
    output = _output(batch)
    config = ObjectLHRCurriculumConfig(height_only_epochs=0, ratio_warmup_epochs=0)
    result = object_lhr_loss(output, batch, epoch=1, config=config)
    assert result.phase == "full_mid"
    assert "visible_height" in result.components
    assert "visible_ratio" in result.components
    assert "mid" in result.components
    assert torch.isfinite(result.total)
    result.total.backward()
    assert output.log_visible_heights.grad is not None


def test_target_ratio_supports_signed_ttc() -> None:
    value = target_log_ratio_from_ttc(
        torch.tensor([0.1, 0.1]),
        torch.tensor([2.0, -1.0]),
    )
    assert value[0] < 0
    assert value[1] > 0
