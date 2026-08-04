from __future__ import annotations

from types import SimpleNamespace

import torch

from e_jepa_ttc.data.garlttc_object_lhr import ObjectLHRBatch
from e_jepa_ttc.training.object_expansion import (
    ObjectExpansionLossConfig,
    curriculum_phase,
    object_expansion_loss,
    targets_from_batch,
)


def _batch() -> ObjectLHRBatch:
    return ObjectLHRBatch(
        events=torch.randn(2, 2, 21, 32, 32),
        delta_t_s=torch.tensor([0.1, 0.1]),
        visible_heights_px=torch.tensor([[50.0, 52.0], [50.0, 49.0]]),
        target_ttc_s=torch.tensor([2.0, -4.0]),
        masks=torch.zeros(2, 2, 1, 32, 32),
        mask_valid=torch.zeros(2, 2, dtype=torch.bool),
        sequence_ids=["a", "b"],
        sample_tokens=["s1", "s2"],
        track_ids=["t1", "t2"],
    )


def test_targets_reconstruct_signed_inverse_and_ratio() -> None:
    batch = _batch()
    targets = targets_from_batch(batch)
    assert torch.allclose(
        targets.signed_inverse_ttc,
        batch.target_ttc_s.reciprocal(),
    )
    assert torch.allclose(
        targets.log_height_ratio,
        torch.log1p(-batch.delta_t_s / batch.target_ttc_s),
    )


def test_loss_is_finite_in_both_phases() -> None:
    batch = _batch()
    targets = targets_from_batch(batch)
    logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]], requires_grad=True)
    output = SimpleNamespace(
        log_abs_inverse_ttc=targets.log_abs_inverse_ttc.clone().requires_grad_(),
        direction_logits=logits,
        signed_inverse_ttc_soft=(
            targets.signed_inverse_ttc.clone().requires_grad_()
        ),
        log_height_ratio_soft=targets.log_height_ratio.clone().requires_grad_(),
        ttc_soft_seconds=batch.target_ttc_s.clone().requires_grad_(),
    )
    config = ObjectExpansionLossConfig(geometry_warmup_epochs=2)
    warmup = object_expansion_loss(output, batch, epoch=1, config=config)
    geometry = object_expansion_loss(output, batch, epoch=3, config=config)
    assert curriculum_phase(1, config) == "inverse_ttc_warmup"
    assert curriculum_phase(3, config) == "geometry_consistency"
    assert torch.isfinite(warmup.total)
    assert torch.isfinite(geometry.total)
    geometry.total.backward()
    assert logits.grad is not None
