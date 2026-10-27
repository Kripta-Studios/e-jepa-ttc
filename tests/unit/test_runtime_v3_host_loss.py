from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from operational.final_refit.host_loss import prepare_effective_batch
from operational.rgb_port import train_producers as original
from operational.rgb_port.recipe import ProducerRecipe
from operational.rgb_port_revision.loss import EffectiveBatch, collect_loss_scalars, effective_loss


def _model() -> CausalScaleTTC:
    return CausalScaleTTC(
        CausalScaleTTCConfig(
            modality="rgb", in_channels=3, hidden_dim=16, geometry_dim=24,
            residual_depth=1, dropout=0.0,
            foreground_decoder="equivariant_separable", foreground_fullres_dim=8,
        )
    )


def _batch(steps: int, prefix: str, targets: torch.Tensor) -> ObjectEventV4Batch:
    size = len(targets)
    events = torch.linspace(0.0, 1.0, size * steps * 3 * 16 * 16).reshape(
        size, steps, 3, 16, 16
    )
    boxes = torch.tensor([2.0, 3.0, 12.0, 11.0]).repeat(size, steps, 1)
    return ObjectEventV4Batch(
        events=events,
        delta_t_s=torch.full((size, steps - 1), 0.1),
        observable_motion=torch.zeros(size, 18),
        visible_heights_px=torch.ones(size, max(1, steps - 1)),
        target_ttc_s=targets.float(), boxes_xyxy=boxes,
        common_square_xyxy=torch.tensor([0.0, 0.0, 16.0, 16.0]).repeat(size, 1),
        sequence_ids=[f"sequence-{prefix}-{index}" for index in range(size)],
        sample_tokens=[f"token-{prefix}-{index}" for index in range(size)],
        track_ids=[f"track-{prefix}-{index}" for index in range(size)],
    )


def _attach_relations(model: CausalScaleTTC, batch: ObjectEventV4Batch) -> None:
    delta = original._pair_intervals(
        batch.delta_t_s, batch_size=len(batch.target_ttc_s), steps=batch.events.shape[1]
    )
    with torch.no_grad():
        output = model(batch.events, delta, return_dense_features=True)
        assert output.endpoint_dense_features is not None
        relation = local_cosine_relation_maps(output.endpoint_dense_features[:, -2:])
    batch.dinov3_relation_targets = relation.values.detach().clone()
    batch.dinov3_relation_valid = relation.valid.detach().clone()


def _recipe() -> ProducerRecipe:
    return cast(ProducerRecipe, SimpleNamespace(modality="rgb", model_config={"in_channels": 3}))


def _gradients(model: CausalScaleTTC) -> list[torch.Tensor]:
    return [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]


@pytest.mark.parametrize("steps", [2, 3])
def test_host_counts_preserve_single_part_loss_and_gradients(steps: int) -> None:
    torch.manual_seed(7)
    model = _model().train()
    batch = _batch(steps, f"t{steps}", torch.tensor([2.0, -0.1, -0.100001, 0.0]))
    batch.boxes_xyxy[-1, :, 2:] = batch.boxes_xyxy[-1, :, :2]
    _attach_relations(model, batch)
    effective = EffectiveBatch([batch])
    config = CausalScaleTTCLossConfig()
    baseline, baseline_components = original._producer_loss(model, effective, config)
    baseline.backward()
    baseline_gradients = _gradients(model)
    model.zero_grad(set_to_none=True)
    prepared = prepare_effective_batch(effective, _recipe())
    optimized, optimized_components = original._producer_loss(model, prepared, config)
    optimized.backward()
    optimized_gradients = _gradients(model)
    torch.testing.assert_close(optimized, baseline, rtol=0.0, atol=0.0)
    for name in baseline_components:
        torch.testing.assert_close(
            optimized_components[name], baseline_components[name], rtol=0.0, atol=0.0
        )
    for left, right in zip(optimized_gradients, baseline_gradients, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert prepared.prepared is not None
    assert prepared.prepared[0].counts.physical_ratio == 2
    assert prepared.prepared[0].counts.supervised_ttc == 3
    assert prepared.prepared[0].counts.foreground_geometry == (steps - 1) * 3


def test_mixed_t2_t3_prepared_loss_matches_legacy_reduction() -> None:
    torch.manual_seed(11)
    model = _model().train()
    t2 = _batch(2, "mixed2", torch.tensor([1.0, float("nan")]))
    t3 = _batch(3, "mixed3", torch.tensor([-2.0, 3.0]))
    _attach_relations(model, t2)
    _attach_relations(model, t3)
    effective = EffectiveBatch([t2, t3])
    config = CausalScaleTTCLossConfig()
    baseline, baseline_components = effective_loss(model, effective, config)
    baseline.backward()
    baseline_gradients = _gradients(model)
    model.zero_grad(set_to_none=True)
    prepared = prepare_effective_batch(effective, _recipe())
    moved = prepared.to(torch.device("cpu"))
    optimized, optimized_components = effective_loss(model, moved, config)
    optimized.backward()
    optimized_gradients = _gradients(model)
    torch.testing.assert_close(optimized, baseline, rtol=0.0, atol=0.0)
    for name in baseline_components:
        torch.testing.assert_close(
            optimized_components[name], baseline_components[name], rtol=0.0, atol=0.0
        )
    for left, right in zip(optimized_gradients, baseline_gradients, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    scalars = collect_loss_scalars(optimized, optimized_components)
    assert set(scalars) == {"total", *optimized_components}


def test_prepared_metadata_cannot_bypass_alignment_or_empty_counts() -> None:
    model = _model().eval()
    batch = _batch(2, "guard", torch.tensor([1.0, 0.0]))
    _attach_relations(model, batch)
    prepared = prepare_effective_batch(EffectiveBatch([batch]), _recipe())
    assert prepared.prepared is not None
    metadata = prepared.prepared[0]
    wrong_counts = replace(
        metadata.counts,
        physical_ratio=metadata.counts.physical_ratio + 1,
    )
    wrong = EffectiveBatch(prepared.parts, prepared=[replace(metadata, counts=wrong_counts)])
    with pytest.raises(ValueError, match="prepared physical_ratio count differs"):
        original._producer_loss(model, wrong, CausalScaleTTCLossConfig())
    batch.sample_tokens[0] = "changed-token"
    with pytest.raises(ValueError, match="metadata differs"):
        original._producer_loss(model, prepared, CausalScaleTTCLossConfig())
    clean = _batch(2, "empty", torch.tensor([0.0, float("nan")]))
    clean.boxes_xyxy[:, :, 2:] = clean.boxes_xyxy[:, :, :2]
    _attach_relations(model, clean)
    empty = prepare_effective_batch(EffectiveBatch([clean]), _recipe())
    assert empty.prepared is not None
    assert empty.prepared[0].counts.physical_ratio == 0
    assert empty.prepared[0].counts.supervised_ttc == 0
    assert empty.prepared[0].counts.foreground_geometry == 0
    loss, _ = original._producer_loss(model, empty, CausalScaleTTCLossConfig())
    assert torch.isfinite(loss)


@pytest.mark.parametrize("field", ["target_ttc_s", "delta_t_s", "boxes_xyxy"])
def test_device_transfer_rejects_stale_target_support(field: str) -> None:
    model = _model().eval()
    batch = _batch(2, f"stale-{field}", torch.tensor([1.0, -2.0]))
    _attach_relations(model, batch)
    prepared = prepare_effective_batch(EffectiveBatch([batch]), _recipe())
    value = getattr(batch, field)
    value.reshape(-1)[0] += 0.01
    with pytest.raises(ValueError, match="metadata differs"):
        prepared.to(torch.device("cpu"))
