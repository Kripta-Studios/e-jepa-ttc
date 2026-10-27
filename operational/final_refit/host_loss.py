"""Host-only validation and target support metadata for RGB producer losses."""

from __future__ import annotations

import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.losses.causal_scale_ttc import PreparedTargetCounts
from e_jepa_ttc.models.causal_scale_ttc import target_log_ratio_from_ttc
from operational.rgb_port import train_producers as original
from operational.rgb_port.recipe import ProducerRecipe
from operational.rgb_port_revision.loss import (
    EffectiveBatch,
    PreparedEffectivePart,
    collect_loss_scalars,
    target_support_sha256,
)


def _identity(values: list[str], *, name: str, batch_size: int) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if len(result) != batch_size or any(not value for value in result):
        raise ValueError(f"{name} must align with every effective-batch row")
    return result


def _prepare_part(part: ObjectEventV4Batch) -> PreparedEffectivePart:
    tensors = {
        "events": part.events,
        "delta_t_s": part.delta_t_s,
        "target_ttc_s": part.target_ttc_s,
        "boxes_xyxy": part.boxes_xyxy,
        "dinov3_relation_targets": part.dinov3_relation_targets,
        "dinov3_relation_valid": part.dinov3_relation_valid,
    }
    if any(value is None or value.device.type != "cpu" for value in tensors.values()):
        raise ValueError("host loss metadata requires all loss inputs on CPU")
    size, steps = part.events.shape[:2]
    if steps not in (2, 3):
        raise ValueError("RGB effective parts must preserve their real T2/T3 length")
    delta = original._pair_intervals(part.delta_t_s, batch_size=size, steps=steps)
    target_valid = torch.isfinite(part.target_ttc_s) & (part.target_ttc_s != 0.0)
    _, physical_valid = target_log_ratio_from_ttc(part.target_ttc_s, delta[:, -1])
    physical_valid &= target_valid
    supervised_valid = (
        torch.isfinite(part.target_ttc_s)
        & (part.target_ttc_s.abs() > 1.0e-8)
        & target_valid
    )
    endpoint_valid = torch.ones(part.boxes_xyxy.shape[:2], dtype=torch.bool)
    endpoint_valid[:, 0] = False
    geometry = original.box_geometry_targets(
        part.boxes_xyxy,
        height=int(part.events.shape[-2]),
        width=int(part.events.shape[-1]),
        endpoint_valid=endpoint_valid & target_valid[:, None],
    )
    geometry_valid = geometry.valid.bool()
    pair_valid = geometry_valid[:, :-1] & geometry_valid[:, 1:]
    counts = PreparedTargetCounts(
        physical_ratio=int(physical_valid.sum().item()),
        supervised_ttc=int(supervised_valid.sum().item()),
        foreground_extent=int(geometry_valid.sum().item()),
        foreground_geometry=int(geometry_valid.sum().item()),
        foreground_pair_ratio=int(pair_valid.sum().item()),
    )
    return PreparedEffectivePart(
        sample_tokens=_identity(part.sample_tokens, name="sample_tokens", batch_size=size),
        sequence_ids=_identity(part.sequence_ids, name="sequence_ids", batch_size=size),
        track_ids=_identity(part.track_ids, name="track_ids", batch_size=size),
        event_shape=tuple(int(value) for value in part.events.shape),
        delta_shape=tuple(int(value) for value in part.delta_t_s.shape),
        target_shape=tuple(int(value) for value in part.target_ttc_s.shape),
        target_support_sha256=target_support_sha256(
            part.target_ttc_s, part.delta_t_s, part.boxes_xyxy
        ),
        counts=counts,
    )


def prepare_effective_batch(
    batch: EffectiveBatch, recipe: ProducerRecipe
) -> EffectiveBatch:
    """Validate an effective CPU batch and attach target-only support counts."""
    if not isinstance(batch, EffectiveBatch) or not batch.parts:
        raise ValueError("host loss preparation requires a nonempty EffectiveBatch")
    original._validate_batch(batch, recipe)
    prepared = [_prepare_part(part) for part in batch.parts]
    if batch.prepared is not None:
        if batch.prepared != prepared:
            raise ValueError("existing host loss metadata differs from recomputed targets")
        return batch
    return EffectiveBatch(parts=batch.parts, prepared=prepared)


__all__ = ["collect_loss_scalars", "prepare_effective_batch"]
