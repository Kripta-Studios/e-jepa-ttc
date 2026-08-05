"""Losses and modality curriculum for Object Event TTC v4."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.models.object_event_v4 import (
    ObjectEventV4Output,
    expansion_to_log_ratio,
)


@dataclass(frozen=True)
class ObjectEventV4LossConfig:
    fused_expansion_weight: float = 3.0
    event_expansion_weight: float = 5.0
    motion_expansion_weight: float = 0.25
    fused_visible_ratio_weight: float = 2.0
    event_visible_ratio_weight: float = 2.0
    official_ratio_weight: float = 1.0
    fused_ttc_aux_weight: float = 0.03
    event_ttc_aux_weight: float = 0.02
    local_jepa_weight: float = 0.5
    local_variance_weight: float = 0.03
    sign_margin_weight: float = 0.5
    event_gate_floor_weight: float = 0.25
    event_gate_target: float = 0.55
    expansion_beta: float = 0.01
    ratio_beta: float = 0.01
    ttc_beta: float = 0.05
    sign_temperature: float = 0.02
    variance_floor: float = 0.20
    negative_sample_weight: float = 2.0
    crucial_sample_weight: float = 1.5

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if min(float(value) for value in values) < 0.0:
            raise ValueError("V4 loss weights and controls must be non-negative")
        if self.expansion_beta <= 0.0 or self.ratio_beta <= 0.0 or self.ttc_beta <= 0.0:
            raise ValueError("Smooth-L1 beta values must be positive")
        if not 0.0 <= self.event_gate_target <= 1.0:
            raise ValueError("event_gate_target must lie in [0,1]")


@dataclass(frozen=True)
class ObjectEventV4ModalityConfig:
    event_only_warmup_epochs: int = 3
    motion_drop_probability: float = 0.50
    event_drop_probability: float = 0.10
    motion_noise_std: float = 0.03

    def __post_init__(self) -> None:
        if self.event_only_warmup_epochs < 0:
            raise ValueError("event_only_warmup_epochs must be non-negative")
        probabilities = (self.motion_drop_probability, self.event_drop_probability)
        if min(probabilities) < 0.0 or sum(probabilities) > 1.0:
            raise ValueError("Modality dropout probabilities are invalid")
        if self.motion_noise_std < 0.0:
            raise ValueError("motion_noise_std must be non-negative")


@dataclass
class ObjectEventV4Targets:
    signed_expansion: torch.Tensor
    official_log_ratio: torch.Tensor
    visible_log_ratio: torch.Tensor
    sample_weight: torch.Tensor


@dataclass
class ModalityDropoutResult:
    events: torch.Tensor
    observable_motion: torch.Tensor
    motion_dropped: torch.Tensor
    events_dropped: torch.Tensor


@dataclass
class ObjectEventV4LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    targets: ObjectEventV4Targets


def targets_from_batch(
    batch: ObjectEventV4Batch,
    config: ObjectEventV4LossConfig,
) -> ObjectEventV4Targets:
    signed_expansion = batch.delta_t_s / batch.target_ttc_s
    official_log_ratio = torch.log1p(-signed_expansion.clamp(max=1.0 - 1.0e-6))
    visible_log_ratio = torch.log(
        batch.visible_heights_px[:, 0].clamp_min(1.0e-6)
        / batch.visible_heights_px[:, 1].clamp_min(1.0e-6)
    )
    sample_weight = torch.ones_like(signed_expansion)
    sample_weight = torch.where(
        signed_expansion < 0.0,
        sample_weight * config.negative_sample_weight,
        sample_weight,
    )
    crucial = (batch.target_ttc_s > 0.0) & (batch.target_ttc_s <= 1.0)
    sample_weight = torch.where(
        crucial,
        sample_weight * config.crucial_sample_weight,
        sample_weight,
    )
    sample_weight = sample_weight / sample_weight.mean().clamp_min(1.0e-6)
    return ObjectEventV4Targets(
        signed_expansion=signed_expansion,
        official_log_ratio=official_log_ratio,
        visible_log_ratio=visible_log_ratio,
        sample_weight=sample_weight,
    )


def apply_modality_dropout(
    events: torch.Tensor,
    observable_motion: torch.Tensor,
    *,
    epoch: int,
    config: ObjectEventV4ModalityConfig,
    generator: torch.Generator | None = None,
) -> ModalityDropoutResult:
    """Drop one modality per sample, never both.

    During the initial warmup every sample is event-only.  Thereafter one random
    draw selects motion-drop, event-drop, or both modalities.  Motion noise is
    applied only where motion remains available.
    """

    if events.shape[0] != observable_motion.shape[0]:
        raise ValueError("Events and motion must share batch size")
    batch = events.shape[0]
    device = events.device
    if epoch <= config.event_only_warmup_epochs:
        motion_dropped = torch.ones(batch, dtype=torch.bool, device=device)
        events_dropped = torch.zeros(batch, dtype=torch.bool, device=device)
    else:
        draw = torch.rand(batch, device=device, generator=generator)
        motion_dropped = draw < config.motion_drop_probability
        events_dropped = (
            draw >= config.motion_drop_probability
        ) & (
            draw
            < config.motion_drop_probability + config.event_drop_probability
        )
    if bool((motion_dropped & events_dropped).any()):
        raise RuntimeError("Modality dropout attempted to remove both modalities")

    kept_events = events.masked_fill(events_dropped[:, None, None, None, None], 0.0)
    kept_motion = observable_motion.masked_fill(motion_dropped[:, None], 0.0)
    if config.motion_noise_std > 0.0:
        scale = observable_motion.detach().std(dim=0, keepdim=True).clamp_min(1.0e-3)
        noise = torch.randn(
            observable_motion.shape,
            device=observable_motion.device,
            dtype=observable_motion.dtype,
            generator=generator,
        ) * scale * config.motion_noise_std
        kept_motion = kept_motion + noise.masked_fill(
            motion_dropped[:, None], 0.0
        )
    return ModalityDropoutResult(
        events=kept_events,
        observable_motion=kept_motion,
        motion_dropped=motion_dropped,
        events_dropped=events_dropped,
    )


def _weighted_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    value = functional.smooth_l1_loss(
        prediction,
        target,
        beta=beta,
        reduction="none",
    )
    return (value * weight).mean()


def _sign_margin(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    sign = torch.where(target < 0.0, -torch.ones_like(target), torch.ones_like(target))
    return functional.softplus(-sign * prediction / temperature).mean()


def _local_jepa_loss(output: ObjectEventV4Output) -> torch.Tensor:
    mask = output.future_token_mask
    predicted = functional.normalize(output.predicted_future_tokens, dim=-1)
    target = functional.normalize(output.target_future_tokens.detach(), dim=-1)
    cosine = 1.0 - (predicted * target).sum(dim=-1)
    return (cosine * mask.to(cosine.dtype)).sum() / mask.sum().clamp_min(1)


def _variance_loss(output: ObjectEventV4Output, floor: float) -> torch.Tensor:
    mask = output.future_token_mask
    values = output.predicted_future_tokens[mask]
    if values.shape[0] < 2:
        return values.new_zeros(())
    std = values.float().std(dim=0, unbiased=False)
    return functional.relu(floor - std).mean().to(values.dtype)


def object_event_v4_loss(
    output: ObjectEventV4Output,
    batch: ObjectEventV4Batch,
    config: ObjectEventV4LossConfig,
) -> ObjectEventV4LossOutput:
    targets = targets_from_batch(batch, config)
    weight = targets.sample_weight
    fused_log_ratio = expansion_to_log_ratio(output.signed_expansion)
    event_log_ratio = expansion_to_log_ratio(output.event_expansion)

    components = {
        "fused_expansion": _weighted_smooth_l1(
            output.signed_expansion,
            targets.signed_expansion,
            weight,
            beta=config.expansion_beta,
        ),
        "event_expansion": _weighted_smooth_l1(
            output.event_expansion,
            targets.signed_expansion,
            weight,
            beta=config.expansion_beta,
        ),
        "motion_expansion": _weighted_smooth_l1(
            output.motion_expansion,
            targets.signed_expansion,
            weight,
            beta=config.expansion_beta,
        ),
        "fused_visible_ratio": _weighted_smooth_l1(
            fused_log_ratio,
            targets.visible_log_ratio,
            weight,
            beta=config.ratio_beta,
        ),
        "event_visible_ratio": _weighted_smooth_l1(
            event_log_ratio,
            targets.visible_log_ratio,
            weight,
            beta=config.ratio_beta,
        ),
        "official_ratio": _weighted_smooth_l1(
            fused_log_ratio,
            targets.official_log_ratio,
            weight,
            beta=config.ratio_beta,
        ),
        "fused_ttc_aux": _weighted_smooth_l1(
            output.ttc_mean_seconds,
            batch.target_ttc_s,
            weight,
            beta=config.ttc_beta,
        ),
        "event_ttc_aux": _weighted_smooth_l1(
            output.event_ttc_seconds,
            batch.target_ttc_s,
            weight,
            beta=config.ttc_beta,
        ),
        "local_jepa": _local_jepa_loss(output),
        "local_variance": _variance_loss(output, config.variance_floor),
        "sign_margin": 0.5
        * (
            _sign_margin(
                output.signed_expansion,
                targets.signed_expansion,
                temperature=config.sign_temperature,
            )
            + _sign_margin(
                output.event_expansion,
                targets.signed_expansion,
                temperature=config.sign_temperature,
            )
        ),
        "event_gate_floor": functional.relu(
            config.event_gate_target - output.event_gate
        ).mean(),
    }
    total = (
        config.fused_expansion_weight * components["fused_expansion"]
        + config.event_expansion_weight * components["event_expansion"]
        + config.motion_expansion_weight * components["motion_expansion"]
        + config.fused_visible_ratio_weight * components["fused_visible_ratio"]
        + config.event_visible_ratio_weight * components["event_visible_ratio"]
        + config.official_ratio_weight * components["official_ratio"]
        + config.fused_ttc_aux_weight * components["fused_ttc_aux"]
        + config.event_ttc_aux_weight * components["event_ttc_aux"]
        + config.local_jepa_weight * components["local_jepa"]
        + config.local_variance_weight * components["local_variance"]
        + config.sign_margin_weight * components["sign_margin"]
        + config.event_gate_floor_weight * components["event_gate_floor"]
    )
    return ObjectEventV4LossOutput(total=total, components=components, targets=targets)


__all__ = [
    "ModalityDropoutResult",
    "ObjectEventV4LossConfig",
    "ObjectEventV4LossOutput",
    "ObjectEventV4ModalityConfig",
    "ObjectEventV4Targets",
    "apply_modality_dropout",
    "object_event_v4_loss",
    "targets_from_batch",
]
