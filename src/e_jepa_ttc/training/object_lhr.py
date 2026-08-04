"""Curriculum and losses for object-centric visible-height-ratio TTC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.data.garlttc_object_lhr import ObjectLHRBatch
from e_jepa_ttc.models.object_lhr import ObjectCentricLHROutput


@dataclass(frozen=True)
class ObjectLHRCurriculumConfig:
    """Loss weights and phase boundaries.

    Phase 1 learns the two observable heights. Phase 2 adds the visible-height
    ratio and direction. Phase 3 adds the TTC-aligned MiD term. This mirrors the
    official Garl-TTC curriculum while keeping each component auditable.
    """

    height_only_epochs: int = 5
    ratio_warmup_epochs: int = 10
    visible_height_weight: float = 1.0
    visible_ratio_weight: float = 10.0
    mid_weight: float = 1.0
    mid_scale: float = 1.0e4
    mask_weight: float = 500.0
    direction_weight: float = 0.1
    signed_ttc_aux_weight: float = 0.0
    smooth_l1_beta: float = 0.05
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    def __post_init__(self) -> None:
        if self.height_only_epochs < 0 or self.ratio_warmup_epochs < self.height_only_epochs:
            raise ValueError("Curriculum epochs must satisfy 0 <= height_only <= ratio_warmup")
        weights = (
            self.visible_height_weight,
            self.visible_ratio_weight,
            self.mid_weight,
            self.mid_scale,
            self.mask_weight,
            self.direction_weight,
            self.signed_ttc_aux_weight,
            self.smooth_l1_beta,
        )
        if min(weights) < 0:
            raise ValueError("Object-LHR loss weights must be non-negative")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")


@dataclass
class ObjectLHRLossResult:
    """Total differentiable loss plus detached component metrics."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    phase: str


def curriculum_phase(epoch: int, config: ObjectLHRCurriculumConfig) -> str:
    """Return the deterministic objective phase for a one-indexed epoch."""

    if epoch <= 0:
        raise ValueError("epoch must be one-indexed and positive")
    if epoch <= config.height_only_epochs:
        return "height_only"
    if epoch <= config.ratio_warmup_epochs:
        return "height_ratio"
    return "full_mid"


def target_log_ratio_from_ttc(delta_t_s: torch.Tensor, target_ttc_s: torch.Tensor) -> torch.Tensor:
    """Compute log(1 - delta_t / TTC) and fail closed outside the LHR domain."""

    if delta_t_s.shape != target_ttc_s.shape:
        raise ValueError("delta_t_s and target_ttc_s must have identical shape")
    ratio = 1.0 - delta_t_s / target_ttc_s
    if not torch.isfinite(ratio).all() or bool((ratio <= 0).any()):
        raise ValueError("Official TTC target produced a non-positive LHR ratio")
    return torch.log(ratio)


def signed_log1p(value: torch.Tensor) -> torch.Tensor:
    """Signed logarithm used only by the optional direct-TTC auxiliary."""

    return value.sign() * torch.log1p(value.abs())


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """Numerically stable binary focal loss."""

    if logits.shape != targets.shape:
        raise ValueError("Focal logits and targets must have identical shapes")
    targets = targets.to(dtype=logits.dtype)
    bce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def object_lhr_loss(
    output: ObjectCentricLHROutput,
    batch: ObjectLHRBatch,
    *,
    epoch: int,
    config: ObjectLHRCurriculumConfig,
) -> ObjectLHRLossResult:
    """Compute the phase-gated geometric training objective."""

    phase = curriculum_phase(epoch, config)
    target_log_heights = torch.log(batch.visible_heights_px.clamp_min(1e-6))
    target_visible_log_ratio = target_log_heights[:, 0] - target_log_heights[:, 1]
    target_ttc_log_ratio = target_log_ratio_from_ttc(batch.delta_t_s, batch.target_ttc_s)

    components: dict[str, torch.Tensor] = {}
    components["visible_height"] = functional.smooth_l1_loss(
        output.log_visible_heights,
        target_log_heights,
        beta=config.smooth_l1_beta,
    )
    total = config.visible_height_weight * components["visible_height"]

    if phase in {"height_ratio", "full_mid"}:
        components["visible_ratio"] = functional.smooth_l1_loss(
            output.log_height_ratio,
            target_visible_log_ratio,
            beta=config.smooth_l1_beta,
        )
        total = total + config.visible_ratio_weight * components["visible_ratio"]
        direction_target = (batch.target_ttc_s < 0).to(dtype=torch.long)
        components["direction"] = functional.cross_entropy(
            output.direction_logits,
            direction_target,
        )
        total = total + config.direction_weight * components["direction"]

    if phase == "full_mid":
        components["mid"] = torch.abs(output.log_height_ratio - target_ttc_log_ratio).mean()
        total = total + config.mid_weight * config.mid_scale * components["mid"]
        if config.signed_ttc_aux_weight > 0:
            components["signed_ttc_aux"] = functional.smooth_l1_loss(
                signed_log1p(output.ttc_mean_seconds),
                signed_log1p(batch.target_ttc_s),
                beta=config.smooth_l1_beta,
            )
            total = total + config.signed_ttc_aux_weight * components["signed_ttc_aux"]

    if output.mask_logits is not None and bool(batch.mask_valid.any()):
        valid = batch.mask_valid[:, :, None, None, None].expand_as(output.mask_logits)
        components["mask_focal"] = binary_focal_loss_with_logits(
            output.mask_logits[valid],
            batch.masks[valid],
            alpha=config.focal_alpha,
            gamma=config.focal_gamma,
        )
        total = total + config.mask_weight * components["mask_focal"]

    components["target_visible_log_ratio_mae"] = torch.abs(
        output.log_height_ratio.detach() - target_visible_log_ratio.detach()
    ).mean()
    components["target_ttc_log_ratio_mae"] = torch.abs(
        output.log_height_ratio.detach() - target_ttc_log_ratio.detach()
    ).mean()
    return ObjectLHRLossResult(total=total, components=components, phase=phase)


def mask_iou(
    logits: torch.Tensor | None,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> float | None:
    """Return endpoint-level foreground IoU for valid masks only."""

    if logits is None or not bool(valid.any()):
        return None
    prediction = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    valid_pixels = valid[:, :, None, None, None].expand_as(prediction)
    prediction = prediction[valid_pixels]
    truth = truth[valid_pixels]
    intersection = torch.logical_and(prediction, truth).sum().float()
    union = torch.logical_or(prediction, truth).sum().float()
    return float((intersection / union.clamp_min(1.0)).detach().cpu())


__all__ = [
    "ObjectLHRCurriculumConfig",
    "ObjectLHRLossResult",
    "binary_focal_loss_with_logits",
    "curriculum_phase",
    "mask_iou",
    "object_lhr_loss",
    "signed_log1p",
    "target_log_ratio_from_ttc",
]
