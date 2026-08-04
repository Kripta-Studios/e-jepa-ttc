"""Stable objectives for object-centric inverse-TTC prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.data.garlttc_object_lhr import ObjectLHRBatch
from e_jepa_ttc.models.object_expansion import ObjectExpansionOutput


@dataclass(frozen=True)
class ObjectExpansionLossConfig:
    """Auditable two-stage objective without the v1 ``mid_scale=10000`` jump."""

    geometry_warmup_epochs: int = 5
    magnitude_weight: float = 1.0
    direction_weight: float = 1.0
    signed_inverse_weight: float = 2.0
    log_ratio_weight: float = 2.0
    ttc_aux_weight: float = 0.05
    direction_negative_weight: float = 3.0
    magnitude_beta: float = 0.05
    signed_inverse_beta: float = 0.02
    log_ratio_beta: float = 0.01
    ttc_aux_beta: float = 0.05
    label_smoothing: float = 0.02

    def __post_init__(self) -> None:
        if self.geometry_warmup_epochs < 0:
            raise ValueError("geometry_warmup_epochs must be non-negative")
        weights = (
            self.magnitude_weight,
            self.direction_weight,
            self.signed_inverse_weight,
            self.log_ratio_weight,
            self.ttc_aux_weight,
            self.direction_negative_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("Loss weights must be non-negative")
        betas = (
            self.magnitude_beta,
            self.signed_inverse_beta,
            self.log_ratio_beta,
            self.ttc_aux_beta,
        )
        if min(betas) <= 0.0:
            raise ValueError("Smooth-L1 beta values must be positive")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must lie in [0,1)")


@dataclass
class ObjectExpansionTargets:
    """Stable supervised coordinates derived from official signed TTC."""

    signed_inverse_ttc: torch.Tensor
    log_abs_inverse_ttc: torch.Tensor
    negative_class: torch.Tensor
    log_height_ratio: torch.Tensor
    visible_log_ratio: torch.Tensor


@dataclass
class ObjectExpansionLossResult:
    """Differentiable total plus detached diagnostic components."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    phase: str


def curriculum_phase(epoch: int, config: ObjectExpansionLossConfig) -> str:
    """Return the deterministic objective phase for a one-indexed epoch."""

    if epoch <= 0:
        raise ValueError("epoch must be one-indexed and positive")
    if epoch <= config.geometry_warmup_epochs:
        return "inverse_ttc_warmup"
    return "geometry_consistency"


def signed_log1p(value: torch.Tensor) -> torch.Tensor:
    """Signed logarithm used only by the low-weight TTC auxiliary."""

    return value.sign() * torch.log1p(value.abs())


def targets_from_batch(batch: ObjectLHRBatch) -> ObjectExpansionTargets:
    """Convert official TTC into stable inverse-time and LHR coordinates."""

    if not torch.isfinite(batch.target_ttc_s).all():
        raise ValueError("TTC targets must be finite")
    if bool((batch.target_ttc_s == 0).any()):
        raise ValueError("TTC targets must be non-zero")
    signed_inverse = batch.target_ttc_s.reciprocal()
    log_abs_inverse = torch.log(signed_inverse.abs())
    negative_class = (batch.target_ttc_s < 0).to(dtype=torch.long)
    normalized_expansion = batch.delta_t_s * signed_inverse
    if bool((normalized_expansion >= 1).any()):
        raise ValueError("Official TTC lies outside the LHR domain")
    log_height_ratio = torch.log1p(-normalized_expansion)
    visible_log_ratio = (
        torch.log(batch.visible_heights_px[:, 0].clamp_min(1.0e-6))
        - torch.log(batch.visible_heights_px[:, 1].clamp_min(1.0e-6))
    )
    return ObjectExpansionTargets(
        signed_inverse_ttc=signed_inverse,
        log_abs_inverse_ttc=log_abs_inverse,
        negative_class=negative_class,
        log_height_ratio=log_height_ratio,
        visible_log_ratio=visible_log_ratio,
    )


def object_expansion_loss(
    output: ObjectExpansionOutput,
    batch: ObjectLHRBatch,
    *,
    epoch: int,
    config: ObjectExpansionLossConfig,
) -> ObjectExpansionLossResult:
    """Optimize sign and inverse TTC before adding LHR consistency."""

    phase = curriculum_phase(epoch, config)
    targets = targets_from_batch(batch)
    components: dict[str, torch.Tensor] = {}

    components["log_abs_inverse_ttc"] = functional.smooth_l1_loss(
        output.log_abs_inverse_ttc,
        targets.log_abs_inverse_ttc,
        beta=config.magnitude_beta,
    )
    class_weight = torch.tensor(
        [1.0, config.direction_negative_weight],
        device=output.direction_logits.device,
        dtype=output.direction_logits.dtype,
    )
    components["direction"] = functional.cross_entropy(
        output.direction_logits,
        targets.negative_class,
        weight=class_weight,
        label_smoothing=config.label_smoothing,
    )
    components["signed_inverse_ttc"] = functional.smooth_l1_loss(
        output.signed_inverse_ttc_soft,
        targets.signed_inverse_ttc,
        beta=config.signed_inverse_beta,
    )
    total = (
        config.magnitude_weight * components["log_abs_inverse_ttc"]
        + config.direction_weight * components["direction"]
        + config.signed_inverse_weight * components["signed_inverse_ttc"]
    )

    if phase == "geometry_consistency":
        components["log_height_ratio"] = functional.smooth_l1_loss(
            output.log_height_ratio_soft,
            targets.log_height_ratio,
            beta=config.log_ratio_beta,
        )
        components["signed_ttc_aux"] = functional.smooth_l1_loss(
            signed_log1p(output.ttc_soft_seconds),
            signed_log1p(batch.target_ttc_s),
            beta=config.ttc_aux_beta,
        )
        total = (
            total
            + config.log_ratio_weight * components["log_height_ratio"]
            + config.ttc_aux_weight * components["signed_ttc_aux"]
        )

    components["target_inverse_mae"] = torch.abs(
        output.signed_inverse_ttc_soft.detach()
        - targets.signed_inverse_ttc.detach()
    ).mean()
    components["target_log_ratio_mae"] = torch.abs(
        output.log_height_ratio_soft.detach()
        - targets.log_height_ratio.detach()
    ).mean()
    components["official_geometry_disagreement"] = torch.abs(
        targets.visible_log_ratio.detach() - targets.log_height_ratio.detach()
    ).mean()
    return ObjectExpansionLossResult(
        total=total,
        components=components,
        phase=phase,
    )


__all__ = [
    "ObjectExpansionLossConfig",
    "ObjectExpansionLossResult",
    "ObjectExpansionTargets",
    "curriculum_phase",
    "object_expansion_loss",
    "signed_log1p",
    "targets_from_batch",
]
