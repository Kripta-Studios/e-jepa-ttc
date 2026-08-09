"""Losses for the geometry-bound v5 causal-scale TTC model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.causal_scale_ttc import (
    CausalScaleTTCOutput,
    soft_vertical_extent_from_logits,
    target_log_ratio_from_ttc,
)


@dataclass(frozen=True)
class CausalScaleTTCLossConfig:
    """Weights for supervised geometry without a direct-TTC primary bypass."""

    log_ratio_nll_weight: float = 1.0
    log_ratio_huber_weight: float = 1.0
    foreground_bce_weight: float = 1.0
    foreground_dice_weight: float = 0.5
    foreground_extent_weight: float = 1.0
    risk_weight: float = 0.25
    auxiliary_inverse_ttc_weight: float = 0.1
    residual_regularization_weight: float = 0.05
    temporal_consistency_weight: float = 0.1
    smooth_l1_beta: float = 0.02

    def __post_init__(self) -> None:
        weights = (
            self.log_ratio_nll_weight,
            self.log_ratio_huber_weight,
            self.foreground_bce_weight,
            self.foreground_dice_weight,
            self.foreground_extent_weight,
            self.risk_weight,
            self.auxiliary_inverse_ttc_weight,
            self.residual_regularization_weight,
            self.temporal_consistency_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("causal-scale loss weights must be non-negative")
        if self.smooth_l1_beta <= 0.0:
            raise ValueError("smooth_l1_beta must be positive")


@dataclass
class CausalScaleTTCLoss:
    """Total loss, auditable components, and target support counts."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]
    counts: dict[str, int]


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _foreground_losses(
    output: CausalScaleTTCOutput,
    target_masks: torch.Tensor | None,
    mask_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if target_masks is None and mask_valid is None:
        zero = _zero(output.foreground_logits)
        return zero, zero, 0
    if target_masks is None or mask_valid is None:
        raise ValueError("target_masks and mask_valid must be provided together")
    if target_masks.shape != output.foreground_logits.shape:
        raise ValueError("target_masks must match foreground_logits")
    if mask_valid.shape != output.foreground_logits.shape[:2]:
        raise ValueError("mask_valid must have shape [B,T]")
    valid = mask_valid.bool()
    count = int(valid.sum().item())
    if count == 0:
        zero = _zero(output.foreground_logits)
        return zero, zero, 0
    logits = output.foreground_logits[valid]
    targets = target_masks[valid].to(dtype=logits.dtype).clamp(0.0, 1.0)
    bce = functional.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(-3, -2, -1))
    denominator = probabilities.sum(dim=(-3, -2, -1)) + targets.sum(dim=(-3, -2, -1))
    dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce, dice, count


def _foreground_extent_loss(
    output: CausalScaleTTCOutput,
    target_masks: torch.Tensor | None,
    mask_valid: torch.Tensor | None,
    *,
    beta: float,
) -> tuple[torch.Tensor, int]:
    if target_masks is None or mask_valid is None:
        return _zero(output.visible_height_normalized), 0
    nonempty = target_masks.flatten(-3).any(dim=-1)
    valid = mask_valid.bool() & nonempty
    count = int(valid.sum().item())
    if count == 0:
        return _zero(output.visible_height_normalized), 0
    target_logits = torch.where(
        target_masks > 0.5,
        target_masks.new_tensor(12.0),
        target_masks.new_tensor(-12.0),
    )
    target_extent = soft_vertical_extent_from_logits(target_logits).height_normalized
    extent = functional.smooth_l1_loss(
        output.visible_height_normalized[valid].clamp_min(1.0e-6).log(),
        target_extent[valid].clamp_min(1.0e-6).log(),
        beta=beta,
    )
    return extent, count


def causal_scale_ttc_loss(
    output: CausalScaleTTCOutput,
    *,
    target_ttc_seconds: torch.Tensor,
    delta_t_s: torch.Tensor,
    risk_thresholds_s: tuple[float, ...],
    target_valid: torch.Tensor | None = None,
    target_masks: torch.Tensor | None = None,
    mask_valid: torch.Tensor | None = None,
    config: CausalScaleTTCLossConfig | None = None,
) -> CausalScaleTTCLoss:
    """Train foreground scale, uncertainty, risk, and an auxiliary direct readout.

    The main regression term is a Gaussian NLL on the physically derived log scale
    ratio.  The auxiliary inverse-TTC head cannot affect the primary model output.
    """

    cfg = config or CausalScaleTTCLossConfig()
    batch = output.ttc_mean_seconds.shape[0]
    if target_ttc_seconds.shape != (batch,):
        raise ValueError("target_ttc_seconds must have shape [B]")
    if delta_t_s.ndim != 2 or delta_t_s.shape[0] != batch:
        raise ValueError("delta_t_s must have shape [B,T-1]")
    if output.log_height_ratio.shape != delta_t_s.shape:
        raise ValueError("output log ratios and delta_t_s must share shape")
    if len(risk_thresholds_s) != output.collision_logits.shape[-1]:
        raise ValueError("risk thresholds must match collision logits")
    target_ratio, physical_valid = target_log_ratio_from_ttc(
        target_ttc_seconds,
        delta_t_s[:, -1],
    )
    if target_valid is not None:
        if target_valid.shape != (batch,):
            raise ValueError("target_valid must have shape [B]")
        physical_valid = physical_valid & target_valid.bool()
    valid_count = int(physical_valid.sum().item())
    if valid_count:
        residual = output.log_height_ratio[:, -1][physical_valid] - target_ratio[physical_valid]
        log_variance = output.log_ratio_log_variance[:, -1][physical_valid]
        ratio_nll = (0.5 * torch.exp(-log_variance) * residual.square() + 0.5 * log_variance).mean()
        ratio_huber = functional.smooth_l1_loss(
            output.log_height_ratio[:, -1][physical_valid],
            target_ratio[physical_valid],
            beta=cfg.smooth_l1_beta,
        )
    else:
        ratio_nll = _zero(output.log_height_ratio)
        ratio_huber = _zero(output.log_height_ratio)

    finite_ttc = torch.isfinite(target_ttc_seconds) & (target_ttc_seconds.abs() > 1.0e-8)
    if target_valid is not None:
        finite_ttc = finite_ttc & target_valid.bool()
    supervised_count = int(finite_ttc.sum().item())
    if supervised_count:
        inverse_target = target_ttc_seconds[finite_ttc].reciprocal()
        auxiliary = functional.smooth_l1_loss(
            output.auxiliary_inverse_ttc[:, -1][finite_ttc],
            inverse_target,
            beta=cfg.smooth_l1_beta,
        )
        thresholds = target_ttc_seconds.new_tensor(risk_thresholds_s)
        labels = (
            (target_ttc_seconds[:, None] > 0.0)
            & (target_ttc_seconds[:, None] <= thresholds[None, :])
        ).to(output.collision_logits.dtype)
        risk = functional.binary_cross_entropy_with_logits(
            output.collision_logits[finite_ttc],
            labels[finite_ttc],
        )
    else:
        auxiliary = _zero(output.auxiliary_inverse_ttc)
        risk = _zero(output.collision_logits)

    foreground_bce, foreground_dice, foreground_count = _foreground_losses(
        output,
        target_masks,
        mask_valid,
    )
    foreground_extent, foreground_extent_count = _foreground_extent_loss(
        output,
        target_masks,
        mask_valid,
        beta=cfg.smooth_l1_beta,
    )
    residual_regularization = output.residual_log_height_ratio.square().mean()
    pair_known = output.diagnostics["pair_known"].bool()
    if output.pair_ttc_seconds.shape[1] >= 2:
        consistency_valid = pair_known[:, :-1] & pair_known[:, 1:]
        consistency_count = int(consistency_valid.sum().item())
        if consistency_count:
            expected_previous = output.pair_ttc_seconds[:, 1:] + delta_t_s[:, 1:]
            temporal_consistency = functional.smooth_l1_loss(
                output.pair_ttc_seconds[:, :-1][consistency_valid],
                expected_previous[consistency_valid],
                beta=cfg.smooth_l1_beta,
            )
        else:
            temporal_consistency = _zero(output.pair_ttc_seconds)
    else:
        consistency_count = 0
        temporal_consistency = _zero(output.pair_ttc_seconds)

    components = {
        "log_ratio_nll": ratio_nll,
        "log_ratio_huber": ratio_huber,
        "foreground_bce": foreground_bce,
        "foreground_dice": foreground_dice,
        "foreground_extent": foreground_extent,
        "risk_bce": risk,
        "auxiliary_inverse_ttc": auxiliary,
        "residual_regularization": residual_regularization,
        "temporal_consistency": temporal_consistency,
    }
    total = (
        cfg.log_ratio_nll_weight * ratio_nll
        + cfg.log_ratio_huber_weight * ratio_huber
        + cfg.foreground_bce_weight * foreground_bce
        + cfg.foreground_dice_weight * foreground_dice
        + cfg.foreground_extent_weight * foreground_extent
        + cfg.risk_weight * risk
        + cfg.auxiliary_inverse_ttc_weight * auxiliary
        + cfg.residual_regularization_weight * residual_regularization
        + cfg.temporal_consistency_weight * temporal_consistency
    )
    return CausalScaleTTCLoss(
        total=total,
        components=components,
        counts={
            "physical_ratio": valid_count,
            "supervised_ttc": supervised_count,
            "foreground": foreground_count,
            "foreground_extent": foreground_extent_count,
            "temporal_consistency": consistency_count,
        },
    )


__all__ = [
    "CausalScaleTTCLoss",
    "CausalScaleTTCLossConfig",
    "causal_scale_ttc_loss",
]
