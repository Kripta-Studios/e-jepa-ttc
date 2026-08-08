"""Losses and diagnostics for Object Event TTC v4.27 scale-correlation LHR."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_27 import ObjectEventV427Output
from e_jepa_ttc.training.object_event_v4_1 import pearson_torch, target_expansion


@dataclass(frozen=True)
class ObjectEventV427LossConfig:
    lhr_weight: float = 4.0
    expansion_weight: float = 1.0
    correlation_weight: float = 1.0
    sign_weight: float = 1.0
    entropy_weight: float = 0.02
    smooth_l1_beta: float = 0.004
    sign_temperature: float = 0.015
    max_abs_expansion: float = 0.25
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if min(
            self.lhr_weight,
            self.expansion_weight,
            self.correlation_weight,
            self.sign_weight,
            self.entropy_weight,
        ) < 0.0:
            raise ValueError("v4.27 loss weights must be non-negative")
        if min(self.smooth_l1_beta, self.sign_temperature, self.epsilon) <= 0.0:
            raise ValueError("v4.27 loss scales must be positive")


def target_log_height_ratio(visible_heights_px: torch.Tensor) -> torch.Tensor:
    if visible_heights_px.ndim != 2 or visible_heights_px.shape[1] != 2:
        raise ValueError("visible_heights_px must be [B,2]")
    return torch.log(visible_heights_px[:, 0].clamp_min(1.0e-4)) - torch.log(
        visible_heights_px[:, 1].clamp_min(1.0e-4)
    )


def balanced_sign_weights(target: torch.Tensor, epsilon: float = 1.0e-6) -> torch.Tensor:
    negative = target < 0.0
    positive = ~negative
    n = float(target.numel())
    neg_count = negative.sum().to(target.dtype).clamp_min(1.0)
    pos_count = positive.sum().to(target.dtype).clamp_min(1.0)
    weights = torch.where(negative, n / (2.0 * neg_count), n / (2.0 * pos_count))
    return weights / weights.mean().clamp_min(epsilon)


def object_event_v4_27_loss(
    output: ObjectEventV427Output,
    delta_t_s: torch.Tensor,
    target_ttc_s: torch.Tensor,
    visible_heights_px: torch.Tensor,
    *,
    config: ObjectEventV427LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_exp = target_expansion(delta_t_s, target_ttc_s, config.max_abs_expansion)
    target_lhr = target_log_height_ratio(visible_heights_px)
    sample_weight = balanced_sign_weights(target_exp, config.epsilon)
    lhr_each = functional.smooth_l1_loss(
        output.predicted_log_eta, target_lhr, beta=config.smooth_l1_beta, reduction="none"
    )
    expansion_each = functional.smooth_l1_loss(
        output.expansion, target_exp, beta=config.smooth_l1_beta, reduction="none"
    )
    signed = torch.where(target_exp >= 0.0, torch.ones_like(target_exp), -torch.ones_like(target_exp))
    sign_each = functional.softplus(-signed * output.expansion / config.sign_temperature)
    components = {
        "lhr": (sample_weight * lhr_each).mean(),
        "expansion": (sample_weight * expansion_each).mean(),
        "correlation": 1.0 - pearson_torch(output.expansion, target_exp),
        "sign": (sample_weight * sign_each).mean(),
        "entropy": output.scale_entropy.mean(),
    }
    total = (
        config.lhr_weight * components["lhr"]
        + config.expansion_weight * components["expansion"]
        + config.correlation_weight * components["correlation"]
        + config.sign_weight * components["sign"]
        + config.entropy_weight * components["entropy"]
    )
    return total, {**components, "target_log_eta": target_lhr, "target_expansion": target_exp}


__all__ = [
    "ObjectEventV427LossConfig",
    "balanced_sign_weights",
    "object_event_v4_27_loss",
    "target_log_height_ratio",
]
