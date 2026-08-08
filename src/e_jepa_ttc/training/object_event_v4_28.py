"""Posterior-aware losses for Object Event TTC v4.28."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_28 import ObjectEventV428Output
from e_jepa_ttc.training.object_event_v4_1 import pearson_torch, target_expansion
from e_jepa_ttc.training.object_event_v4_27 import balanced_sign_weights, target_log_height_ratio


@dataclass(frozen=True)
class ObjectEventV428LossConfig:
    lhr_weight: float = 4.0
    expansion_weight: float = 1.0
    correlation_weight: float = 1.0
    sign_weight: float = 1.0
    posterior_weight: float = 0.75
    entropy_weight: float = 0.0
    smooth_l1_beta: float = 0.004
    sign_temperature: float = 0.015
    posterior_sigma: float = 0.015
    max_abs_expansion: float = 0.25
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        weights = (
            self.lhr_weight,
            self.expansion_weight,
            self.correlation_weight,
            self.sign_weight,
            self.posterior_weight,
            self.entropy_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v4.28 loss weights must be non-negative")
        if min(self.smooth_l1_beta, self.sign_temperature, self.posterior_sigma, self.epsilon) <= 0.0:
            raise ValueError("v4.28 loss scales must be positive")


def gaussian_scale_target(
    target_log_eta: torch.Tensor,
    candidates: torch.Tensor,
    sigma: float,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    distance = (candidates[None].to(target_log_eta) - target_log_eta[:, None]) / float(sigma)
    probabilities = torch.exp(-0.5 * distance.square())
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def posterior_kl_loss(
    logits: torch.Tensor,
    target_log_eta: torch.Tensor,
    candidates: torch.Tensor,
    *,
    sigma: float,
    epsilon: float,
) -> torch.Tensor:
    target = gaussian_scale_target(target_log_eta, candidates, sigma, epsilon)
    log_prob = torch.log_softmax(logits, dim=-1)
    target_log = target.clamp_min(epsilon).log()
    return (target * (target_log - log_prob)).sum(dim=-1)


def object_event_v4_28_loss(
    output: ObjectEventV428Output,
    delta_t_s: torch.Tensor,
    target_ttc_s: torch.Tensor,
    visible_heights_px: torch.Tensor,
    *,
    config: ObjectEventV428LossConfig,
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
    posterior_each = posterior_kl_loss(
        output.scale_logits,
        target_lhr,
        output.log_scale_candidates,
        sigma=config.posterior_sigma,
        epsilon=config.epsilon,
    )
    signed = torch.where(target_exp >= 0.0, torch.ones_like(target_exp), -torch.ones_like(target_exp))
    sign_each = functional.softplus(-signed * output.expansion / config.sign_temperature)
    components = {
        "lhr": (sample_weight * lhr_each).mean(),
        "expansion": (sample_weight * expansion_each).mean(),
        "correlation": 1.0 - pearson_torch(output.expansion, target_exp),
        "sign": (sample_weight * sign_each).mean(),
        "posterior": (sample_weight * posterior_each).mean(),
        "entropy": output.scale_entropy.mean(),
    }
    total = (
        config.lhr_weight * components["lhr"]
        + config.expansion_weight * components["expansion"]
        + config.correlation_weight * components["correlation"]
        + config.sign_weight * components["sign"]
        + config.posterior_weight * components["posterior"]
        + config.entropy_weight * components["entropy"]
    )
    return total, {**components, "target_log_eta": target_lhr, "target_expansion": target_exp}


__all__ = [
    "ObjectEventV428LossConfig",
    "gaussian_scale_target",
    "object_event_v4_28_loss",
    "posterior_kl_loss",
]
