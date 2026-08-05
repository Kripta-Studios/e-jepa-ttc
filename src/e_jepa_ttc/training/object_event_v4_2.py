"""Losses for the v4.2 full event-only screen."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_2 import ObjectEventV42Output
from e_jepa_ttc.training.object_event_v4_1 import pearson_torch, target_expansion


@dataclass(frozen=True)
class ObjectEventV42LossConfig:
    expansion_weight: float = 4.0
    correlation_weight: float = 1.0
    ranking_weight: float = 0.25
    sign_weight: float = 0.25
    variance_weight: float = 0.05
    reversal_weight: float = 0.0
    reversal_start_epoch: int = 99
    smooth_l1_beta: float = 0.005
    sign_temperature: float = 0.25
    max_abs_expansion: float = 0.25

    def __post_init__(self) -> None:
        weights = (
            self.expansion_weight,
            self.correlation_weight,
            self.ranking_weight,
            self.sign_weight,
            self.variance_weight,
            self.reversal_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v4.2 loss weights must be non-negative")
        if self.smooth_l1_beta <= 0.0 or self.sign_temperature <= 0.0:
            raise ValueError("v4.2 loss scales must be positive")
        if self.reversal_start_epoch < 0:
            raise ValueError("reversal_start_epoch must be non-negative")


@dataclass
class ObjectEventV42LossOutput:
    total: torch.Tensor
    target: torch.Tensor
    components: dict[str, torch.Tensor]


def _ranking(prediction: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    pred_delta = prediction[:, None] - prediction[None, :]
    target_delta = target[:, None] - target[None, :]
    mask = ~torch.eye(prediction.shape[0], dtype=torch.bool, device=prediction.device)
    return functional.smooth_l1_loss(pred_delta[mask], target_delta[mask], beta=beta)


def object_event_v4_2_loss(
    output: ObjectEventV42Output,
    delta_t_s: torch.Tensor,
    target_ttc_s: torch.Tensor,
    *,
    epoch: int,
    config: ObjectEventV42LossConfig,
) -> ObjectEventV42LossOutput:
    target = target_expansion(delta_t_s, target_ttc_s, config.max_abs_expansion)
    label = (target >= 0.0).to(output.raw_score.dtype)
    components = {
        "expansion": functional.smooth_l1_loss(
            output.expansion, target, beta=config.smooth_l1_beta
        ),
        "correlation": 1.0 - pearson_torch(output.expansion, target),
        "ranking": _ranking(output.expansion, target, config.smooth_l1_beta),
        "sign": functional.binary_cross_entropy_with_logits(
            output.raw_score / config.sign_temperature, label
        ),
        "variance": (
            output.expansion.float().std(unbiased=False)
            - target.float().std(unbiased=False)
        ).abs(),
        "reversal": 0.5
        * (
            functional.smooth_l1_loss(
                output.reverse_expansion,
                -target,
                beta=config.smooth_l1_beta,
            )
            + output.reversal_consistency_error.mean()
        ),
    }
    reversal_scale = (
        config.reversal_weight if epoch >= config.reversal_start_epoch else 0.0
    )
    total = (
        config.expansion_weight * components["expansion"]
        + config.correlation_weight * components["correlation"]
        + config.ranking_weight * components["ranking"]
        + config.sign_weight * components["sign"]
        + config.variance_weight * components["variance"]
        + reversal_scale * components["reversal"]
    )
    return ObjectEventV42LossOutput(total=total, target=target, components=components)


__all__ = [
    "ObjectEventV42LossConfig",
    "ObjectEventV42LossOutput",
    "object_event_v4_2_loss",
]
