"""Losses and metrics for the event-only v4.1 learnability diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Output


@dataclass(frozen=True)
class ObjectEventV41LossConfig:
    expansion_weight: float = 4.0
    encoded_aux_weight: float = 0.25
    activity_aux_weight: float = 0.10
    correlation_weight: float = 1.0
    ranking_weight: float = 0.50
    sign_weight: float = 0.50
    variance_weight: float = 0.10
    reversal_weight: float = 0.05
    reversal_start_step: int = 160
    smooth_l1_beta: float = 0.005
    sign_temperature: float = 0.02
    max_abs_expansion: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.expansion_weight,
            self.encoded_aux_weight,
            self.activity_aux_weight,
            self.correlation_weight,
            self.ranking_weight,
            self.sign_weight,
            self.variance_weight,
            self.reversal_weight,
        )
        if min(values) < 0.0:
            raise ValueError("Loss weights must be non-negative")
        if self.reversal_start_step < 0:
            raise ValueError("reversal_start_step must be non-negative")
        if self.smooth_l1_beta <= 0.0 or self.sign_temperature <= 0.0:
            raise ValueError("Loss scales must be positive")


@dataclass
class ObjectEventV41LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    target_expansion: torch.Tensor


def target_expansion(delta_t_s: torch.Tensor, target_ttc_s: torch.Tensor, maximum: float) -> torch.Tensor:
    if delta_t_s.shape != target_ttc_s.shape:
        raise ValueError("delta_t_s and target_ttc_s must share shape")
    if bool((target_ttc_s == 0).any()):
        raise ValueError("TTC targets must be non-zero")
    return (delta_t_s / target_ttc_s).clamp(-maximum * 0.999, maximum * 0.999)


def pearson_torch(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.float() - first.float().mean()
    second = second.float() - second.float().mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    return (first * second).sum() / denominator.clamp_min(1.0e-8)


def _regression(prediction: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    return functional.smooth_l1_loss(prediction, target, beta=beta)


def _ranking(prediction: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    if prediction.numel() < 2:
        return prediction.new_zeros(())
    prediction_delta = prediction[:, None] - prediction[None, :]
    target_delta = target[:, None] - target[None, :]
    mask = ~torch.eye(prediction.shape[0], dtype=torch.bool, device=prediction.device)
    return functional.smooth_l1_loss(
        prediction_delta[mask], target_delta[mask], beta=beta
    )


def _sign_loss(raw_score: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    label = (target >= 0.0).to(raw_score.dtype)
    return functional.binary_cross_entropy_with_logits(raw_score / temperature, label)


def object_event_v4_1_loss(
    output: ObjectEventV41Output,
    delta_t_s: torch.Tensor,
    target_ttc_s: torch.Tensor,
    *,
    step: int,
    config: ObjectEventV41LossConfig,
) -> ObjectEventV41LossOutput:
    target = target_expansion(delta_t_s, target_ttc_s, config.max_abs_expansion)
    correlation = pearson_torch(output.expansion, target)
    components = {
        "expansion": _regression(output.expansion, target, config.smooth_l1_beta),
        "encoded_aux": _regression(
            output.encoded_expansion, target, config.smooth_l1_beta
        ),
        "activity_aux": _regression(
            output.activity_expansion, target, config.smooth_l1_beta
        ),
        "correlation": 1.0 - correlation,
        "ranking": _ranking(output.expansion, target, config.smooth_l1_beta),
        "sign": _sign_loss(output.raw_score, target, config.sign_temperature),
        "variance": (output.expansion.float().std(unbiased=False) - target.float().std(unbiased=False)).abs(),
        "reversal": 0.5
        * (
            _regression(
                output.reverse_expansion,
                -target,
                config.smooth_l1_beta,
            )
            + output.reversal_consistency_error.mean()
        ),
    }
    reversal_scale = config.reversal_weight if step >= config.reversal_start_step else 0.0
    total = (
        config.expansion_weight * components["expansion"]
        + config.encoded_aux_weight * components["encoded_aux"]
        + config.activity_aux_weight * components["activity_aux"]
        + config.correlation_weight * components["correlation"]
        + config.ranking_weight * components["ranking"]
        + config.sign_weight * components["sign"]
        + config.variance_weight * components["variance"]
        + reversal_scale * components["reversal"]
    )
    return ObjectEventV41LossOutput(total=total, components=components, target_expansion=target)


__all__ = [
    "ObjectEventV41LossConfig",
    "ObjectEventV41LossOutput",
    "object_event_v4_1_loss",
    "pearson_torch",
    "target_expansion",
]
