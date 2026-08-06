"""Losses for Object Event TTC v4.8 dense temporal log-scale field."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_8 import ObjectEventV48Output
from e_jepa_ttc.training.object_event_v4_1 import pearson_torch, target_expansion
from e_jepa_ttc.training.object_event_v4_6 import boxes_to_feature_masks


@dataclass(frozen=True)
class ObjectEventV48LossConfig:
    pooled_log_eta_weight: float = 4.0
    dense_log_eta_weight: float = 2.0
    expansion_weight: float = 1.0
    correlation_weight: float = 1.0
    sign_weight: float = 0.75
    confidence_weight: float = 0.25
    background_zero_weight: float = 0.20
    total_variation_weight: float = 0.05
    smooth_l1_beta: float = 0.005
    sign_temperature: float = 0.02
    max_abs_expansion: float = 0.25
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        weights = (
            self.pooled_log_eta_weight,
            self.dense_log_eta_weight,
            self.expansion_weight,
            self.correlation_weight,
            self.sign_weight,
            self.confidence_weight,
            self.background_zero_weight,
            self.total_variation_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v4.8 loss weights must be non-negative")
        if min(self.smooth_l1_beta, self.sign_temperature, self.epsilon) <= 0.0:
            raise ValueError("v4.8 loss scales must be positive")


@dataclass
class ObjectEventV48LossOutput:
    total: torch.Tensor
    target_expansion: torch.Tensor
    target_log_eta: torch.Tensor
    foreground_intersection: torch.Tensor
    components: dict[str, torch.Tensor]


def _masked_mean(values: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(epsilon)


def object_event_v4_8_loss(
    output: ObjectEventV48Output,
    delta_t_s: torch.Tensor,
    target_ttc_s: torch.Tensor,
    visible_heights_px: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    config: ObjectEventV48LossConfig,
) -> ObjectEventV48LossOutput:
    if visible_heights_px.ndim != 2 or visible_heights_px.shape[1] != 2:
        raise ValueError("visible_heights_px must be [B,2]")
    target = target_expansion(delta_t_s, target_ttc_s, config.max_abs_expansion)
    target_log_eta = (
        torch.log(visible_heights_px[:, 0].clamp_min(1.0e-4))
        - torch.log(visible_heights_px[:, 1].clamp_min(1.0e-4))
    )
    map_h, map_w = output.local_log_eta.shape[-2:]
    masks = boxes_to_feature_masks(
        boxes_xyxy,
        source_height=source_height,
        source_width=source_width,
        target_height=map_h,
        target_width=map_w,
    )
    endpoint = masks[:, 0:2]
    intersection = endpoint[:, 0] * endpoint[:, 1]
    union = endpoint.amax(dim=1)
    target_map = target_log_eta[:, None, None].expand_as(output.local_log_eta)
    dense_error = functional.smooth_l1_loss(
        output.local_log_eta,
        target_map,
        beta=config.smooth_l1_beta,
        reduction="none",
    )
    background = 1.0 - union
    tv_y = (output.local_log_eta[:, 1:] - output.local_log_eta[:, :-1]).abs()
    tv_x = (output.local_log_eta[:, :, 1:] - output.local_log_eta[:, :, :-1]).abs()
    labels = (target >= 0.0).to(output.raw_score.dtype)
    components = {
        "pooled_log_eta": functional.smooth_l1_loss(
            output.pooled_log_eta,
            target_log_eta,
            beta=config.smooth_l1_beta,
        ),
        "dense_log_eta": _masked_mean(dense_error, intersection, config.epsilon),
        "expansion": functional.smooth_l1_loss(
            output.expansion,
            target,
            beta=config.smooth_l1_beta,
        ),
        "correlation": 1.0 - pearson_torch(output.expansion, target),
        "sign": functional.binary_cross_entropy_with_logits(
            output.raw_score / config.sign_temperature,
            labels,
        ),
        "confidence": functional.binary_cross_entropy_with_logits(
            output.confidence_logits,
            intersection,
        ),
        "background_zero": _masked_mean(
            output.local_log_eta.abs(),
            background,
            config.epsilon,
        ),
        "total_variation": 0.5 * (tv_y.mean() + tv_x.mean()),
    }
    total = (
        config.pooled_log_eta_weight * components["pooled_log_eta"]
        + config.dense_log_eta_weight * components["dense_log_eta"]
        + config.expansion_weight * components["expansion"]
        + config.correlation_weight * components["correlation"]
        + config.sign_weight * components["sign"]
        + config.confidence_weight * components["confidence"]
        + config.background_zero_weight * components["background_zero"]
        + config.total_variation_weight * components["total_variation"]
    )
    return ObjectEventV48LossOutput(
        total=total,
        target_expansion=target,
        target_log_eta=target_log_eta,
        foreground_intersection=intersection,
        components=components,
    )


__all__ = [
    "ObjectEventV48LossConfig",
    "ObjectEventV48LossOutput",
    "object_event_v4_8_loss",
]
