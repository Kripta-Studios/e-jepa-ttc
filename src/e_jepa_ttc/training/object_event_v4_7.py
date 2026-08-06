"""Losses for Object Event TTC v4.7 high-resolution foreground extent."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_7 import ObjectEventV47Output
from e_jepa_ttc.training.object_event_v4_1 import pearson_torch, target_expansion
from e_jepa_ttc.training.object_event_v4_6 import boxes_to_feature_masks


@dataclass(frozen=True)
class ObjectEventV47LossConfig:
    height_ratio_weight: float = 4.0
    expansion_weight: float = 1.0
    correlation_weight: float = 0.75
    foreground_bce_weight: float = 1.0
    foreground_dice_weight: float = 1.0
    row_profile_weight: float = 0.5
    sign_weight: float = 0.75
    smooth_l1_beta: float = 0.005
    sign_temperature: float = 0.02
    max_abs_expansion: float = 0.25

    def __post_init__(self) -> None:
        weights = (
            self.height_ratio_weight,
            self.expansion_weight,
            self.correlation_weight,
            self.foreground_bce_weight,
            self.foreground_dice_weight,
            self.row_profile_weight,
            self.sign_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v4.7 loss weights must be non-negative")
        if min(self.smooth_l1_beta, self.sign_temperature) <= 0.0:
            raise ValueError("v4.7 loss scales must be positive")


@dataclass
class ObjectEventV47LossOutput:
    total: torch.Tensor
    target_expansion: torch.Tensor
    target_height_log_eta: torch.Tensor
    foreground_targets: torch.Tensor
    components: dict[str, torch.Tensor]


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def object_event_v4_7_loss(
    output: ObjectEventV47Output,
    delta_t_s: torch.Tensor,
    target_ttc_s: torch.Tensor,
    visible_heights_px: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    config: ObjectEventV47LossConfig,
) -> ObjectEventV47LossOutput:
    if visible_heights_px.ndim != 2 or visible_heights_px.shape[1] != 2:
        raise ValueError("visible_heights_px must be [B,2]")
    target = target_expansion(delta_t_s, target_ttc_s, config.max_abs_expansion)
    target_height_log_eta = (
        torch.log(visible_heights_px[:, 0].clamp_min(1.0e-4))
        - torch.log(visible_heights_px[:, 1].clamp_min(1.0e-4))
    )
    map_h, map_w = output.foreground_logits.shape[-2:]
    masks = boxes_to_feature_masks(
        boxes_xyxy,
        source_height=source_height,
        source_width=source_width,
        target_height=map_h,
        target_width=map_w,
    )
    endpoint_logits = output.foreground_logits[:, 1:3]
    endpoint_rows = output.row_profiles[:, 1:3]
    target_rows = masks.mean(dim=-1)
    labels = (target >= 0.0).to(output.raw_score.dtype)
    components = {
        "height_ratio": functional.smooth_l1_loss(
            output.height_log_eta,
            target_height_log_eta,
            beta=config.smooth_l1_beta,
        ),
        "expansion": functional.smooth_l1_loss(
            output.expansion,
            target,
            beta=config.smooth_l1_beta,
        ),
        "correlation": 1.0 - pearson_torch(output.expansion, target),
        "foreground_bce": functional.binary_cross_entropy_with_logits(
            endpoint_logits, masks
        ),
        "foreground_dice": _dice_loss(endpoint_logits, masks),
        "row_profile": functional.smooth_l1_loss(endpoint_rows, target_rows, beta=0.02),
        "sign": functional.binary_cross_entropy_with_logits(
            output.raw_score / config.sign_temperature,
            labels,
        ),
    }
    total = (
        config.height_ratio_weight * components["height_ratio"]
        + config.expansion_weight * components["expansion"]
        + config.correlation_weight * components["correlation"]
        + config.foreground_bce_weight * components["foreground_bce"]
        + config.foreground_dice_weight * components["foreground_dice"]
        + config.row_profile_weight * components["row_profile"]
        + config.sign_weight * components["sign"]
    )
    return ObjectEventV47LossOutput(
        total=total,
        target_expansion=target,
        target_height_log_eta=target_height_log_eta,
        foreground_targets=masks,
        components=components,
    )


__all__ = [
    "ObjectEventV47LossConfig",
    "ObjectEventV47LossOutput",
    "object_event_v4_7_loss",
]
