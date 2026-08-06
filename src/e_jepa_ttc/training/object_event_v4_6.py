"""Losses and supervision helpers for Object Event TTC v4.6."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_6 import ObjectEventV46Output
from e_jepa_ttc.training.object_event_v4_1 import pearson_torch, target_expansion


@dataclass(frozen=True)
class ObjectEventV46LossConfig:
    fused_mid_weight: float = 4.0
    fused_expansion_weight: float = 1.0
    fused_correlation_weight: float = 0.5
    height_ratio_weight: float = 2.5
    height_expansion_weight: float = 0.5
    foreground_bce_weight: float = 1.0
    foreground_dice_weight: float = 1.0
    sign_weight: float = 0.75
    blend_weight: float = 0.05
    smooth_l1_beta: float = 0.005
    sign_temperature: float = 0.02
    max_abs_expansion: float = 0.25
    crucial_weight: float = 0.5
    small_weight: float = 0.3
    large_weight: float = 0.1
    negative_weight: float = 0.1

    def __post_init__(self) -> None:
        weights = (
            self.fused_mid_weight,
            self.fused_expansion_weight,
            self.fused_correlation_weight,
            self.height_ratio_weight,
            self.height_expansion_weight,
            self.foreground_bce_weight,
            self.foreground_dice_weight,
            self.sign_weight,
            self.blend_weight,
        )
        if min(weights) < 0.0:
            raise ValueError("v4.6 loss weights must be non-negative")
        if self.smooth_l1_beta <= 0.0 or self.sign_temperature <= 0.0:
            raise ValueError("v4.6 loss scales must be positive")


@dataclass
class ObjectEventV46LossOutput:
    total: torch.Tensor
    target_expansion: torch.Tensor
    target_log_eta: torch.Tensor
    target_height_log_eta: torch.Tensor
    foreground_targets: torch.Tensor
    components: dict[str, torch.Tensor]


def boxes_to_feature_masks(
    boxes_xyxy: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    endpoint_indices: tuple[int, ...] = (1, 2),
) -> torch.Tensor:
    """Rasterise supervision-only boxes onto an encoded feature grid.

    Boxes may extend outside the common ROI.  Clipping happens only for the mask
    target and never changes the event input.
    """

    if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must be [B,T,4]")
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("mask dimensions must be positive")
    selected = boxes_xyxy[:, list(endpoint_indices)].float()
    batch, endpoints, _ = selected.shape
    masks = torch.zeros(
        batch,
        endpoints,
        target_height,
        target_width,
        dtype=torch.float32,
        device=boxes_xyxy.device,
    )
    scale_x = target_width / float(source_width)
    scale_y = target_height / float(source_height)
    for b in range(batch):
        for e in range(endpoints):
            x0, y0, x1, y1 = selected[b, e]
            left = int(torch.floor(x0 * scale_x).clamp(0, target_width - 1).item())
            top = int(torch.floor(y0 * scale_y).clamp(0, target_height - 1).item())
            right = int(torch.ceil(x1 * scale_x).clamp(1, target_width).item())
            bottom = int(torch.ceil(y1 * scale_y).clamp(1, target_height).item())
            if right > left and bottom > top:
                masks[b, e, top:bottom, left:right] = 1.0
    return masks


def official_range_weights(
    target_ttc_s: torch.Tensor,
    config: ObjectEventV46LossConfig,
) -> torch.Tensor:
    negative = target_ttc_s < 0.0
    absolute = target_ttc_s.abs()
    crucial = (~negative) & (absolute <= 1.0)
    small = (~negative) & (absolute > 1.0) & (absolute <= 3.0)
    large = (~negative) & (absolute > 3.0)
    weights = torch.empty_like(target_ttc_s, dtype=torch.float32)
    weights[crucial] = config.crucial_weight
    weights[small] = config.small_weight
    weights[large] = config.large_weight
    weights[negative] = config.negative_weight
    return weights / weights.mean().clamp_min(1.0e-8)


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def object_event_v4_6_loss(
    output: ObjectEventV46Output,
    delta_t_s: torch.Tensor,
    target_ttc_s: torch.Tensor,
    visible_heights_px: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    config: ObjectEventV46LossConfig,
) -> ObjectEventV46LossOutput:
    if visible_heights_px.ndim != 2 or visible_heights_px.shape[1] != 2:
        raise ValueError("visible_heights_px must be [B,2]")
    target = target_expansion(delta_t_s, target_ttc_s, config.max_abs_expansion)
    target_log_eta = torch.log1p(-target)
    target_height_log_eta = (
        torch.log(visible_heights_px[:, 0].clamp_min(1.0e-4))
        - torch.log(visible_heights_px[:, 1].clamp_min(1.0e-4))
    )
    map_h, map_w = output.foreground_logits.shape[-2:]
    foreground_targets = boxes_to_feature_masks(
        boxes_xyxy,
        source_height=source_height,
        source_width=source_width,
        target_height=map_h,
        target_width=map_w,
    )
    endpoint_logits = output.foreground_logits[:, 1:3]
    range_weight = official_range_weights(target_ttc_s, config)
    mid_error = functional.smooth_l1_loss(
        output.fused_log_eta,
        target_log_eta,
        beta=config.smooth_l1_beta,
        reduction="none",
    )
    label = (target >= 0.0).to(output.raw_score.dtype)
    components = {
        "fused_mid": (range_weight * mid_error).mean(),
        "fused_expansion": functional.smooth_l1_loss(
            output.expansion, target, beta=config.smooth_l1_beta
        ),
        "fused_correlation": 1.0 - pearson_torch(output.expansion, target),
        "height_ratio": functional.smooth_l1_loss(
            output.height_log_eta,
            target_height_log_eta,
            beta=config.smooth_l1_beta,
        ),
        "height_expansion": functional.smooth_l1_loss(
            output.height_expansion, target, beta=config.smooth_l1_beta
        ),
        "foreground_bce": functional.binary_cross_entropy_with_logits(
            endpoint_logits, foreground_targets
        ),
        "foreground_dice": _dice_loss(endpoint_logits, foreground_targets),
        "sign": functional.binary_cross_entropy_with_logits(
            output.raw_score / config.sign_temperature, label
        ),
        "blend": output.blend.mean(),
    }
    total = (
        config.fused_mid_weight * components["fused_mid"]
        + config.fused_expansion_weight * components["fused_expansion"]
        + config.fused_correlation_weight * components["fused_correlation"]
        + config.height_ratio_weight * components["height_ratio"]
        + config.height_expansion_weight * components["height_expansion"]
        + config.foreground_bce_weight * components["foreground_bce"]
        + config.foreground_dice_weight * components["foreground_dice"]
        + config.sign_weight * components["sign"]
        + config.blend_weight * components["blend"]
    )
    return ObjectEventV46LossOutput(
        total=total,
        target_expansion=target,
        target_log_eta=target_log_eta,
        target_height_log_eta=target_height_log_eta,
        foreground_targets=foreground_targets,
        components=components,
    )


__all__ = [
    "ObjectEventV46LossConfig",
    "ObjectEventV46LossOutput",
    "boxes_to_feature_masks",
    "object_event_v4_6_loss",
    "official_range_weights",
]
