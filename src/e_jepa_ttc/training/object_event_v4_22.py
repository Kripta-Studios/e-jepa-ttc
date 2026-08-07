"""Direct geometry-encoder pseudoflow losses for Object Event TTC v4.22."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_19 import dense_flow_scores


@dataclass(frozen=True)
class ObjectEventV422LossConfig:
    flow_weight: float = 1.0
    divergence_weight: float = 1.0
    vertical_scale_weight: float = 1.0
    encoder_anchor_weight: float = 0.02
    smooth_l1_beta: float = 0.10
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if min(self.flow_weight, self.divergence_weight, self.vertical_scale_weight, self.encoder_anchor_weight) < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.smooth_l1_beta <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("loss scales must be positive")


def masked_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    epsilon: float,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("flow tensors must align as [B,2,H,W]")
    if mask.shape != prediction[:, 0].shape:
        raise ValueError("mask must align with flow spatial dimensions")
    error = functional.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    weights = mask[:, None].expand_as(error)
    return (error * weights).sum() / weights.sum().clamp_min(epsilon)


def relative_parameter_anchor(
    named_parameters: Mapping[str, torch.Tensor],
    initial_parameters: Mapping[str, torch.Tensor],
    *,
    epsilon: float,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for name, parameter in named_parameters.items():
        if name not in initial_parameters:
            raise KeyError(f"missing initial parameter {name}")
        initial = initial_parameters[name].to(device=parameter.device, dtype=parameter.dtype)
        scale = initial.square().mean().clamp_min(epsilon)
        terms.append((parameter - initial).square().mean() / scale)
    if not terms:
        raise ValueError("no parameters supplied to anchor")
    return torch.stack(terms).mean()



def vertical_log_scale_from_flow(
    flow: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Estimate log vertical scale from an affine-like dense flow field.

    For v(y) = translation + (s_y - 1) * (y - c_y), weighted least
    squares recovers slope s_y - 1. The log-scale is log(s_y).
    """
    if flow.ndim != 4 or flow.shape[1] != 2 or mask.shape != flow[:, 0].shape:
        raise ValueError("flow/mask shapes must be [B,2,H,W] and [B,H,W]")
    _, _, height, width = flow.shape
    del width
    y = torch.arange(height, device=flow.device, dtype=flow.dtype).view(1, height, 1)
    weights = mask.to(dtype=flow.dtype).clamp_min(0.0)
    norm = weights.sum(dim=(-2, -1)).clamp_min(epsilon)
    y_mean = (weights * y).sum(dim=(-2, -1)) / norm
    v = flow[:, 1]
    v_mean = (weights * v).sum(dim=(-2, -1)) / norm
    yc = y - y_mean[:, None, None]
    vc = v - v_mean[:, None, None]
    covariance = (weights * yc * vc).sum(dim=(-2, -1)) / norm
    variance = (weights * yc.square()).sum(dim=(-2, -1)) / norm
    slope = covariance / variance.clamp_min(epsilon)
    scale = (1.0 + slope).clamp_min(0.05)
    return torch.log(scale)

def encoder_pseudoflow_loss(
    forward_flow: torch.Tensor,
    reverse_flow: torch.Tensor,
    target_forward: torch.Tensor,
    mask_forward: torch.Tensor,
    target_reverse: torch.Tensor,
    mask_reverse: torch.Tensor,
    named_parameters: Mapping[str, torch.Tensor],
    initial_parameters: Mapping[str, torch.Tensor],
    *,
    config: ObjectEventV422LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ones_f = torch.ones_like(mask_forward)
    ones_r = torch.ones_like(mask_reverse)
    pred_div_f, _, _ = dense_flow_scores(
        forward_flow[:, 0], forward_flow[:, 1], mask_forward, ones_f,
        foreground_floor=config.epsilon, confidence_floor=config.epsilon, epsilon=config.epsilon,
    )
    target_div_f, _, _ = dense_flow_scores(
        target_forward[:, 0], target_forward[:, 1], mask_forward, ones_f,
        foreground_floor=config.epsilon, confidence_floor=config.epsilon, epsilon=config.epsilon,
    )
    pred_div_r, _, _ = dense_flow_scores(
        reverse_flow[:, 0], reverse_flow[:, 1], mask_reverse, ones_r,
        foreground_floor=config.epsilon, confidence_floor=config.epsilon, epsilon=config.epsilon,
    )
    target_div_r, _, _ = dense_flow_scores(
        target_reverse[:, 0], target_reverse[:, 1], mask_reverse, ones_r,
        foreground_floor=config.epsilon, confidence_floor=config.epsilon, epsilon=config.epsilon,
    )
    pred_vertical_f = vertical_log_scale_from_flow(forward_flow, mask_forward, epsilon=config.epsilon)
    target_vertical_f = vertical_log_scale_from_flow(target_forward, mask_forward, epsilon=config.epsilon)
    pred_vertical_r = vertical_log_scale_from_flow(reverse_flow, mask_reverse, epsilon=config.epsilon)
    target_vertical_r = vertical_log_scale_from_flow(target_reverse, mask_reverse, epsilon=config.epsilon)
    flow = 0.5 * (
        masked_flow_loss(forward_flow, target_forward, mask_forward, beta=config.smooth_l1_beta, epsilon=config.epsilon)
        + masked_flow_loss(reverse_flow, target_reverse, mask_reverse, beta=config.smooth_l1_beta, epsilon=config.epsilon)
    )
    divergence = 0.5 * (
        functional.smooth_l1_loss(pred_div_f, target_div_f, beta=config.smooth_l1_beta)
        + functional.smooth_l1_loss(pred_div_r, target_div_r, beta=config.smooth_l1_beta)
    )
    vertical_scale = 0.5 * (
        functional.smooth_l1_loss(pred_vertical_f, target_vertical_f, beta=config.smooth_l1_beta)
        + functional.smooth_l1_loss(pred_vertical_r, target_vertical_r, beta=config.smooth_l1_beta)
    )
    anchor = relative_parameter_anchor(named_parameters, initial_parameters, epsilon=config.epsilon)
    total = (
        config.flow_weight * flow
        + config.divergence_weight * divergence
        + config.vertical_scale_weight * vertical_scale
        + config.encoder_anchor_weight * anchor
    )
    return total, {
        "flow": flow,
        "divergence": divergence,
        "vertical_scale": vertical_scale,
        "encoder_anchor": anchor,
    }


__all__ = [
    "ObjectEventV422LossConfig",
    "encoder_pseudoflow_loss",
    "masked_flow_loss",
    "relative_parameter_anchor",
    "vertical_log_scale_from_flow",
]
