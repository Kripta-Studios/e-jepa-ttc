"""Geometry-only supervision helpers for Object Event TTC v4.20."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_19 import dense_flow_scores


@dataclass(frozen=True)
class ObjectEventV420LossConfig:
    flow_weight: float = 1.0
    divergence_weight: float = 0.5
    residual_weight: float = 0.02
    smooth_l1_beta: float = 0.10
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if min(self.flow_weight, self.divergence_weight, self.residual_weight) < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.smooth_l1_beta <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("loss scales must be positive")


def box_affine_pseudoflow(
    boxes_xyxy: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
    first_index: int,
    second_index: int,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each grid point in the first box to the same normalized point in the second."""
    if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must be [B,T,4]")
    if not (0 <= first_index < boxes_xyxy.shape[1] and 0 <= second_index < boxes_xyxy.shape[1]):
        raise IndexError("endpoint index out of range")
    if min(source_height, source_width, target_height, target_width) <= 1:
        raise ValueError("spatial sizes must exceed one")

    boxes = boxes_xyxy.float().clone()
    boxes[..., (0, 2)] *= float(target_width) / float(source_width)
    boxes[..., (1, 3)] *= float(target_height) / float(source_height)
    first = boxes[:, first_index]
    second = boxes[:, second_index]

    x1a, y1a, x1b, y1b = first.unbind(dim=1)
    x2a, y2a, x2b, y2b = second.unbind(dim=1)
    w1 = (x1b - x1a).clamp_min(epsilon)
    h1 = (y1b - y1a).clamp_min(epsilon)
    w2 = (x2b - x2a).clamp_min(epsilon)
    h2 = (y2b - y2a).clamp_min(epsilon)
    cx1 = 0.5 * (x1a + x1b)
    cy1 = 0.5 * (y1a + y1b)
    cx2 = 0.5 * (x2a + x2b)
    cy2 = 0.5 * (y2a + y2b)

    yy, xx = torch.meshgrid(
        torch.arange(target_height, device=boxes.device, dtype=boxes.dtype) + 0.5,
        torch.arange(target_width, device=boxes.device, dtype=boxes.dtype) + 0.5,
        indexing="ij",
    )
    xx = xx[None]
    yy = yy[None]
    mask = (
        (xx >= x1a[:, None, None])
        & (xx <= x1b[:, None, None])
        & (yy >= y1a[:, None, None])
        & (yy <= y1b[:, None, None])
    ).to(boxes.dtype)

    mapped_x = cx2[:, None, None] + (w2 / w1)[:, None, None] * (xx - cx1[:, None, None])
    mapped_y = cy2[:, None, None] + (h2 / h1)[:, None, None] * (yy - cy1[:, None, None])
    flow = torch.stack((mapped_x - xx, mapped_y - yy), dim=1)
    return flow, mask


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    epsilon: float,
) -> torch.Tensor:
    error = functional.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    weights = mask[:, None].expand_as(error)
    return (error * weights).sum() / weights.sum().clamp_min(epsilon)


def pseudoflow_loss(
    refined_forward: torch.Tensor,
    residual_forward: torch.Tensor,
    refined_reverse: torch.Tensor,
    residual_reverse: torch.Tensor,
    target_forward: torch.Tensor,
    mask_forward: torch.Tensor,
    target_reverse: torch.Tensor,
    mask_reverse: torch.Tensor,
    *,
    config: ObjectEventV420LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Geometry-only flow loss. No TTC label enters this objective."""
    ones_f = torch.ones_like(mask_forward)
    ones_r = torch.ones_like(mask_reverse)
    div_pf, _, _ = dense_flow_scores(
        refined_forward[:, 0], refined_forward[:, 1], mask_forward, ones_f,
        foreground_floor=config.epsilon, confidence_floor=config.epsilon, epsilon=config.epsilon,
    )
    div_tf, _, _ = dense_flow_scores(
        target_forward[:, 0], target_forward[:, 1], mask_forward, ones_f,
        foreground_floor=config.epsilon, confidence_floor=config.epsilon, epsilon=config.epsilon,
    )
    div_pr, _, _ = dense_flow_scores(
        refined_reverse[:, 0], refined_reverse[:, 1], mask_reverse, ones_r,
        foreground_floor=config.epsilon, confidence_floor=config.epsilon, epsilon=config.epsilon,
    )
    div_tr, _, _ = dense_flow_scores(
        target_reverse[:, 0], target_reverse[:, 1], mask_reverse, ones_r,
        foreground_floor=config.epsilon, confidence_floor=config.epsilon, epsilon=config.epsilon,
    )
    components = {
        "flow": 0.5 * (
            _masked_smooth_l1(refined_forward, target_forward, mask_forward, beta=config.smooth_l1_beta, epsilon=config.epsilon)
            + _masked_smooth_l1(refined_reverse, target_reverse, mask_reverse, beta=config.smooth_l1_beta, epsilon=config.epsilon)
        ),
        "divergence": 0.5 * (
            functional.smooth_l1_loss(div_pf, div_tf, beta=config.smooth_l1_beta)
            + functional.smooth_l1_loss(div_pr, div_tr, beta=config.smooth_l1_beta)
        ),
        "residual": 0.5 * (residual_forward.square().mean() + residual_reverse.square().mean()),
    }
    total = (
        config.flow_weight * components["flow"]
        + config.divergence_weight * components["divergence"]
        + config.residual_weight * components["residual"]
    )
    return total, components


__all__ = [
    "ObjectEventV420LossConfig",
    "box_affine_pseudoflow",
    "pseudoflow_loss",
]
