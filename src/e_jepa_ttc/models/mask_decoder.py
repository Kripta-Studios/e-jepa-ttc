"""Foreground training losses for the event-only mask decoder."""

from __future__ import annotations

import torch
from torch.nn import functional


def boxes_to_soft_masks(
    boxes_xyxy: torch.Tensor,
    spatial_shape: tuple[int, int],
    *,
    edge_softness: float = 40.0,
) -> torch.Tensor:
    """Rasterize normalized boxes into differentiable soft targets."""

    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must have shape [B,4].")
    height, width = spatial_shape
    y = torch.linspace(0.0, 1.0, height, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    x = torch.linspace(0.0, 1.0, width, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    x0, y0, x1, y1 = boxes_xyxy.unbind(dim=-1)
    horizontal = torch.sigmoid((xx - x0[:, None, None]) * edge_softness)
    horizontal = horizontal * torch.sigmoid((x1[:, None, None] - xx) * edge_softness)
    vertical = torch.sigmoid((yy - y0[:, None, None]) * edge_softness)
    vertical = vertical * torch.sigmoid((y1[:, None, None] - yy) * edge_softness)
    return horizontal * vertical


def foreground_mask_loss(
    mask_logits: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return BCE, Dice and their sum."""

    if mask_logits.shape != target_mask.shape:
        raise ValueError("mask_logits and target_mask must have matching shapes.")
    target = target_mask.to(mask_logits.dtype).clamp(0.0, 1.0)
    binary_cross_entropy = functional.binary_cross_entropy_with_logits(mask_logits, target)
    probability = torch.sigmoid(mask_logits)
    intersection = (probability * target).sum(dim=(-1, -2))
    denominator = probability.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return {
        "mask_bce": binary_cross_entropy,
        "mask_dice": dice,
        "mask_total": binary_cross_entropy + dice,
    }


__all__ = ["boxes_to_soft_masks", "foreground_mask_loss"]
