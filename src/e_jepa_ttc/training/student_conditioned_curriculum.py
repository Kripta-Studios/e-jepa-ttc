"""GT → mixed → predicted ROI curriculum."""

from __future__ import annotations

import torch


def predicted_roi_probability(
    epoch: int,
    total_epochs: int,
    *,
    gt_fraction: float = 0.25,
    mixed_fraction: float = 0.50,
) -> float:
    """Return the scheduled predicted-ROI probability."""

    if total_epochs <= 0 or epoch < 0 or epoch > total_epochs:
        raise ValueError("Invalid epoch range.")
    progress = epoch / total_epochs
    if progress <= gt_fraction:
        return 0.0
    if progress >= gt_fraction + mixed_fraction:
        return 1.0
    return (progress - gt_fraction) / mixed_fraction


def select_curriculum_boxes(
    ground_truth_boxes: torch.Tensor,
    predicted_boxes: torch.Tensor,
    *,
    predicted_probability: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose predicted boxes per sample without leaking any future annotation."""

    if ground_truth_boxes.shape != predicted_boxes.shape:
        raise ValueError("Ground-truth and predicted box shapes must match.")
    if not 0.0 <= predicted_probability <= 1.0:
        raise ValueError("predicted_probability must lie in [0,1].")
    batch = ground_truth_boxes.shape[0]
    selected = (
        torch.rand(batch, device=ground_truth_boxes.device, generator=generator)
        < predicted_probability
    )
    mask_shape = (batch,) + (1,) * (ground_truth_boxes.ndim - 1)
    boxes = torch.where(selected.view(mask_shape), predicted_boxes, ground_truth_boxes)
    return boxes, selected


__all__ = ["predicted_roi_probability", "select_curriculum_boxes"]
