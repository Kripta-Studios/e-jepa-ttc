"""Differentiable voxel-warp contrast expansion expert."""

from __future__ import annotations

import torch
from torch.nn import functional


def event_contrast_inverse_ttc(
    event_frames: torch.Tensor,
    times_s: torch.Tensor,
    *,
    soft_masks: torch.Tensor | None = None,
    event_window_s: float = 0.1,
    candidates: int = 7,
    maximum_inverse_ttc: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate inverse TTC by softly maximizing warped-event contrast.

    The polarity-separated temporal bins of the latest voxel window are
    radially warped to its final bin for a small fixed set of inverse-TTC
    candidates.  A soft argmax keeps the operation differentiable with
    respect to the event tensor and soft object mask.  This is intentionally
    bounded to one vectorized pass; it is not a slow iterative CMax solver.
    """

    if event_frames.ndim != 5:
        raise ValueError("event_frames must have shape [B,T,C,H,W].")
    if event_frames.shape[2] < 4 or event_frames.shape[2] % 2:
        raise ValueError("event channels must be two polarity groups with >=2 bins.")
    if event_window_s <= 0 or candidates < 2 or maximum_inverse_ttc <= 0:
        raise ValueError("event-window and candidate parameters must be positive.")
    del times_s  # The voxel window has its own fixed temporal support.
    latest = event_frames[:, -1].float()
    bins = latest.shape[1] // 2
    activity = latest[:, :bins].abs() + latest[:, bins:].abs()
    batch, _, height, width = activity.shape
    if soft_masks is not None:
        if soft_masks.ndim == 4:
            masks = soft_masks[:, -1]
        elif soft_masks.ndim == 5 and soft_masks.shape[2] == 1:
            masks = soft_masks[:, -1, 0]
        else:
            raise ValueError("soft_masks must have shape [B,T,H,W] or [B,T,1,H,W].")
        mask = functional.interpolate(
            masks[:, None].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0].clamp(0.0, 1.0)
    else:
        mask = torch.ones(batch, height, width, device=activity.device, dtype=activity.dtype)
    y_axis = torch.linspace(-1.0, 1.0, height, device=activity.device)
    x_axis = torch.linspace(-1.0, 1.0, width, device=activity.device)
    grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1)
    mask_weight = mask.sum(dim=(-1, -2)).clamp_min(1e-6)
    center_x = (mask * grid_x).sum(dim=(-1, -2)) / mask_weight
    center_y = (mask * grid_y).sum(dim=(-1, -2)) / mask_weight
    center = torch.stack((center_x, center_y), dim=-1)
    candidate_q = torch.linspace(
        0.0,
        maximum_inverse_ttc,
        candidates,
        device=activity.device,
        dtype=activity.dtype,
    )
    ages = torch.linspace(
        event_window_s,
        0.0,
        bins,
        device=activity.device,
        dtype=activity.dtype,
    )
    # Under constant approach, older apparent scale relative to the current
    # endpoint is TTC_current / (TTC_current + age) = 1 / (1 + q*age).
    scale = (1.0 / (1.0 + candidate_q[:, None] * ages[None])).clamp(0.25, 1.25)
    centered_grid = base_grid[None, None, None] - center[:, None, None, None, None, :]
    sampling_grid = center[:, None, None, None, None, :] + (
        centered_grid * scale[None, :, :, None, None, None]
    )
    sampling_grid = sampling_grid.expand(batch, candidates, bins, height, width, 2)
    source = activity[:, None].expand(batch, candidates, bins, height, width)
    warped = functional.grid_sample(
        source.reshape(batch * candidates * bins, 1, height, width),
        sampling_grid.reshape(batch * candidates * bins, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch, candidates, bins, height, width)
    image_of_warped_events = warped.sum(dim=2)
    expanded_mask = mask[:, None]
    mean = (image_of_warped_events * expanded_mask).sum(dim=(-1, -2)) / mask_weight[:, None]
    contrast = ((image_of_warped_events - mean[..., None, None]).square() * expanded_mask).sum(
        dim=(-1, -2)
    ) / mask_weight[:, None]
    standardized = (contrast - contrast.mean(dim=-1, keepdim=True)) / contrast.std(
        dim=-1,
        keepdim=True,
        unbiased=False,
    ).clamp_min(1e-5)
    probabilities = functional.softmax(standardized, dim=-1)
    inverse_ttc = (probabilities * candidate_q[None]).sum(dim=-1)
    uniform = 1.0 / candidates
    concentration = ((probabilities.amax(dim=-1) - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
    support = activity.sum(dim=(1, 2, 3))
    support_confidence = 1.0 - torch.exp(-support / max(float(height * width) * 0.01, 1.0))
    confidence = concentration * support_confidence
    inverse_ttc = torch.where(support > 0, inverse_ttc, torch.zeros_like(inverse_ttc))
    return inverse_ttc.to(event_frames.dtype), confidence.to(event_frames.dtype)


__all__ = ["event_contrast_inverse_ttc"]
