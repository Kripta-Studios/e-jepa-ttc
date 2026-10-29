"""Quality scores for causal geometric tracks."""

from __future__ import annotations

import torch


def geometry_track_confidence(
    boxes_xyxy: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Score box validity, temporal coverage and center/shape smoothness."""

    if boxes_xyxy.ndim != 4 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must have shape [B,T,O,4].")
    if valid_mask.shape != boxes_xyxy.shape[:-1]:
        raise ValueError("valid_mask must match [B,T,O].")
    x0, y0, x1, y1 = boxes_xyxy.unbind(dim=-1)
    width = (x1 - x0).clamp_min(0.0)
    height = (y1 - y0).clamp_min(0.0)
    centers = torch.stack(((x0 + x1) * 0.5, (y0 + y1) * 0.5), dim=-1)
    coverage = valid_mask.float().mean(dim=1)
    if boxes_xyxy.shape[1] < 2:
        return coverage
    valid_pairs = valid_mask[:, 1:] & valid_mask[:, :-1]
    center_delta = (centers[:, 1:] - centers[:, :-1]).square().sum(dim=-1).sqrt()
    scale = torch.sqrt((width * height).clamp_min(1e-6))
    scale_delta = torch.abs(scale[:, 1:].log() - scale[:, :-1].log())
    pair_weights = valid_pairs.float()
    denominator = pair_weights.sum(dim=1).clamp_min(1.0)
    jitter = ((center_delta + 0.25 * scale_delta) * pair_weights).sum(dim=1) / denominator
    return (coverage * torch.exp(-8.0 * jitter)).clamp(0.0, 1.0)


__all__ = ["geometry_track_confidence"]
