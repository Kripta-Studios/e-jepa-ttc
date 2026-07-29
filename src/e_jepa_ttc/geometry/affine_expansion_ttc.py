"""Differentiable weighted affine-expansion TTC expert."""

from __future__ import annotations

import torch


def affine_expansion_inverse_ttc(
    boxes_xyxy: torch.Tensor,
    times_s: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    regularization: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve translation, radial expansion and in-plane rotation from box corners.

    For every adjacent pair the four current box corners are treated as
    correspondences.  The backward displacement is modelled as::

        dx = tx + kappa * x - omega * y
        dy = ty + kappa * y + omega * x

    Coordinates are relative to the current center.  This makes
    ``kappa`` equal to ``1 - scale_prev / scale_now``. Inverse TTC at the
    current endpoint is therefore ``kappa / (delta_t * (1 - kappa))`` rather
    than the previous-endpoint quantity ``kappa / delta_t``.
    The normal equations are solved in FP32 and remain differentiable.  Each
    adjacent estimate refers to the *current* endpoint of that pair, so the
    most recent valid pair is returned.  Older pairs only reduce confidence
    when their estimates disagree; averaging them would estimate TTC at a
    mixture of timestamps.
    """

    if boxes_xyxy.ndim != 4 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must have shape [B,T,O,4].")
    if boxes_xyxy.shape[1] < 2:
        raise ValueError("At least two causal boxes are required.")
    if regularization <= 0:
        raise ValueError("regularization must be positive.")
    batch, steps, objects = boxes_xyxy.shape[:3]
    if times_s.ndim == 1:
        if times_s.shape[0] != steps:
            raise ValueError("times_s and boxes must have the same temporal length.")
        dt = (times_s[1:] - times_s[:-1])[None, :, None].expand(batch, -1, objects)
    elif times_s.ndim == 2 and times_s.shape == (batch, steps):
        dt = (times_s[:, 1:] - times_s[:, :-1])[:, :, None].expand(-1, -1, objects)
    else:
        raise ValueError("times_s must have shape [T] or [B,T].")
    previous = _box_corners(boxes_xyxy[:, :-1])
    current = _box_corners(boxes_xyxy[:, 1:])
    center = current.mean(dim=-2, keepdim=True)
    relative = current - center
    displacement = current - previous
    x = relative[..., 0]
    y = relative[..., 1]
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    row_x = torch.stack((ones, zeros, x, -y), dim=-1)
    row_y = torch.stack((zeros, ones, y, x), dim=-1)
    design = torch.stack((row_x, row_y), dim=-2).flatten(-3, -2)
    target = displacement.flatten(-2, -1).unsqueeze(-1)
    solve_design = design.float()
    solve_target = target.float()
    normal = solve_design.transpose(-1, -2) @ solve_design
    identity = torch.eye(4, device=normal.device, dtype=normal.dtype)
    normal = normal + regularization * identity
    rhs = solve_design.transpose(-1, -2) @ solve_target
    parameters = torch.linalg.solve(normal, rhs).squeeze(-1)
    fitted = (solve_design @ parameters.unsqueeze(-1)).squeeze(-1)
    residual = (fitted - solve_target.squeeze(-1)).square().mean(dim=-1)
    signal = solve_target.squeeze(-1).square().mean(dim=-1).clamp_min(1e-8)
    condition = torch.linalg.cond(normal).clamp_min(1.0)
    pair_confidence = torch.exp(-residual / signal) / (1.0 + condition.log())
    kappa = parameters[..., 2]
    pair_rate = kappa / (
        dt.float().clamp_min(1e-6) * (1.0 - kappa).clamp_min(1e-4)
    )
    widths = boxes_xyxy[..., 2] - boxes_xyxy[..., 0]
    heights = boxes_xyxy[..., 3] - boxes_xyxy[..., 1]
    valid = (
        (widths[:, :-1] > 0)
        & (widths[:, 1:] > 0)
        & (heights[:, :-1] > 0)
        & (heights[:, 1:] > 0)
        & (dt > 1e-6)
        & pair_rate.isfinite()
    )
    if valid_mask is not None:
        if valid_mask.shape != boxes_xyxy.shape[:-1]:
            raise ValueError("valid_mask must match boxes [B,T,O].")
        valid = valid & valid_mask[:, :-1].bool() & valid_mask[:, 1:].bool()
    valid_float = valid.float()
    count = valid_float.sum(dim=1)
    mean_rate = (pair_rate * valid_float).sum(dim=1) / count.clamp_min(1.0)
    disagreement = (
        (pair_rate - mean_rate[:, None]).abs() * valid_float
    ).sum(dim=1) / count.clamp_min(1.0)
    pair_axis = pair_rate.shape[1]
    pair_indices = torch.arange(pair_axis, device=boxes_xyxy.device).view(1, -1, 1)
    latest_index = torch.where(
        valid,
        pair_indices,
        torch.full_like(pair_indices, -1),
    ).amax(dim=1)
    gather_index = latest_index.clamp_min(0)[:, None]
    rate = pair_rate.gather(1, gather_index).squeeze(1)
    latest_confidence = pair_confidence.gather(1, gather_index).squeeze(1)
    has_valid = latest_index >= 0
    rate = torch.where(has_valid, rate, torch.zeros_like(rate))
    latest_confidence = torch.where(
        has_valid,
        latest_confidence,
        torch.zeros_like(latest_confidence),
    )
    relative_disagreement = disagreement / rate.abs().clamp_min(0.05)
    support = (count / 2.0).clamp(0.0, 1.0)
    confidence = latest_confidence * torch.exp(-relative_disagreement) * support
    approaching = rate > 0
    return (
        rate.clamp_min(0.0).to(boxes_xyxy.dtype),
        (confidence * approaching.float()).clamp(0.0, 1.0).to(boxes_xyxy.dtype),
    )


def _box_corners(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes_xyxy.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((x0, y0), dim=-1),
            torch.stack((x1, y0), dim=-1),
            torch.stack((x1, y1), dim=-1),
            torch.stack((x0, y1), dim=-1),
        ),
        dim=-2,
    )


__all__ = ["affine_expansion_inverse_ttc"]
