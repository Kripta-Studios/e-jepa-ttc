"""Object Event TTC v4.19: dense local-correlation flow/divergence probe.

This module deliberately contains no trainable prediction head. It asks a
representation question: do the frozen v4.8 spatial encoder maps contain local
correspondences from which translation-invariant expansion can be recovered?
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True)
class ObjectEventV419Config:
    search_radius: int = 4
    correlation_temperature: float = 0.07
    foreground_floor: float = 0.05
    confidence_floor: float = 0.05
    minimum_score_scale: float = 1.0e-4
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.search_radius <= 0:
            raise ValueError("search_radius must be positive")
        if min(
            self.correlation_temperature,
            self.foreground_floor,
            self.confidence_floor,
            self.minimum_score_scale,
            self.epsilon,
        ) <= 0.0:
            raise ValueError("v4.19 numerical controls must be positive")


def _offsets(radius: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(values, values, indexing="ij")
    return dx.reshape(-1), dy.reshape(-1)


def local_correlation_flow(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    radius: int,
    temperature: float,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft local feature matching, returning flow-x, flow-y and confidence.

    Features are L2-normalised channel-wise before correlation. Padding positions
    are masked rather than treated as zero-similarity candidates. Flow is in
    feature-grid pixels, from ``first`` to ``second``.
    """
    if first.ndim != 4 or second.shape != first.shape:
        raise ValueError("first and second must align as [B,C,H,W]")
    if radius <= 0 or temperature <= 0.0:
        raise ValueError("radius and temperature must be positive")
    batch, channels, height, width = first.shape
    kernel = 2 * radius + 1
    candidates = kernel * kernel

    first_n = functional.normalize(first.float(), dim=1, eps=epsilon)
    second_n = functional.normalize(second.float(), dim=1, eps=epsilon)
    patches = functional.unfold(second_n, kernel_size=kernel, padding=radius)
    patches = patches.reshape(batch, channels, candidates, height, width)
    correlation = (first_n[:, :, None] * patches).sum(dim=1)

    validity = functional.unfold(
        torch.ones((1, 1, height, width), device=first.device, dtype=first_n.dtype),
        kernel_size=kernel,
        padding=radius,
    ).reshape(1, candidates, height, width) > 0.5
    correlation = correlation.masked_fill(~validity, -1.0e4)
    probability = torch.softmax(correlation / float(temperature), dim=1)

    dx, dy = _offsets(radius, device=first.device, dtype=first_n.dtype)
    flow_x = (probability * dx[None, :, None, None]).sum(dim=1)
    flow_y = (probability * dy[None, :, None, None]).sum(dim=1)

    entropy = -(probability.clamp_min(epsilon) * probability.clamp_min(epsilon).log()).sum(dim=1)
    valid_count = validity.sum(dim=1).to(first_n.dtype).clamp_min(2.0)
    confidence = (1.0 - entropy / valid_count.log()).clamp(0.0, 1.0)
    return flow_x, flow_y, confidence


def _finite_derivatives(flow_x: torch.Tensor, flow_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if flow_x.ndim != 3 or flow_y.shape != flow_x.shape:
        raise ValueError("flow fields must align as [B,H,W]")
    du_dx = torch.zeros_like(flow_x)
    dv_dy = torch.zeros_like(flow_y)
    du_dx[:, :, 1:-1] = 0.5 * (flow_x[:, :, 2:] - flow_x[:, :, :-2])
    du_dx[:, :, 0] = flow_x[:, :, 1] - flow_x[:, :, 0]
    du_dx[:, :, -1] = flow_x[:, :, -1] - flow_x[:, :, -2]
    dv_dy[:, 1:-1, :] = 0.5 * (flow_y[:, 2:, :] - flow_y[:, :-2, :])
    dv_dy[:, 0, :] = flow_y[:, 1, :] - flow_y[:, 0, :]
    dv_dy[:, -1, :] = flow_y[:, -1, :] - flow_y[:, -2, :]
    return du_dx, dv_dy


def dense_flow_scores(
    flow_x: torch.Tensor,
    flow_y: torch.Tensor,
    foreground_pair: torch.Tensor,
    confidence: torch.Tensor,
    *,
    foreground_floor: float,
    confidence_floor: float,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return mean divergence, radial scale slope and translation magnitude."""
    if foreground_pair.shape != flow_x.shape or confidence.shape != flow_x.shape:
        raise ValueError("foreground/confidence must align with flow")
    weight = (
        (foreground_floor + foreground_pair.float().clamp_min(0.0))
        * (confidence_floor + confidence.float().clamp(0.0, 1.0))
    )
    mass = weight.sum(dim=(-2, -1)).clamp_min(epsilon)

    du_dx, dv_dy = _finite_derivatives(flow_x, flow_y)
    divergence = ((du_dx + dv_dy) * weight).sum(dim=(-2, -1)) / mass

    batch, height, width = flow_x.shape
    y = torch.arange(height, device=flow_x.device, dtype=flow_x.dtype)
    x = torch.arange(width, device=flow_x.device, dtype=flow_x.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cx = (weight * xx).sum(dim=(-2, -1)) / mass
    cy = (weight * yy).sum(dim=(-2, -1)) / mass
    dx = xx[None] - cx[:, None, None]
    dy = yy[None] - cy[:, None, None]
    radial_numerator = (weight * (flow_x * dx + flow_y * dy)).sum(dim=(-2, -1))
    radial_denominator = (weight * (dx.square() + dy.square())).sum(dim=(-2, -1)).clamp_min(epsilon)
    radial_slope = radial_numerator / radial_denominator

    mean_u = (weight * flow_x).sum(dim=(-2, -1)) / mass
    mean_v = (weight * flow_y).sum(dim=(-2, -1)) / mass
    translation = torch.sqrt(mean_u.square() + mean_v.square() + epsilon)
    return divergence, radial_slope, translation


def antisymmetric_correspondence_scores(
    first_features: torch.Tensor,
    second_features: torch.Tensor,
    first_foreground: torch.Tensor,
    second_foreground: torch.Tensor,
    config: ObjectEventV419Config,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute endpoint-swap antisymmetric divergence/radial scores."""
    if first_foreground.ndim != 3 or second_foreground.shape != first_foreground.shape:
        raise ValueError("foreground endpoints must align as [B,H,W]")
    target_size = first_features.shape[-2:]
    fg1 = functional.interpolate(first_foreground[:, None].float(), size=target_size, mode="bilinear", align_corners=False)[:, 0]
    fg2 = functional.interpolate(second_foreground[:, None].float(), size=target_size, mode="bilinear", align_corners=False)[:, 0]
    foreground_pair = torch.sqrt((fg1 * fg2).clamp_min(0.0))

    fx, fy, conf_f = local_correlation_flow(
        first_features,
        second_features,
        radius=config.search_radius,
        temperature=config.correlation_temperature,
        epsilon=config.epsilon,
    )
    rx, ry, conf_r = local_correlation_flow(
        second_features,
        first_features,
        radius=config.search_radius,
        temperature=config.correlation_temperature,
        epsilon=config.epsilon,
    )
    div_f, radial_f, translation_f = dense_flow_scores(
        fx, fy, foreground_pair, conf_f,
        foreground_floor=config.foreground_floor,
        confidence_floor=config.confidence_floor,
        epsilon=config.epsilon,
    )
    div_r, radial_r, translation_r = dense_flow_scores(
        rx, ry, foreground_pair, conf_r,
        foreground_floor=config.foreground_floor,
        confidence_floor=config.confidence_floor,
        epsilon=config.epsilon,
    )
    divergence = 0.5 * (div_f - div_r)
    radial = 0.5 * (radial_f - radial_r)
    diagnostics = {
        "mean_confidence": 0.5 * (conf_f.mean(dim=(-2, -1)) + conf_r.mean(dim=(-2, -1))),
        "translation_magnitude": 0.5 * (translation_f + translation_r),
        "forward_divergence": div_f,
        "reverse_divergence": div_r,
        "forward_radial_slope": radial_f,
        "reverse_radial_slope": radial_r,
    }
    return divergence, radial, diagnostics


__all__ = [
    "ObjectEventV419Config",
    "antisymmetric_correspondence_scores",
    "dense_flow_scores",
    "local_correlation_flow",
]
