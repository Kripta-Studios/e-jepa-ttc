"""Differentiable local cross-time transport for event-native A5 TTC models.

The module deliberately keeps correspondence local and interpretable.  It does
not predict TTC directly.  Instead it builds a cosine-similarity cost volume
between dense endpoint features, extracts an expected displacement field, and
summarises translation, anisotropic/isotropic expansion, confidence, entropy,
and forward/reverse cycle consistency.

All operations are inference-time event-only when called from ``CausalScaleTTC``.
No RGB, DINO feature, bbox, track ID, or TTC label is consumed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional


TRANSPORT_FEATURE_NAMES: tuple[str, ...] = (
    "translation_x",
    "translation_y",
    "divergence_x",
    "divergence_y",
    "divergence_isotropic",
    "flow_magnitude",
    "confidence_margin",
    "entropy",
    "cycle_error",
    "foreground_translation_x",
    "foreground_translation_y",
    "foreground_divergence_x",
    "foreground_divergence_y",
    "foreground_divergence_isotropic",
    "foreground_flow_magnitude",
    "foreground_confidence_margin",
    "foreground_entropy",
    "foreground_cycle_error",
)


@dataclass(frozen=True)
class LocalTransportMatch:
    """Dense expected displacement and confidence from a local cost volume."""

    dx: torch.Tensor  # [B,H,W], in feature-grid cells
    dy: torch.Tensor  # [B,H,W], in feature-grid cells
    confidence_margin: torch.Tensor  # [B,H,W], top1-top2 cosine similarity
    entropy: torch.Tensor  # [B,H,W], normalised to [0,1]
    valid: torch.Tensor  # [B,H,W]
    probability: torch.Tensor | None = None  # optional [B,K,H,W]


def _candidate_displacements(
    radius: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    return dx.reshape(-1), dy.reshape(-1)


def local_correlation_match(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    radius: int,
    temperature: float,
    return_probability: bool = False,
) -> LocalTransportMatch:
    """Match ``previous[p]`` against ``current[p + delta]`` within ``radius``.

    Features are L2-normalised and correlated in float32.  Invalid candidates
    outside the feature map are masked before softmax, so there is no wraparound.
    The expected displacement is differentiable with respect to both endpoints.
    """

    if previous.ndim != 4 or current.ndim != 4 or previous.shape != current.shape:
        raise ValueError("transport endpoints must share shape [B,C,H,W]")
    if radius < 1:
        raise ValueError("transport radius must be >= 1")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("transport temperature must be finite and positive")
    batch, _channels, height, width = previous.shape
    if min(height, width) <= 2 * radius:
        raise ValueError("transport radius is too large for the dense feature grid")

    previous_f = functional.normalize(previous.float(), dim=1, eps=1.0e-6)
    current_f = functional.normalize(current.float(), dim=1, eps=1.0e-6)

    # Memory contract: never materialise [B,C,K,H,W].  At the production
    # 32x32 grid, r=4 and batch=32x2 temporal pairs, an unfold-based
    # implementation would transiently exceed a gigabyte even for the 64-channel
    # A4 encoder and scale linearly with capacity.  A single padded endpoint plus
    # shifted *views* keeps the differentiable state to O(BCHW + BKHW), avoiding
    # both the giant unfold and 81 autograd-retained BCHW copies.
    padded_current = functional.pad(current_f, (radius, radius, radius, radius))
    y_coords = torch.arange(height, device=current_f.device)
    x_coords = torch.arange(width, device=current_f.device)
    correlations: list[torch.Tensor] = []
    valid_maps: list[torch.Tensor] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y0 = radius + dy
            x0 = radius + dx
            shifted = padded_current[:, :, y0 : y0 + height, x0 : x0 + width]
            correlations.append((previous_f * shifted).sum(dim=1))
            valid_y = (y_coords + dy >= 0) & (y_coords + dy < height)
            valid_x = (x_coords + dx >= 0) & (x_coords + dx < width)
            valid_2d = valid_y[:, None] & valid_x[None, :]
            valid_maps.append(valid_2d[None].expand(batch, -1, -1))

    correlation = torch.stack(correlations, dim=1)
    candidate_valid = torch.stack(valid_maps, dim=1)
    negative_large = torch.finfo(correlation.dtype).min
    masked_correlation = correlation.masked_fill(~candidate_valid, negative_large)
    probability = torch.softmax(masked_correlation / float(temperature), dim=1)
    probability = probability * candidate_valid.to(probability.dtype)
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

    dx_values, dy_values = _candidate_displacements(
        radius,
        device=probability.device,
        dtype=probability.dtype,
    )
    dx = (probability * dx_values[None, :, None, None]).sum(dim=1)
    dy = (probability * dy_values[None, :, None, None]).sum(dim=1)

    top2 = masked_correlation.topk(k=2, dim=1).values
    confidence_margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
    valid_count = candidate_valid.sum(dim=1).clamp_min(2).to(probability.dtype)
    entropy = -(probability.clamp_min(1.0e-12).log() * probability).sum(dim=1)
    entropy = (entropy / valid_count.log()).clamp(0.0, 1.0)
    valid = candidate_valid.any(dim=1)

    return LocalTransportMatch(
        dx=dx,
        dy=dy,
        confidence_margin=confidence_margin,
        entropy=entropy,
        valid=valid,
        probability=probability if return_probability else None,
    )


def _sample_reverse_flow(
    reverse: LocalTransportMatch,
    forward: LocalTransportMatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the reverse field at ``p + d_forward(p)`` for cycle diagnostics."""

    batch, height, width = forward.dx.shape
    dtype = forward.dx.dtype
    device = forward.dx.device
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    scale_x = 2.0 / float(max(width - 1, 1))
    scale_y = 2.0 / float(max(height - 1, 1))
    sample_grid = torch.stack(
        (
            grid_x[None].expand(batch, -1, -1) + scale_x * forward.dx,
            grid_y[None].expand(batch, -1, -1) + scale_y * forward.dy,
        ),
        dim=-1,
    )
    reverse_field = torch.stack((reverse.dx, reverse.dy), dim=1)
    sampled = functional.grid_sample(
        reverse_field,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    reverse_valid = functional.grid_sample(
        reverse.valid[:, None].to(dtype),
        sample_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0] > 0.5
    return sampled[:, 0], sampled[:, 1], reverse_valid


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    numerator = (values * weights).sum(dim=(-2, -1))
    denominator = weights.sum(dim=(-2, -1)).clamp_min(1.0e-6)
    return numerator / denominator


def _weighted_slope(
    coordinate: torch.Tensor,
    displacement: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    coordinate_mean = _weighted_mean(coordinate, weights)
    displacement_mean = _weighted_mean(displacement, weights)
    centered_coordinate = coordinate - coordinate_mean[:, None, None]
    centered_displacement = displacement - displacement_mean[:, None, None]
    covariance = _weighted_mean(centered_coordinate * centered_displacement, weights)
    variance = _weighted_mean(centered_coordinate.square(), weights).clamp_min(1.0e-6)
    return covariance / variance


def _summarise_one(
    forward: LocalTransportMatch,
    reverse: LocalTransportMatch,
    *,
    spatial_weight: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    batch, height, width = forward.dx.shape
    dtype = forward.dx.dtype
    device = forward.dx.device
    if spatial_weight.shape != (batch, height, width):
        raise ValueError("transport spatial weight must have shape [B,H,W]")

    base_weight = spatial_weight.float().clamp_min(0.0) * forward.valid.float()
    # Confidence is used only as a gentle reliability weight; a floor prevents
    # early random features from collapsing the physical summaries to zero.
    reliability = (forward.confidence_margin / 0.10).clamp(0.05, 1.0)
    weights = base_weight * reliability

    x = torch.linspace(-0.5, 0.5, width, device=device, dtype=dtype)
    y = torch.linspace(-0.5, 0.5, height, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx[None].expand(batch, -1, -1)
    yy = yy[None].expand(batch, -1, -1)

    dx_normalized = forward.dx / float(max(width - 1, 1))
    dy_normalized = forward.dy / float(max(height - 1, 1))
    translation_x = _weighted_mean(dx_normalized, weights)
    translation_y = _weighted_mean(dy_normalized, weights)
    divergence_x = _weighted_slope(xx, dx_normalized, weights)
    divergence_y = _weighted_slope(yy, dy_normalized, weights)
    divergence_isotropic = 0.5 * (divergence_x + divergence_y)
    magnitude = _weighted_mean(
        torch.sqrt(dx_normalized.square() + dy_normalized.square() + 1.0e-12),
        weights,
    )
    confidence = _weighted_mean(forward.confidence_margin, base_weight)
    entropy = _weighted_mean(forward.entropy, base_weight)

    reverse_dx, reverse_dy, reverse_valid = _sample_reverse_flow(reverse, forward)
    cycle_valid = forward.valid & reverse_valid
    cycle_weight = spatial_weight.float().clamp_min(0.0) * cycle_valid.float()
    cycle_magnitude = torch.sqrt(
        (forward.dx + reverse_dx).square() + (forward.dy + reverse_dy).square() + 1.0e-12
    ) / float(max(radius, 1))
    cycle_error = _weighted_mean(cycle_magnitude, cycle_weight)

    return torch.stack(
        (
            translation_x,
            translation_y,
            divergence_x,
            divergence_y,
            divergence_isotropic,
            magnitude,
            confidence,
            entropy,
            cycle_error,
        ),
        dim=-1,
    )


def transport_physical_features(
    forward: LocalTransportMatch,
    reverse: LocalTransportMatch,
    *,
    foreground_weight: torch.Tensor | None,
    radius: int,
) -> torch.Tensor:
    """Return 18 interpretable per-pair transport scalars.

    The first nine use the whole feature grid; the final nine use the model's
    own foreground probability as a soft spatial weight.  Bboxes are never used.
    """

    if forward.dx.shape != reverse.dx.shape:
        raise ValueError("forward/reverse transport fields must share shape")
    batch, height, width = forward.dx.shape
    global_weight = torch.ones(
        (batch, height, width),
        device=forward.dx.device,
        dtype=forward.dx.dtype,
    )
    if foreground_weight is None:
        foreground_weight = global_weight
    elif foreground_weight.ndim == 4 and foreground_weight.shape[1] == 1:
        foreground_weight = foreground_weight[:, 0]
    if foreground_weight.shape != (batch, height, width):
        raise ValueError("foreground_weight must have shape [B,H,W] or [B,1,H,W]")

    global_features = _summarise_one(
        forward,
        reverse,
        spatial_weight=global_weight,
        radius=radius,
    )
    foreground_features = _summarise_one(
        forward,
        reverse,
        spatial_weight=foreground_weight,
        radius=radius,
    )
    features = torch.cat((global_features, foreground_features), dim=-1)
    if features.shape[-1] != len(TRANSPORT_FEATURE_NAMES):
        raise RuntimeError("transport feature schema drifted")
    return features


__all__ = [
    "TRANSPORT_FEATURE_NAMES",
    "LocalTransportMatch",
    "local_correlation_match",
    "transport_physical_features",
]
