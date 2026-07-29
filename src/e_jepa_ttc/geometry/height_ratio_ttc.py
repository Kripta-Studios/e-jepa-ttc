"""Closed-form apparent-height TTC estimates."""

from __future__ import annotations

import torch


def _causal_log_slope(
    values: torch.Tensor,
    times_s: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a least-squares log slope and an R²-like confidence."""

    if values.ndim < 2:
        raise ValueError("values must include a temporal dimension.")
    if times_s.ndim == 1:
        times_s = times_s.view(*([1] * (values.ndim - 2)), -1, 1)
    elif times_s.ndim == 2 and values.ndim >= 3:
        times_s = times_s.view(times_s.shape[0], times_s.shape[1], *([1] * (values.ndim - 2)))
    if times_s.shape[-2] != values.shape[-2]:
        raise ValueError("times_s and values must have the same temporal length.")
    valid = values.isfinite() & (values > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.bool()
    weights = valid.to(values.dtype)
    safe_values = torch.where(valid, values, torch.ones_like(values))
    log_values = safe_values.clamp_min(1e-6).log()
    count = weights.sum(dim=-2).clamp_min(1.0)
    mean_t = (times_s * weights).sum(dim=-2) / count
    mean_y = (log_values * weights).sum(dim=-2) / count
    centered_t = times_s - mean_t.unsqueeze(-2)
    centered_y = log_values - mean_y.unsqueeze(-2)
    covariance = (weights * centered_t * centered_y).sum(dim=-2)
    time_variance = (weights * centered_t.square()).sum(dim=-2).clamp_min(1e-8)
    slope = covariance / time_variance
    fitted = mean_y.unsqueeze(-2) + slope.unsqueeze(-2) * centered_t
    residual = (weights * (log_values - fitted).square()).sum(dim=-2)
    total = (weights * centered_y.square()).sum(dim=-2).clamp_min(1e-8)
    confidence = (1.0 - residual / total).clamp(0.0, 1.0)
    confidence = confidence * (count >= 2).to(confidence.dtype)
    return slope, confidence


def height_ratio_inverse_ttc(
    heights: torch.Tensor,
    times_s: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate inverse TTC with the exact causal height-ratio identity.

    For a rigid object under constant translational approach, inverse TTC at
    the current endpoint is
    ``q_current = (h_current / h_previous - 1) / delta_t``.  The earliest
    and latest valid observations are used so annotation jitter is not
    amplified by a single short adjacent interval.  This remains an exact
    current-endpoint identity under the constant-velocity looming model.
    The commonly quoted ``(1 - h_previous/h_current)/delta_t`` refers to the
    previous endpoint. Disagreement among adjacent estimates only reduces
    confidence.
    """

    return _causal_pair_ratio_rate(
        heights,
        times_s,
        valid_mask=valid_mask,
        ratio_power=1.0,
    )


def _causal_pair_ratio_rate(
    values: torch.Tensor,
    times_s: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None,
    ratio_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an exact apparent-scale ratio over every valid causal pair."""

    if values.ndim < 2 or values.shape[-2] < 2:
        raise ValueError("values must contain at least two temporal observations.")
    if ratio_power <= 0:
        raise ValueError("ratio_power must be positive.")
    if times_s.ndim == 1:
        if times_s.shape[0] != values.shape[-2]:
            raise ValueError("times_s and values must have the same temporal length.")
        dt = times_s[1:] - times_s[:-1]
        dt = dt.view(*([1] * (values.ndim - 2)), -1, 1)
    elif times_s.ndim == 2:
        if times_s.shape[1] != values.shape[-2]:
            raise ValueError("times_s and values must have the same temporal length.")
        dt = times_s[:, 1:] - times_s[:, :-1]
        dt = dt.view(times_s.shape[0], times_s.shape[1] - 1, *([1] * (values.ndim - 2)))
    else:
        raise ValueError("times_s must have shape [T] or [B,T].")
    previous = values[..., :-1, :]
    current = values[..., 1:, :]
    valid = (
        previous.isfinite()
        & current.isfinite()
        & (previous > 0)
        & (current > 0)
        & (dt > 1e-6)
    )
    if valid_mask is not None:
        if valid_mask.shape != values.shape:
            raise ValueError("valid_mask must match values.")
        valid = valid & valid_mask[..., :-1, :].bool() & valid_mask[..., 1:, :].bool()
    scale_growth = (current.clamp_min(1e-6) / previous.clamp_min(1e-6)).pow(
        ratio_power
    )
    pair_rate = (scale_growth - 1.0) / dt.clamp_min(1e-6)
    pair_rate = torch.where(valid, pair_rate, torch.zeros_like(pair_rate))
    weights = valid.to(values.dtype)
    count = weights.sum(dim=-2)
    mean_rate = (pair_rate * weights).sum(dim=-2) / count.clamp_min(1.0)
    disagreement = (
        (pair_rate - mean_rate.unsqueeze(-2)).abs() * weights
    ).sum(dim=-2) / count.clamp_min(1.0)
    # Adjacent identities are useful for consistency, but a single interval
    # amplifies box jitter. Use the widest valid causal span and evaluate the
    # exact ratio at its current endpoint.
    pair_axis = pair_rate.shape[-2]
    pair_indices = torch.arange(pair_axis, device=values.device)
    pair_indices = pair_indices.view(
        *([1] * (valid.ndim - 2)),
        pair_axis,
        1,
    )
    latest_index = torch.where(
        valid,
        pair_indices,
        torch.full_like(pair_indices, -1),
    ).amax(dim=-2)
    earliest_index = torch.where(
        valid,
        pair_indices,
        torch.full_like(pair_indices, pair_axis),
    ).amin(dim=-2)
    first_value = values.gather(
        -2,
        earliest_index.clamp(0, values.shape[-2] - 1).unsqueeze(-2),
    ).squeeze(-2)
    last_value = values.gather(
        -2,
        (latest_index + 1).clamp(0, values.shape[-2] - 1).unsqueeze(-2),
    ).squeeze(-2)
    if times_s.ndim == 1:
        first_time = times_s[earliest_index.clamp(0, times_s.shape[0] - 1)]
        last_time = times_s[(latest_index + 1).clamp(0, times_s.shape[0] - 1)]
    else:
        first_time = times_s.gather(
            1,
            earliest_index.clamp(0, times_s.shape[1] - 1),
        )
        last_time = times_s.gather(
            1,
            (latest_index + 1).clamp(0, times_s.shape[1] - 1),
        )
    span_dt = (last_time - first_time).clamp_min(1e-6)
    span_rate = (
        (last_value.clamp_min(1e-6) / first_value.clamp_min(1e-6)).pow(ratio_power)
        - 1.0
    ) / span_dt
    has_span = (latest_index >= 0) & (earliest_index < pair_axis)
    span_rate = torch.where(has_span, span_rate, torch.zeros_like(span_rate))
    relative_disagreement = disagreement / span_rate.abs().clamp_min(0.05)
    support = (count / 2.0).clamp(0.0, 1.0)
    approaching = span_rate > 0
    confidence = torch.exp(-relative_disagreement) * support
    confidence = confidence * approaching.to(confidence.dtype)
    return span_rate.clamp_min(0.0), confidence


__all__ = ["height_ratio_inverse_ttc"]
