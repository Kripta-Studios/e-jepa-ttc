"""Stable benchmark-phase mathematics for the E-Clock X0 experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CollisionClockMetricConfig:
    """Frozen metric constants and physical prediction guardrails."""

    metric_delta_t_s: float = 0.1
    ttc_clip_seconds: float = 60.0
    minimum_abs_prediction_ttc_s: float = 0.1
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        values = (
            self.metric_delta_t_s,
            self.ttc_clip_seconds,
            self.minimum_abs_prediction_ttc_s,
            self.eps,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("collision-clock metric constants must be finite and positive")


def ttc_to_benchmark_phase(
    ttc_seconds: torch.Tensor,
    *,
    metric_delta_t_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return start-anchor benchmark phase and its exact real-domain mask."""

    if not math.isfinite(metric_delta_t_s) or metric_delta_t_s <= 0.0:
        raise ValueError("metric_delta_t_s must be finite and positive")
    finite = torch.isfinite(ttc_seconds)
    nonzero = ttc_seconds != 0.0
    positive_domain = (ttc_seconds < 0.0) | (ttc_seconds > metric_delta_t_s)
    valid = finite & nonzero & positive_domain
    safe = torch.where(valid, ttc_seconds, torch.ones_like(ttc_seconds))
    phase = -torch.log1p(-metric_delta_t_s / safe)
    return torch.where(valid, phase, torch.full_like(phase, float("nan"))), valid


def benchmark_phase_to_inverse_ttc(
    benchmark_phase: torch.Tensor,
    *,
    metric_delta_t_s: float,
) -> torch.Tensor:
    """Convert benchmark phase to signed inverse TTC using a stable expm1."""

    if not math.isfinite(metric_delta_t_s) or metric_delta_t_s <= 0.0:
        raise ValueError("metric_delta_t_s must be finite and positive")
    if not bool(torch.isfinite(benchmark_phase).all()):
        raise ValueError("benchmark phase must be finite")
    return -torch.expm1(-benchmark_phase) / metric_delta_t_s


def finite_ttc_from_inverse(
    inverse_ttc: torch.Tensor,
    *,
    clip_seconds: float,
) -> torch.Tensor:
    """Map inverse TTC to a finite signed compatibility value."""

    if not math.isfinite(clip_seconds) or clip_seconds <= 0.0:
        raise ValueError("clip_seconds must be finite and positive")
    if not bool(torch.isfinite(inverse_ttc).all()):
        raise ValueError("inverse TTC must be finite")
    minimum = 1.0 / clip_seconds
    sign = torch.where(
        inverse_ttc < 0.0,
        -torch.ones_like(inverse_ttc),
        torch.ones_like(inverse_ttc),
    )
    safe = sign * inverse_ttc.abs().clamp_min(minimum)
    return torch.reciprocal(safe).clamp(-clip_seconds, clip_seconds)


def benchmark_phase_to_ttc(
    benchmark_phase: torch.Tensor,
    *,
    metric_delta_t_s: float,
    clip_seconds: float,
) -> torch.Tensor:
    """Convert phase to finite TTC; phase remains the primary coordinate."""

    inverse = benchmark_phase_to_inverse_ttc(
        benchmark_phase,
        metric_delta_t_s=metric_delta_t_s,
    )
    return finite_ttc_from_inverse(inverse, clip_seconds=clip_seconds)


def phase_lower_bound(*, metric_delta_t_s: float, minimum_abs_prediction_ttc_s: float) -> float:
    """Return the phase bound that excludes the official failure interval."""

    if min(metric_delta_t_s, minimum_abs_prediction_ttc_s) <= 0.0:
        raise ValueError("phase-bound constants must be positive")
    return -math.log1p(metric_delta_t_s / minimum_abs_prediction_ttc_s)


def neutral_raw_phase(*, metric_delta_t_s: float, minimum_abs_prediction_ttc_s: float) -> float:
    """Return the raw head bias that maps exactly to neutral phase zero."""

    lower = phase_lower_bound(
        metric_delta_t_s=metric_delta_t_s,
        minimum_abs_prediction_ttc_s=minimum_abs_prediction_ttc_s,
    )
    return math.log(math.expm1(-lower))


__all__ = [
    "CollisionClockMetricConfig",
    "benchmark_phase_to_inverse_ttc",
    "benchmark_phase_to_ttc",
    "finite_ttc_from_inverse",
    "neutral_raw_phase",
    "phase_lower_bound",
    "ttc_to_benchmark_phase",
]
