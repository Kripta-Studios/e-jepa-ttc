"""Narrow, fail-closed benchmark-phase losses for E-Clock X0."""

from __future__ import annotations

import torch
from torch.nn import functional

UNIFORM_PHASE_REDUCTION = "mean_smooth_l1_benchmark_phase_error"
WEIGHTED_PHASE_REDUCTION = "normalized_weighted_absolute_phase_error"


def uniform_benchmark_phase_loss(
    predicted_benchmark_phase: torch.Tensor,
    target_benchmark_phase: torch.Tensor,
) -> torch.Tensor:
    """Mean Smooth-L1 loss for the uniform X0 arms."""

    if predicted_benchmark_phase.shape != target_benchmark_phase.shape:
        raise ValueError("phase prediction and target shapes differ")
    if not bool(torch.isfinite(predicted_benchmark_phase).all()) or not bool(
        torch.isfinite(target_benchmark_phase).all()
    ):
        raise ValueError("phase prediction and target must be finite")
    return functional.smooth_l1_loss(predicted_benchmark_phase, target_benchmark_phase)


def normalized_weighted_absolute_phase_error(
    predicted_benchmark_phase: torch.Tensor,
    target_benchmark_phase: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Official-weight X0-DYN-W loss, normalized by the observed weight sum."""

    if not (
        predicted_benchmark_phase.shape == target_benchmark_phase.shape == sample_weight.shape
    ):
        raise ValueError("prediction, target and sample_weight shapes differ")
    predicted64 = predicted_benchmark_phase.to(torch.float64)
    target64 = target_benchmark_phase.to(torch.float64)
    weight64 = sample_weight.to(torch.float64)
    if not bool(torch.isfinite(predicted64).all()):
        raise ValueError("predicted benchmark phase contains non-finite values")
    if not bool(torch.isfinite(target64).all()):
        raise ValueError("target benchmark phase contains non-finite values")
    if not bool(torch.isfinite(weight64).all()):
        raise ValueError("sample_weight contains non-finite values")
    if bool((weight64 < 0.0).any()):
        raise ValueError("sample_weight cannot be negative")
    error64 = (predicted64 - target64).abs()
    if not bool(torch.isfinite(error64).all()):
        raise ValueError("absolute phase error contains non-finite values")
    numerator = (weight64 * error64).sum(dtype=torch.float64)
    denominator = weight64.sum(dtype=torch.float64)
    if not bool(torch.isfinite(numerator)):
        raise ValueError("weighted loss numerator is non-finite")
    if not bool(torch.isfinite(denominator)) or bool(denominator <= 0.0):
        raise ValueError("weighted loss denominator must be finite and positive")
    loss64 = numerator / denominator
    if not bool(torch.isfinite(loss64)):
        raise ValueError("weighted phase loss is non-finite")
    return loss64


__all__ = [
    "UNIFORM_PHASE_REDUCTION",
    "WEIGHTED_PHASE_REDUCTION",
    "normalized_weighted_absolute_phase_error",
    "uniform_benchmark_phase_loss",
]
