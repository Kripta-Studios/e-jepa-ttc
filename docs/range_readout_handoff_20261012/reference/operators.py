"""Testable operators for V15. No dataset loader or trainer is implemented here."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def importance_estimate(losses: Tensor, masses: Tensor, probabilities: Tensor) -> Tensor:
    """Estimate sum_i w_i*l_i from samples drawn with replacement from p_i.

    Arguments are the values at sampled IDs, not the full population. In particular,
    their masses/probabilities need NOT sum to one in the sampled batch. Repeated IDs
    remain repeated. Do not combine this estimator with another N*w multiplier.
    """
    if losses.ndim != 1 or losses.numel() == 0:
        raise ValueError("A nonempty vector of per-query losses is required")
    if masses.shape != losses.shape or probabilities.shape != losses.shape:
        raise ValueError("Losses, global masses and sampling probabilities must align")
    if masses.device != losses.device or probabilities.device != losses.device:
        raise ValueError("All inputs must use the same device")
    if not all(x.is_floating_point() for x in (losses, masses, probabilities)):
        raise TypeError("Floating point tensors required")
    if masses.requires_grad or probabilities.requires_grad:
        raise ValueError("Sampling masses are fixed TRAIN metadata, not learnable")
    if not all(bool(torch.isfinite(x).all()) for x in (losses, masses, probabilities)):
        raise ValueError("Nonfinite estimator input")
    if bool((masses < 0).any()) or bool((probabilities <= 0).any()):
        raise ValueError("Nonnegative masses and strictly positive probabilities required")
    if bool((masses > 1).any()) or bool((probabilities > 1).any()):
        raise ValueError("Global normalized masses/probabilities cannot exceed one")
    return ((masses / probabilities).to(losses.dtype) * losses).mean()


@dataclass(frozen=True)
class ResidualSpec:
    """Serializable identity; the native C*tanh(s) loader is not interchangeable."""
    bound: float
    slope: float = 0.05
    operator: str = "equal_slope_tanh_v1"

    def __post_init__(self) -> None:
        if self.bound not in (0.05, 0.20):
            raise ValueError("Only the two registered bounds are admitted")
        if self.slope != 0.05 or self.operator != "equal_slope_tanh_v1":
            raise ValueError("Residual operator identity changed")


def equal_slope_residual(score: Tensor, spec: ResidualSpec) -> Tensor:
    """Bounded odd correction with derivative 0.05 at the origin for both arms.

    score is (ordered_forward_score - ordered_reverse_score)/2. Finite arithmetic
    may reach the bound numerically. No straight-through derivative is used.
    """
    if not score.is_floating_point():
        raise TypeError("A floating point scorer tensor is required")
    if not bool(torch.isfinite(score).all()):
        raise ValueError("Nonfinite scorer value")
    return spec.bound * torch.tanh((spec.slope / spec.bound) * score)


def ordered_residual(forward: Tensor, reverse: Tensor, spec: ResidualSpec) -> Tensor:
    if forward.shape != reverse.shape:
        raise ValueError("Forward and reverse scores must align")
    return equal_slope_residual(0.5 * (forward - reverse), spec)


def require_same_operator(saved: ResidualSpec, requested: ResidualSpec) -> None:
    """Fail closed when loading a patch under another residual definition."""
    if saved != requested:
        raise ValueError("Checkpoint residual operator does not match runtime")


def current_log_ratio(ttc_seconds: float, delta_seconds: float) -> float:
    """Approaching current-anchor geometry; NOT benchmark phase -log(1-dt/T)."""
    if not all(math.isfinite(x) and x > 0 for x in (ttc_seconds, delta_seconds)):
        raise ValueError("Positive finite times required by this illustration")
    return math.log1p(delta_seconds / ttc_seconds)
