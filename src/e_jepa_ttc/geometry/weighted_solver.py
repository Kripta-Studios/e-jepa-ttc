"""Differentiable confidence-weighted inverse-TTC solver."""

from __future__ import annotations

import torch


def weighted_inverse_ttc(
    estimates: torch.Tensor,
    confidence: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine experts without silently accepting invalid or negative estimates."""

    if estimates.shape != confidence.shape or estimates.ndim < 1:
        raise ValueError("estimates and confidence must have the same non-scalar shape.")
    valid = estimates.isfinite() & confidence.isfinite() & (estimates >= 0)
    weights = torch.where(valid, confidence.clamp_min(0.0), torch.zeros_like(confidence))
    total = weights.sum(dim=-1)
    mixture = (torch.where(valid, estimates, torch.zeros_like(estimates)) * weights).sum(dim=-1)
    mixture = mixture / total.clamp_min(epsilon)
    return mixture, total.clamp(0.0, 1.0)


__all__ = ["weighted_inverse_ttc"]
