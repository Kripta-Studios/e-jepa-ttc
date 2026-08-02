"""Finite signed TTC losses with no hidden label transforms."""

from __future__ import annotations

import torch
from torch.nn import functional


def signed_log_ttc_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    """Huber loss in signed-log space, preserving negative TTC values."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes.")
    if beta <= 0:
        raise ValueError("beta must be positive.")
    return functional.smooth_l1_loss(
        prediction.sign() * torch.log1p(prediction.abs()),
        target.sign() * torch.log1p(target.abs()),
        beta=beta,
    )


__all__ = ["signed_log_ttc_loss"]
