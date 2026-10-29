"""TTC regression losses with explicit transforms."""

from __future__ import annotations

import torch
from torch.nn import functional

from e_jepa_ttc.losses.garl_ttc import signed_log_ttc_loss


def huber_ttc_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Return finite Huber loss on TTC values."""

    if delta <= 0.0:
        raise ValueError("delta must be positive.")
    if predicted.shape != target.shape:
        raise ValueError("predicted and target must have equal shapes.")
    return functional.smooth_l1_loss(predicted, target, beta=delta)


__all__ = ["huber_ttc_loss", "signed_log_ttc_loss"]
