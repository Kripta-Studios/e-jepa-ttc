"""Heteroscedastic regression loss."""

from __future__ import annotations

import torch


def gaussian_nll(error: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    """Compute Gaussian NLL with bounded caller-provided log variance."""

    if error.shape != log_variance.shape:
        raise ValueError("error and log_variance must have equal shapes.")
    return 0.5 * (torch.exp(-log_variance) * error.square() + log_variance).mean()


__all__ = ["gaussian_nll"]
