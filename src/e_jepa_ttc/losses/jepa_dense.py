"""Dense JEPA predictive and anti-collapse losses."""

from __future__ import annotations

import torch
from torch.nn import functional


def cosine_prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return ``1-cosine`` averaged over all non-feature axes."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes.")
    return (1.0 - functional.cosine_similarity(prediction, target, dim=-1)).mean()


def variance_covariance_loss(
    embedding: torch.Tensor,
    *,
    variance_floor: float = 1.0,
) -> torch.Tensor:
    """VICReg-style variance/covariance regularizer over the sample axis."""

    if embedding.ndim != 2 or embedding.shape[0] < 2:
        raise ValueError("embedding must have shape [N,D] with N>=2.")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive.")
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = functional.relu(variance_floor - std).mean()
    covariance = centered.T @ centered / (embedding.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    covariance_loss = off_diagonal.square().mean()
    return variance + covariance_loss


__all__ = ["cosine_prediction_loss", "variance_covariance_loss"]
