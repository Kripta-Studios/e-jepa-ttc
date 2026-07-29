"""Embedding and router health diagnostics with fail-closed collapse detection."""

from __future__ import annotations

import torch


def embedding_health(embeddings: torch.Tensor, *, std_threshold: float = 1e-3) -> dict[str, float]:
    """Compute collapse indicators from a batch of embeddings."""

    if embeddings.ndim < 2:
        raise ValueError("embeddings must include batch and feature dimensions.")
    flat = embeddings.reshape(-1, embeddings.shape[-1]).float()
    std = flat.std(dim=0, unbiased=False)
    centered = flat - flat.mean(dim=0)
    singular = torch.linalg.svdvals(centered)
    probability = singular / singular.sum().clamp_min(1e-8)
    effective_rank = torch.exp(-(probability * probability.clamp_min(1e-8).log()).sum())
    return {
        "mean_dimension_std": float(std.mean()),
        "collapsed_dimension_fraction": float((std < std_threshold).float().mean()),
        "effective_rank": float(effective_rank),
        "mean_embedding_norm": float(flat.norm(dim=-1).mean()),
    }


def assert_embeddings_not_collapsed(
    health: dict[str, float],
    *,
    maximum_collapsed_fraction: float = 0.80,
) -> None:
    """Abort when the configured collapse condition is met."""

    if health["collapsed_dimension_fraction"] > maximum_collapsed_fraction:
        raise RuntimeError(
            "Embedding collapse detected: "
            f"{health['collapsed_dimension_fraction']:.1%} dimensions below threshold."
        )


__all__ = ["assert_embeddings_not_collapsed", "embedding_health"]
