"""TTC and collision heads with explicit uncertainty separation."""

from __future__ import annotations

import torch
from torch import nn

from e_jepa_ttc.models.ttc_readout import SignedTTCReadout
from e_jepa_ttc.models.uncertainty_head import TTCUncertaintyHead


class CollisionHead(nn.Module):
    """Predict independent collision logits for configured time thresholds."""

    def __init__(self, dim: int, threshold_count: int) -> None:
        super().__init__()
        if dim <= 0 or threshold_count <= 0:
            raise ValueError("dim and threshold_count must be positive.")
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, threshold_count))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[B, threshold_count]``."""

        if embedding.ndim != 2:
            raise ValueError("embedding must have shape [B,D].")
        return self.projection(embedding)


__all__ = ["CollisionHead", "SignedTTCReadout", "TTCUncertaintyHead"]
