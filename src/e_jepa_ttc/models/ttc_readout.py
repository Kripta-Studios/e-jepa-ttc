"""Explicit TTC readouts shared by supervised and zero-shot adapters."""

from __future__ import annotations

import torch
from torch import nn

from e_jepa_ttc.models.height_ratio_head import LearnedHeightRatioHead


class SignedTTCReadout(nn.Module):
    """Learn a signed TTC value from a latent representation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return one finite-capable signed TTC prediction per sample."""

        if embedding.ndim != 2:
            raise ValueError("embedding must have shape [B,D].")
        return self.projection(embedding).squeeze(-1)


__all__ = ["LearnedHeightRatioHead", "SignedTTCReadout"]
