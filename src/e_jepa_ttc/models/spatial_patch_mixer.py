"""Bidirectional within-frame patch interaction."""

from __future__ import annotations

import torch
from torch import nn


class SpatialPatchMixer(nn.Module):
    """Apply self-attention independently inside every event frame."""

    def __init__(self, dim: int, *, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mix ``[B,T,P,D]`` without imposing an arbitrary patch order."""

        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,T,P,D].")
        batch, steps, patches, dim = tokens.shape
        flat = tokens.reshape(batch * steps, patches, dim)
        normalized = self.norm(flat)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        value = flat + attended
        value = value + self.feed_forward(value)
        return value.reshape(batch, steps, patches, dim)


__all__ = ["SpatialPatchMixer"]
