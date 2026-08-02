"""Reusable neural blocks exposed by the generic model API."""

from __future__ import annotations

import torch
from torch import nn

from e_jepa_ttc.models.tiny_cnn import ResidualBlock
from e_jepa_ttc.models.token_transformer import RotaryTransformerEncoderLayer


class MLPBlock(nn.Module):
    """Layer-normalized two-layer MLP for latent predictors and heads."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        hidden = hidden_dim or dim * 4
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the residual-compatible MLP to the final tensor dimension."""

        return self.net(values)


__all__ = ["MLPBlock", "ResidualBlock", "RotaryTransformerEncoderLayer"]
