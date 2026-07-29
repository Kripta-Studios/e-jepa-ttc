"""Heteroscedastic TTC uncertainty."""

from __future__ import annotations

import torch
from torch import nn


class TTCUncertaintyHead(nn.Module):
    """Predict bounded log variance in inverse-TTC space."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, object_token: torch.Tensor) -> torch.Tensor:
        return self.head(object_token).squeeze(-1).clamp(-8.0, 5.0)


__all__ = ["TTCUncertaintyHead"]
