"""Causal object-motion prediction head."""

from __future__ import annotations

import torch
from torch import nn


class ObjectMotionHead(nn.Module):
    """Predict center velocity and log-scale expansion from object tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

    def forward(self, object_tokens: torch.Tensor) -> torch.Tensor:
        """Return ``dx/dt, dy/dt, dlogw/dt, dlogh/dt``."""

        return self.head(object_tokens)


__all__ = ["ObjectMotionHead"]
