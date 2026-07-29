"""Lightweight high-resolution event mask refiner."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class HighResolutionMaskRefiner(nn.Module):
    """Refine a coarse soft mask using the current event tensor."""

    def __init__(self, in_channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, current_events: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        """Return native-event-resolution mask logits."""

        if current_events.ndim != 4 or coarse_logits.ndim != 3:
            raise ValueError("Expected current_events [B,C,H,W] and coarse_logits [B,h,w].")
        coarse = functional.interpolate(
            coarse_logits[:, None],
            size=current_events.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return coarse[:, 0] + self.refiner(torch.cat((current_events, coarse), dim=1))[:, 0]


__all__ = ["HighResolutionMaskRefiner"]
