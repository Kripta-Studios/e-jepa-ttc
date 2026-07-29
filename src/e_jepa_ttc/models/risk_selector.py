"""Per-object collision-risk prediction and selection."""

from __future__ import annotations

import torch
from torch import nn


class RiskSelector(nn.Module):
    """Predict risk thresholds and select the most urgent valid object."""

    def __init__(self, dim: int, thresholds_s: tuple[float, ...]) -> None:
        super().__init__()
        if not thresholds_s:
            raise ValueError("thresholds_s cannot be empty.")
        self.thresholds_s = thresholds_s
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, len(thresholds_s)))

    def forward(
        self,
        object_tokens: torch.Tensor,
        inverse_ttc: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return risk logits and selected object index."""

        logits = self.head(object_tokens)
        masked_inverse = torch.where(
            object_mask,
            inverse_ttc,
            torch.full_like(inverse_ttc, -torch.inf),
        )
        return logits, masked_inverse.argmax(dim=-1)


__all__ = ["RiskSelector"]
