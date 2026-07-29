"""Bounded neural correction on top of physical inverse TTC."""

from __future__ import annotations

import torch
from torch import nn


class BoundedInverseTTCResidual(nn.Module):
    """Limit neural correction to a fraction of the geometric estimate."""

    def __init__(self, dim: int, *, residual_limit: float = 0.30) -> None:
        super().__init__()
        if not 0.0 <= residual_limit <= 1.0:
            raise ValueError("residual_limit must lie in [0,1].")
        self.residual_limit = residual_limit
        self.network = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        geometry_inverse_ttc: torch.Tensor,
        object_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return corrected inverse TTC and the signed bounded residual."""

        raw = torch.tanh(self.network(object_token).squeeze(-1))
        residual = raw * self.residual_limit * geometry_inverse_ttc.detach().clamp_min(0.05)
        return (geometry_inverse_ttc + residual).clamp_min(1e-4), residual


__all__ = ["BoundedInverseTTCResidual"]
