"""Learned Height Ratio head used by the local Garl-TTC replica."""

from __future__ import annotations

import torch
from torch import nn


def raw_garl_height_ratio_ttc(
    first_height: torch.Tensor,
    last_height: torch.Tensor,
    elapsed_s: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the unclamped Garl formula for singularity diagnostics."""

    return elapsed_s / (1.0 - first_height / last_height)


class LearnedHeightRatioHead(nn.Module):
    """Regress two raw visible heights and apply the source Garl formula."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.height_regressor = nn.Linear(dim, 2)

    def forward(
        self,
        fused_pair_token: torch.Tensor,
        elapsed_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return inverse TTC, ``h1/h2`` and the two regressed heights."""

        heights = self.height_regressor(fused_pair_token)
        first_height, last_height = heights.unbind(dim=-1)
        safe_last_height = torch.where(
            last_height.abs() >= 1e-6,
            last_height,
            torch.where(
                last_height < 0,
                torch.full_like(last_height, -1e-6),
                torch.full_like(last_height, 1e-6),
            ),
        )
        height_ratio = first_height / safe_last_height
        while elapsed_s.ndim < height_ratio.ndim:
            elapsed_s = elapsed_s.unsqueeze(-1)
        inverse_ttc = (1.0 - height_ratio) / elapsed_s.clamp_min(1e-6)
        inverse_ttc = torch.where(
            inverse_ttc.abs() >= 1e-6,
            inverse_ttc,
            torch.where(
                inverse_ttc < 0,
                torch.full_like(inverse_ttc, -1e-6),
                torch.full_like(inverse_ttc, 1e-6),
            ),
        )
        return inverse_ttc, height_ratio, heights


__all__ = ["LearnedHeightRatioHead", "raw_garl_height_ratio_ttc"]
