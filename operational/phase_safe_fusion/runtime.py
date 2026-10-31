"""Ensemble aggregation preserving direction and exact native fallback."""

import torch
from torch import nn

from operational.phase_safe_fusion.model import SafeFusion


def combine_seeds(
    predictions: torch.Tensor, native: torch.Tensor, quality: torch.Tensor
) -> torch.Tensor:
    """Median of exactly three seeds; directly return F0 without fresh history."""
    if predictions.shape[0] != 3:
        raise ValueError("Expected exactly three seed predictions")
    result = predictions.median(dim=0).values
    history = torch.isfinite(quality).all(1) & (quality[:, 9] > 0) & (quality[:, 11] <= 0.15)
    return torch.where(history, result, native[:, 0])


class SafeEnsemble(nn.Module):
    """Three fixed seeds combined without signed arithmetic cancellation."""

    def __init__(self, models: list[SafeFusion]) -> None:
        super().__init__()
        if len(models) != 3:
            raise ValueError("Expected three trained seeds")
        self.models = nn.ModuleList(models)

    def forward(
        self, native: torch.Tensor, geometry: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        """Return signed median TTC with exact missing-history fallback."""
        outputs = torch.stack([m(native, geometry, quality) for m in self.models])
        return combine_seeds(outputs, native, quality)
