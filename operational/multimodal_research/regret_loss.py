"""H-only excess phase risk relative to the frozen RGB prediction."""

from __future__ import annotations

import torch
from torch import Tensor


def rgb_regret(
    point: Tensor, target: Tensor, anchor: Tensor, mass: Tensor, population: int
) -> Tensor:
    if point.ndim != 1 or not (point.shape == target.shape == anchor.shape == mass.shape):
        raise ValueError("Expected matching one-dimensional point, target, anchor and mass")
    excess = (point - target).abs() - (anchor - target).abs()
    return (population * mass * torch.relu(excess) / 0.03).mean()
