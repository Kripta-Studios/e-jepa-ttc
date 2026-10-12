"""Small correction from frozen expert/refiner phases and clocks, excluding hidden states."""

import torch
from torch import Tensor, nn

from .phase_units import PhaseUnitAnchor


class PhaseOnlyAnchor(PhaseUnitAnchor):
    def __init__(self, mode: str) -> None:
        super().__init__(mode)
        self.mlp = nn.Sequential(
            nn.Linear(20, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.output = nn.Linear(32, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def inputs(self, features: Tensor) -> Tensor:
        values = super().inputs(features)
        return torch.cat([values[:, 160:170], values[:, 330:340]], -1)
