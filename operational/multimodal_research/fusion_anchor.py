"""RGB anchor with a controlled addition of the frozen fusion representation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .phase_units import PhaseUnitAnchor


class FusionStateAnchor(PhaseUnitAnchor):
    def __init__(self, mode: str) -> None:
        if mode not in {"FUSED", "ZERO_FUSED"}:
            raise ValueError("Expected FUSED or ZERO_FUSED")
        super().__init__("EVENT")
        self.fusion_mode = mode
        self.mlp[0] = nn.Linear(502, 160)

    def inputs(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 502:
            raise ValueError("Expected [B,502] refiner and fusion states")
        base = super().inputs(features[:, :340])
        fused = features[:, 340:].clone()
        if self.fusion_mode == "ZERO_FUSED":
            fused.zero_()
        fused[:, 160:] /= 0.03
        return torch.cat([base, fused], -1)
