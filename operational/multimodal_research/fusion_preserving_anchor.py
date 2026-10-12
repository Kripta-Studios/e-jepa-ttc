"""Preserve frozen F_TRUE point/quantiles and test incremental refined RGB evidence."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc

from .phase_units import PhaseUnitAnchor


class FusionPreservingAnchor(PhaseUnitAnchor):
    frozen_width_weight: Tensor
    frozen_width_bias: Tensor

    def __init__(self, mode: str) -> None:
        if mode not in {"REFINED_RGB", "ZERO_RGB"}:
            raise ValueError("Expected REFINED_RGB or ZERO_RGB")
        super().__init__("EVENT")
        self.refiner_mode = mode
        self.mlp[0] = nn.Linear(502, 160)
        self.register_buffer("frozen_width_weight", torch.zeros(2, 160))
        self.register_buffer("frozen_width_bias", torch.zeros(2))

    def inputs(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 502:
            raise ValueError("Expected [B,502] E/R/F frozen state features")
        base = super().inputs(features[:, :340]).clone()
        if self.refiner_mode == "ZERO_RGB":
            base[:, :165] = 0
        fused = features[:, 340:].clone()
        fused[:, 160:] /= 0.03
        return torch.cat([base, fused], -1)

    def forward(self, features: Tensor, rgb_width_logits: Tensor) -> dict[str, Tensor]:
        if rgb_width_logits.shape != (len(features), 2):
            raise ValueError("Input ABI requires [B,2]; RGB widths are not used by this model")
        correction = self.output(self.mlp(self.inputs(features)))
        location = features[:, 500] + 0.03 * correction[:, 0]
        frozen_width = functional.linear(
            features[:, 340:500], self.frozen_width_weight, self.frozen_width_bias
        )
        widths = 0.03 * functional.softplus(frozen_width + correction[:, 1:])
        return {
            "raw_location": location,
            "point_phase": emitted_phase(location),
            "ttc": phase_to_ttc(location),
            "q10": (location - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
            "q90": (location + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
            "correction": correction,
        }

    def initialize_width(self, state: dict) -> None:
        with torch.no_grad():
            self.frozen_width_weight.copy_(state["width.weight"])
            self.frozen_width_bias.copy_(state["width.bias"])
