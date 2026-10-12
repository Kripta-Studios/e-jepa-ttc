"""Uncapped relative phase correction registered from internal V evidence."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc

from .phase_units import PhaseUnitAnchor


class RelativePhaseAnchor(PhaseUnitAnchor):
    def forward(self, features: Tensor, rgb_width_logits: Tensor) -> dict[str, Tensor]:
        if rgb_width_logits.shape != (len(features), 2):
            raise ValueError("Expected R_CTX width logits [B,2]")
        correction = self.output(self.mlp(self.inputs(features)))
        location = features[:, 160] + features[:, 161].abs() * correction[:, 0]
        widths = 0.03 * functional.softplus(rgb_width_logits + correction[:, 1:])
        return {
            "raw_location": location,
            "point_phase": emitted_phase(location),
            "ttc": phase_to_ttc(location),
            "q10": (location - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
            "q90": (location + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
            "correction": correction,
        }
