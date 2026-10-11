"""Exact three-value median without CUDA median-with-indices."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc


def deterministic_forward(
    self: DualClockFusion,
    event_features: Tensor,
    event_timing: Tensor,
    event_valid: Tensor,
    event_expert_phase: Tensor,
    rgb_features: Tensor,
    rgb_timing: Tensor,
    rgb_valid: Tensor,
) -> dict[str, Tensor]:
    """Preserve fusion math; select the middle of three sorted finite phases.

    Expert inputs are frozen cache values: gradients into tied experts are not
    part of training. Parameter gradients and anchor values remain unchanged.
    """
    if (
        event_expert_phase.shape != (len(event_features), 3)
        or not torch.isfinite(event_expert_phase).all()
    ):
        raise ValueError("fusion anchor requires three current event expert phases")
    event_hidden = self.event(event_features, event_timing, event_valid)
    rgb_input = torch.zeros_like(rgb_features) if self.mode == "F_ZERO" else rgb_features
    rgb_hidden = self.rgb(rgb_input, rgb_timing, rgb_valid)
    hidden = self.combine(torch.cat((event_hidden, rgb_hidden), -1))
    anchor = event_expert_phase.sort(dim=-1, stable=True).values[:, 1]
    location = anchor + 0.03 * self.location(hidden).squeeze(-1)
    widths = 0.03 * functional.softplus(self.width(hidden))
    point = emitted_phase(location)
    return {
        "raw_location": location,
        "point_phase": point,
        "ttc": phase_to_ttc(location),
        "q10": (location - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
        "q90": (location + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
        "event_hidden": event_hidden,
        "rgb_hidden": rgb_hidden,
        "hidden": hidden,
    }
