"""Native H8 with a controlled current-observation geometric bypass."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase


class RepresentationRefiner(TemporalRefiner):
    """The control has exactly the native H8 function and shared initialization."""

    def __init__(self, active: bool) -> None:
        super().__init__(TemporalConfig(feature_count=17, hidden=160, backbone="gru"))
        self.active = active
        self.bypass = nn.Sequential(
            nn.Linear(896, 160), nn.LayerNorm(160), nn.SiLU(), nn.Linear(160, 160)
        )
        last = self.bypass[-1]
        assert isinstance(last, nn.Linear) and last.bias is not None
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self, features: Tensor, times: Tensor, valid: Tensor, experts: Tensor, side: Tensor
    ) -> dict[str, Tensor]:
        if features.ndim != 3 or features.shape[2] != 17:
            raise ValueError("Expected PHASE17")
        b, length, _ = features.shape
        if not 1 <= length <= 16 or side.shape != (b, 896):
            raise ValueError("Input shape outside frozen contract")
        if times.shape != (b, length, 4) or valid.shape != (b, length) or valid.dtype != torch.bool:
            raise ValueError("Timing/mask mismatch")
        if (
            experts.shape != (b, 3)
            or not valid[:, -1].all()
            or (valid[:, :-1] & ~valid[:, 1:]).any()
        ):
            raise ValueError("Invalid native suffix/expert contract")
        if not all(torch.isfinite(v).all() for v in (features[valid], times[valid], experts, side)):
            raise ValueError("Nonfinite inputs")
        safe_x = torch.where(valid[..., None], features, torch.zeros_like(features))
        safe_t = torch.where(valid[..., None], times, torch.zeros_like(times))
        z = self.project(torch.cat((safe_x, safe_t), -1))
        addition = self.bypass(side) * float(self.active)
        z = torch.cat((z[:, :-1], (z[:, -1] + addition)[:, None]), dim=1)
        states = [z.new_zeros((b, 160)), z.new_zeros((b, 160))]
        for step in range(length):
            proposal = self.cells[0](z[:, step], states[0])
            states[0] = torch.where(valid[:, step, None], proposal, states[0])
            proposal = self.cells[1](states[0], states[1])
            states[1] = torch.where(valid[:, step, None], proposal, states[1])
        hidden = states[-1]
        anchor = (
            experts.sort(dim=-1, stable=True).values[:, 1]
            if experts.is_cuda
            else experts.median(-1).values
        )
        delta = 0.03 * self.location(hidden).squeeze(-1)
        location = anchor + delta
        widths = 0.03 * functional.softplus(self.width(hidden))
        cost = torch.cat((experts.new_zeros((b, 1)), self.cost(hidden)), -1)
        return {
            "point_phase": emitted_phase(location),
            "raw_location": location,
            "raw_residual": delta,
            "q10": (location - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
            "q90": (location + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
            "relative_cost": cost,
            "expert_index": cost.argmin(-1),
        }
