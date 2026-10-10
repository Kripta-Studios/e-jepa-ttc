"""Controlled RGB-PORT late fusion with separate event and RGB clocks."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc, pinball


class _MaskedClockGRU(nn.Module):
    def __init__(self, feature_count: int = 17, hidden: int = 160) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.hidden = hidden
        self.project = nn.Sequential(
            nn.Linear(feature_count + 4, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.cells = nn.ModuleList((nn.GRUCell(hidden, hidden), nn.GRUCell(hidden, hidden)))

    def forward(self, features: Tensor, timing: Tensor, valid: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_count:
            raise ValueError("stream features must be [B,L,17]")
        batch, length, _ = features.shape
        if not 1 <= length <= 8 or timing.shape != (batch, length, 4):
            raise ValueError("stream length/timing differs from RGB-PORT contract")
        if valid.shape != (batch, length) or valid.dtype != torch.bool:
            raise ValueError("stream mask differs")
        if not valid[:, -1].all() or (valid[:, :-1] & ~valid[:, 1:]).any():
            raise ValueError("stream histories require contiguous valid suffixes")
        if not torch.isfinite(features[valid]).all() or not torch.isfinite(timing[valid]).all():
            raise ValueError("valid stream rows must be finite")
        safe_features = torch.where(valid[..., None], features, torch.zeros_like(features))
        safe_timing = torch.where(valid[..., None], timing, torch.zeros_like(timing))
        projected = self.project(torch.cat((safe_features, safe_timing), -1))
        states = [projected.new_zeros((batch, self.hidden)) for _ in range(2)]
        for index in range(length):
            first = self.cells[0](projected[:, index], states[0])
            states[0] = torch.where(valid[:, index, None], first, states[0])
            second = self.cells[1](states[0], states[1])
            states[1] = torch.where(valid[:, index, None], second, states[1])
        return states[-1]


class DualClockFusion(nn.Module):
    """Fuse two causal histories; F_ZERO removes all RGB predictive coordinates."""

    def __init__(self, mode: str = "F_TRUE") -> None:
        super().__init__()
        if mode not in {"F_TRUE", "F_ZERO"}:
            raise ValueError("fusion mode must be F_TRUE or F_ZERO")
        self.mode = mode
        self.event = _MaskedClockGRU()
        self.rgb = _MaskedClockGRU()
        self.combine = nn.Sequential(nn.Linear(320, 160), nn.LayerNorm(160), nn.SiLU())
        self.location = nn.Linear(160, 1)
        self.width = nn.Linear(160, 2)
        for head in (self.location, self.width):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        event_features: Tensor,
        event_timing: Tensor,
        event_valid: Tensor,
        event_expert_phase: Tensor,
        rgb_features: Tensor,
        rgb_timing: Tensor,
        rgb_valid: Tensor,
    ) -> dict[str, Tensor]:
        if (
            event_expert_phase.shape != (len(event_features), 3)
            or not torch.isfinite(event_expert_phase).all()
        ):
            raise ValueError("fusion anchor requires three current event expert phases")
        event_hidden = self.event(event_features, event_timing, event_valid)
        rgb_input = torch.zeros_like(rgb_features) if self.mode == "F_ZERO" else rgb_features
        rgb_hidden = self.rgb(rgb_input, rgb_timing, rgb_valid)
        hidden = self.combine(torch.cat((event_hidden, rgb_hidden), -1))
        anchor = event_expert_phase.median(-1).values
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


def fusion_loss(
    output: Mapping[str, Tensor], truth: Tensor, mass: Tensor, population: int
) -> Tensor:
    """Global-mass phase/quantile loss; both fusion controls have lambda_cost=0."""

    if truth.ndim != 1 or mass.shape != truth.shape or population <= 0:
        raise ValueError("invalid fusion supervision")
    if not torch.isfinite(truth).all() or not torch.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("fusion truth/mass must be finite and nonnegative")
    point = (output["point_phase"] - truth).abs() / 0.03
    quantile = (pinball(output["q10"], truth, 0.1) + pinball(output["q90"], truth, 0.9)) / 0.03
    return (population * mass * (point + 0.1 * quantile)).mean()


class MissingRGBFallback(nn.Module):
    """Return the frozen E_CTX output exactly when RGB is unavailable."""

    def __init__(self, event_context: nn.Module, fusion: DualClockFusion) -> None:
        super().__init__()
        self.event_context = event_context
        self.fusion = fusion

    def forward(
        self,
        event_features: Tensor,
        event_timing: Tensor,
        event_valid: Tensor,
        event_expert_phase: Tensor,
        *,
        rgb_available: Tensor,
        rgb_features: Tensor | None = None,
        rgb_timing: Tensor | None = None,
        rgb_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        event_output = self.event_context(
            event_features, event_timing, event_valid, event_expert_phase
        )
        if rgb_available.shape != (len(event_features),) or rgb_available.dtype != torch.bool:
            raise ValueError("RGB availability must be bool[B]")
        if not rgb_available.any():
            return event_output
        if rgb_features is None or rgb_timing is None or rgb_valid is None:
            raise ValueError("available RGB requires its frozen feature history")
        selected = rgb_available.nonzero(as_tuple=False).squeeze(1)
        fused = self.fusion(
            event_features[selected],
            event_timing[selected],
            event_valid[selected],
            event_expert_phase[selected],
            rgb_features[selected],
            rgb_timing[selected],
            rgb_valid[selected],
        )
        result = dict(event_output)
        prediction_keys = {"raw_location", "point_phase", "ttc", "q10", "q90"}
        common = set(event_output).intersection(fused).intersection(prediction_keys)
        for key in common:
            if (
                event_output[key].shape[:1] == rgb_available.shape
                and fused[key].shape[:1] == selected.shape
            ):
                merged = event_output[key].clone()
                merged[selected] = fused[key]
                result[key] = merged
        return result


__all__ = ["DualClockFusion", "MissingRGBFallback", "fusion_loss"]
