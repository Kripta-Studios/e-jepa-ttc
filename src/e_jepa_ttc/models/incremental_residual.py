"""Frozen A5 dynamic residual adapter for the preregistered E-Clock X1 screen."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from e_jepa_ttc.models.collision_clock_math import phase_lower_bound


@dataclass(frozen=True)
class IncrementalResidualConfig:
    """Architecture and physical-domain constants frozen before X0.5 results."""

    slot_count: int = 9
    hidden_dim: int = 32
    dropout: float = 0.05
    metric_delta_t_s: float = 0.1
    minimum_abs_prediction_ttc_s: float = 0.1

    def __post_init__(self) -> None:
        if self.slot_count != 9 or self.hidden_dim != 32:
            raise ValueError("X1 topology is frozen to 9 -> 32 -> 32 -> 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.metric_delta_t_s, self.minimum_abs_prediction_ttc_s)
        ):
            raise ValueError("phase-domain constants must be finite and positive")


def add_safe_phase_residual(
    a5_phase: torch.Tensor,
    raw_delta: torch.Tensor,
    *,
    metric_delta_t_s: float,
    minimum_abs_prediction_ttc_s: float,
) -> torch.Tensor:
    """Add a residual while preserving the E-Clock lower phase boundary.

    Negative corrections are bounded by the row-specific distance from A5 to
    the valid-domain boundary. Positive corrections are unbounded. A raw delta
    of exactly zero is added as exactly zero, so zero initialization replays A5
    bit-for-bit in the scientific phase coordinate.
    """

    if a5_phase.shape != raw_delta.shape:
        raise ValueError("A5 phase and residual must have identical shapes")
    if not bool(torch.isfinite(a5_phase).all()) or not bool(torch.isfinite(raw_delta).all()):
        raise ValueError("phase residual inputs must be finite")
    lower = phase_lower_bound(
        metric_delta_t_s=metric_delta_t_s,
        minimum_abs_prediction_ttc_s=minimum_abs_prediction_ttc_s,
    )
    margin = a5_phase - lower
    if bool((margin < 0.0).any()):
        raise ValueError("A5 phase is outside the frozen valid domain")
    bounded = torch.where(raw_delta < 0.0, margin * torch.tanh(raw_delta), raw_delta)
    result = a5_phase + bounded
    if not bool(torch.isfinite(result).all()) or bool((result < lower).any()):
        raise RuntimeError("safe phase residual violated the valid domain")
    return result


class FrozenA5DynamicResidualAdapter(nn.Module):
    """Minimal residual MLP over nine frozen dynamic slots and frozen A5 phase."""

    slot_mean: torch.Tensor
    slot_std: torch.Tensor

    def __init__(
        self,
        slot_mean: torch.Tensor,
        slot_std: torch.Tensor,
        config: IncrementalResidualConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or IncrementalResidualConfig()
        mean = torch.as_tensor(slot_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(slot_std, dtype=torch.float32).reshape(-1)
        if mean.shape != (9,) or std.shape != (9,):
            raise ValueError("slot normalization must contain exactly nine values")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("slot normalization must be finite")
        if bool((std <= 0.0).any()):
            raise ValueError("slot standard deviations must be positive")
        self.register_buffer("slot_mean", mean.clone())
        self.register_buffer("slot_std", std.clone())
        self.network = nn.Sequential(
            nn.Linear(9, 32),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("residual adapter must end with Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, a5_phase: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        if a5_phase.ndim != 1 or slots.shape != (a5_phase.shape[0], 9):
            raise ValueError("adapter expects phase [B] and slots [B,9]")
        if not bool(torch.isfinite(slots).all()):
            raise ValueError("dynamic slots must be finite")
        normalized = (slots.to(torch.float32) - self.slot_mean) / self.slot_std
        raw_delta = self.network(normalized).squeeze(-1)
        return add_safe_phase_residual(
            a5_phase.to(torch.float32),
            raw_delta,
            metric_delta_t_s=self.config.metric_delta_t_s,
            minimum_abs_prediction_ttc_s=self.config.minimum_abs_prediction_ttc_s,
        )


__all__ = [
    "FrozenA5DynamicResidualAdapter",
    "IncrementalResidualConfig",
    "add_safe_phase_residual",
]
