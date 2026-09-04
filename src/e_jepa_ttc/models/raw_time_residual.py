"""Frozen RAW16-MTR residual and matched Stage 64 interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.incremental_residual import add_safe_phase_residual

RawArm = Literal["S64-STATE-L1", "S64-COUNT-L1", "S64-RAW-L1", "S64-PERM-L1"]
RAW_ARMS: tuple[RawArm, ...] = (
    "S64-STATE-L1",
    "S64-COUNT-L1",
    "S64-RAW-L1",
    "S64-PERM-L1",
)


@dataclass(frozen=True)
class RawModelBatch:
    """Label-free model inputs; identifiers and supervision cannot be stored here."""

    normalized_rates: torch.Tensor
    a5_phase: torch.Tensor
    normalized_a5_state: torch.Tensor
    times: torch.Tensor
    valid_patches: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.a5_phase.shape[0]
        if self.normalized_rates.shape != (batch, 2, 16, 2, 64, 64):
            raise ValueError("normalized_rates must be [B,2,16,2,64,64]")
        if self.normalized_a5_state.shape != (batch, 3) or self.times.shape != (batch, 6):
            raise ValueError("A5 state/times shape mismatch")
        if self.valid_patches.shape != (batch, 16) or self.valid_patches.dtype != torch.bool:
            raise ValueError("valid_patches must be boolean [B,16]")
        for value in (
            self.normalized_rates,
            self.a5_phase,
            self.normalized_a5_state,
            self.times,
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError("RawModelBatch contains non-finite values")


@dataclass(frozen=True)
class RawTimeResidualOutput:
    """Prediction plus the two distinct residual quantities required for audit."""

    benchmark_phase: torch.Tensor
    helper_input: torch.Tensor
    applied_phase_delta: torch.Tensor
    patch_weights: torch.Tensor
    fallback: torch.Tensor


class CausalTemporalBlock(nn.Module):
    """Depthwise causal temporal convolution followed by a pointwise residual."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.left = 2 * dilation
        self.depthwise = nn.Conv1d(channels, channels, 3, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply a strictly left-padded residual block."""

        return inputs + functional.silu(
            self.pointwise(self.depthwise(functional.pad(inputs, (self.left, 0))))
        )


class RawTimeResidual(nn.Module):
    """30k-parameter spatial-temporal residual over a frozen A5 phase."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.temporal = nn.Sequential(*(CausalTemporalBlock(32, d) for d in (1, 2, 4)))
        self.readout = nn.Sequential(nn.Linear(203, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.proposal = nn.Linear(32, 1)
        self.reliability = nn.Linear(32, 1)
        nn.init.zeros_(self.proposal.weight)
        nn.init.zeros_(self.proposal.bias)
        axis = (torch.arange(4, dtype=torch.float32) + 0.5) / 2.0 - 1.0
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.coords: torch.Tensor
        self.register_buffer("coords", torch.stack((x.flatten(), y.flatten()), dim=-1))

    def forward(self, batch: RawModelBatch) -> RawTimeResidualOutput:
        """Predict phase while returning exact A5 for all-empty support masks."""

        rates = batch.normalized_rates
        size = rates.shape[0]
        features = self.stem(rates.reshape(size * 2 * 16, 2, 64, 64))
        features = features.reshape(size, 2, 16, 32, 16).permute(0, 1, 4, 3, 2)
        scales: list[torch.Tensor] = []
        for window in range(2):
            sequence = features[:, window].reshape(size * 16, 32, 16)
            for length in (4, 8, 16):
                scales.append(self.temporal(sequence[..., -length:])[..., -1].reshape(size, 16, 32))
        patch = torch.cat(
            (
                *scales,
                self.coords[None].expand(size, -1, -1),
                batch.times[:, None].expand(-1, 16, -1),
                batch.normalized_a5_state[:, None].expand(-1, 16, -1),
            ),
            dim=-1,
        )
        hidden = self.readout(patch)
        logits = self.reliability(hidden).squeeze(-1)
        any_valid = batch.valid_patches.any(dim=1)
        safe_mask = batch.valid_patches | (~any_valid[:, None])
        weights = torch.softmax(
            logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min), dim=1
        )
        helper_input = 0.05 * torch.tanh((weights * self.proposal(hidden).squeeze(-1)).sum(dim=1))
        predicted = add_safe_phase_residual(
            batch.a5_phase,
            helper_input,
            metric_delta_t_s=0.1,
            minimum_abs_prediction_ttc_s=0.1,
        )
        phase = torch.where(any_valid, predicted, batch.a5_phase)
        helper_input = torch.where(any_valid, helper_input, torch.zeros_like(helper_input))
        return RawTimeResidualOutput(
            benchmark_phase=phase,
            helper_input=helper_input,
            applied_phase_delta=phase - batch.a5_phase,
            patch_weights=weights,
            fallback=~any_valid,
        )


def count_only_counts(counts: torch.Tensor, durations_s: torch.Tensor) -> torch.Tensor:
    """Remove time order while preserving mass per window/pixel/polarity."""

    if counts.ndim != 6 or counts.shape[1:4] != (2, 16, 2):
        raise ValueError("counts must be [B,2,16,2,H,W]")
    if durations_s.shape != counts.shape[:3] or bool((durations_s <= 0).any()):
        raise ValueError("durations_s must be positive [B,2,16]")
    fractions = durations_s / durations_s.sum(dim=2, keepdim=True)
    return counts.sum(dim=2, keepdim=True) * fractions[..., None, None, None]


def permute_counts(counts: torch.Tensor, permutations: torch.Tensor) -> torch.Tensor:
    """Apply a per-token/window temporal-bin bijection without changing count mass."""

    if counts.ndim != 6 or permutations.shape != counts.shape[:3]:
        raise ValueError("count/permutation shapes disagree")
    expected = torch.arange(16, device=permutations.device).expand_as(permutations)
    if bool((torch.sort(permutations, dim=2).values != expected).any()) or bool(
        (permutations == expected).any()
    ):
        raise ValueError("PERM requires a fixed-point-free bijection")
    index = permutations[..., None, None, None].expand(-1, -1, -1, *counts.shape[3:])
    return torch.gather(counts, 2, index)


def normalize_arm_rates(
    counts: torch.Tensor,
    durations_s: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    arm: RawArm,
    permutations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize one matched intervention and apply shared RAW-train normalization."""

    if arm not in RAW_ARMS:
        raise ValueError(f"unknown Stage 64 arm: {arm}")
    values = counts.to(torch.float32)
    if arm == "S64-COUNT-L1":
        values = count_only_counts(values, durations_s)
    elif arm == "S64-PERM-L1":
        if permutations is None:
            raise ValueError("PERM arm requires deterministic permutations")
        values = permute_counts(values, permutations)
    rates = torch.log1p(values / durations_s[..., None, None, None])
    if mean.shape != (2,) or std.shape != (2,) or bool((std <= 0).any()):
        raise ValueError("shared rate normalization must contain two polarities")
    normalized = (rates - mean[None, None, None, :, None, None]) / std[
        None, None, None, :, None, None
    ]
    if arm == "S64-STATE-L1":
        normalized = torch.zeros_like(normalized)
    return normalized


__all__ = [
    "RAW_ARMS",
    "CausalTemporalBlock",
    "RawArm",
    "RawModelBatch",
    "RawTimeResidual",
    "RawTimeResidualOutput",
    "count_only_counts",
    "normalize_arm_rates",
    "permute_counts",
]
