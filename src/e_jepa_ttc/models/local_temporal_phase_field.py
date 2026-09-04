"""Stage 62 local temporal phase field and matched interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.data.stage61_pair_feature_cache import (
    LOCAL_FEATURE_DIM,
    LocalTemporalFieldBatch,
)
from e_jepa_ttc.models.incremental_residual import add_safe_phase_residual
from e_jepa_ttc.models.local_transport import LocalTransportMatch, local_correlation_match

LocalFieldMode = Literal["local", "global"]


@dataclass(frozen=True)
class LocalTemporalPhaseFieldConfig:
    """Frozen Stage 62 topology and physical-domain constants."""

    patch_grid: int = 4
    input_dim: int = LOCAL_FEATURE_DIM
    hidden_dim: int = 48
    patch_dim: int = 32
    residual_bound: float = 0.05
    metric_delta_t_s: float = 0.1
    minimum_abs_prediction_ttc_s: float = 0.1

    def __post_init__(self) -> None:
        if (self.patch_grid, self.input_dim, self.hidden_dim, self.patch_dim) != (4, 34, 48, 32):
            raise ValueError("X2 topology is frozen to 4x4 patches and 34→48→32")
        if self.residual_bound != 0.05:
            raise ValueError("X2 residual bound is frozen at 0.05")


@dataclass(frozen=True)
class LocalTemporalPhaseFieldOutput:
    """Phase prediction plus auditable per-patch proposal/reliability tensors."""

    benchmark_phase: torch.Tensor
    bounded_residual: torch.Tensor
    patch_proposals: torch.Tensor
    patch_weights: torch.Tensor


class LocalTemporalPhaseField(nn.Module):
    """Reliability-weighted local residual over a frozen A5 phase."""

    def __init__(self, config: LocalTemporalPhaseFieldConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalTemporalPhaseFieldConfig()
        self.patch_encoder = nn.Sequential(
            nn.LayerNorm(34),
            nn.Linear(34, 48),
            nn.SiLU(),
            nn.Linear(48, 32),
            nn.SiLU(),
        )
        self.proposal_head = nn.Linear(32, 1)
        self.reliability_head = nn.Linear(32, 1)
        nn.init.zeros_(self.proposal_head.weight)
        nn.init.zeros_(self.proposal_head.bias)

    def forward(self, batch: LocalTemporalFieldBatch) -> LocalTemporalPhaseFieldOutput:
        encoded = self.patch_encoder(batch.patch_features)
        proposal = self.proposal_head(encoded).squeeze(-1)
        logits = self.reliability_head(encoded).squeeze(-1)
        valid = batch.patch_valid.to(dtype=torch.bool)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("each X2 row requires at least one valid patch")
        weights = torch.softmax(logits.masked_fill(~valid, torch.finfo(logits.dtype).min), dim=1)
        row_residual = torch.sum(weights * proposal, dim=1)
        bounded = self.config.residual_bound * torch.tanh(row_residual)
        phase = add_safe_phase_residual(
            batch.a5_phase,
            bounded,
            metric_delta_t_s=self.config.metric_delta_t_s,
            minimum_abs_prediction_ttc_s=self.config.minimum_abs_prediction_ttc_s,
        )
        return LocalTemporalPhaseFieldOutput(phase, bounded, proposal, weights)


def _sample_reverse(
    reverse: LocalTransportMatch, forward: LocalTransportMatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, height, width = forward.dx.shape
    dtype, device = forward.dx.dtype, forward.dx.device
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack(
        (
            gx[None].expand(batch, -1, -1) + (2.0 / max(width - 1, 1)) * forward.dx,
            gy[None].expand(batch, -1, -1) + (2.0 / max(height - 1, 1)) * forward.dy,
        ),
        dim=-1,
    )
    field = torch.stack((reverse.dx, reverse.dy), dim=1)
    sampled = functional.grid_sample(field, grid, align_corners=True, padding_mode="zeros")
    valid = (
        functional.grid_sample(
            reverse.valid[:, None].to(dtype),
            grid,
            mode="nearest",
            align_corners=True,
            padding_mode="zeros",
        )[:, 0]
        > 0.5
    )
    return sampled[:, 0], sampled[:, 1], valid


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum(dim=(-2, -1)) / weight.sum(dim=(-2, -1)).clamp_min(1.0e-6)


def _weighted_slope(
    coordinate: torch.Tensor, displacement: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    cm = _weighted_mean(coordinate, weight)
    dm = _weighted_mean(displacement, weight)
    centered_c = coordinate - cm[:, None, None]
    centered_d = displacement - dm[:, None, None]
    return _weighted_mean(centered_c * centered_d, weight) / _weighted_mean(
        centered_c.square(), weight
    ).clamp_min(1.0e-6)


def _patch_summaries(
    forward: LocalTransportMatch, reverse: LocalTransportMatch, *, radius: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[B,16,10]`` summaries and patch-valid flags."""

    batch, height, width = forward.dx.shape
    if height % 4 or width % 4:
        raise ValueError("dense transport grid must divide exactly into 4x4 patches")
    reverse_dx, reverse_dy, reverse_valid = _sample_reverse(reverse, forward)
    outputs: list[torch.Tensor] = []
    valid_outputs: list[torch.Tensor] = []
    patch_h, patch_w = height // 4, width // 4
    for py in range(4):
        for px in range(4):
            ys = slice(py * patch_h, (py + 1) * patch_h)
            xs = slice(px * patch_w, (px + 1) * patch_w)
            valid = forward.valid[:, ys, xs]
            weight = valid.to(forward.dx.dtype)
            dx = forward.dx[:, ys, xs] / float(max(width - 1, 1))
            dy = forward.dy[:, ys, xs] / float(max(height - 1, 1))
            x = torch.linspace(-0.5, 0.5, patch_w, device=dx.device, dtype=dx.dtype)
            y = torch.linspace(-0.5, 0.5, patch_h, device=dx.device, dtype=dx.dtype)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            xx = xx[None].expand(batch, -1, -1)
            yy = yy[None].expand(batch, -1, -1)
            div_x = _weighted_slope(xx, dx, weight)
            div_y = _weighted_slope(yy, dy, weight)
            cycle_valid = valid & reverse_valid[:, ys, xs]
            cycle_weight = cycle_valid.to(dx.dtype)
            cycle = torch.sqrt(
                (forward.dx[:, ys, xs] + reverse_dx[:, ys, xs]).square()
                + (forward.dy[:, ys, xs] + reverse_dy[:, ys, xs]).square()
                + 1.0e-12
            ) / float(max(radius, 1))
            valid_fraction = weight.mean(dim=(-2, -1))
            values = torch.stack(
                (
                    _weighted_mean(dx, weight),
                    _weighted_mean(dy, weight),
                    div_x,
                    div_y,
                    0.5 * (div_x + div_y),
                    _weighted_mean(torch.sqrt(dx.square() + dy.square() + 1.0e-12), weight),
                    _weighted_mean(forward.confidence_margin[:, ys, xs], weight),
                    _weighted_mean(forward.entropy[:, ys, xs], weight),
                    _weighted_mean(cycle, cycle_weight),
                    valid_fraction,
                ),
                dim=-1,
            )
            outputs.append(values)
            valid_outputs.append(valid_fraction > 0.0)
    return torch.stack(outputs, dim=1), torch.stack(valid_outputs, dim=1)


def build_local_temporal_field_features(
    dense_endpoints: torch.Tensor,
    *,
    a5_phase: torch.Tensor,
    a5_log_variance: torch.Tensor,
    sensor_support: torch.Tensor,
    radius: int,
    temperature: float,
    mode: LocalFieldMode = "local",
    time_swap: bool = False,
) -> LocalTemporalFieldBatch:
    """Build the frozen ``[B,16,34]`` X2 field from A5 dense endpoints."""

    if dense_endpoints.ndim != 5 or dense_endpoints.shape[1] != 3:
        raise ValueError("dense endpoints must have shape [B,3,C,H,W]")
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    valid_pairs: list[torch.Tensor] = []
    for first, second in ((0, 1), (1, 2)):
        forward = local_correlation_match(
            dense_endpoints[:, first],
            dense_endpoints[:, second],
            radius=radius,
            temperature=temperature,
        )
        reverse = local_correlation_match(
            dense_endpoints[:, second],
            dense_endpoints[:, first],
            radius=radius,
            temperature=temperature,
        )
        summary, valid = _patch_summaries(forward, reverse, radius=radius)
        pairs.append((summary, summary[..., :9]))
        valid_pairs.append(valid)
    m01, m12 = pairs[0][0], pairs[1][0]
    if time_swap:
        m01, m12 = m12, m01
    difference = m12[..., :9] - m01[..., :9]
    coords = torch.tensor(
        [((px + 0.5) / 4.0, (py + 0.5) / 4.0) for py in range(4) for px in range(4)],
        device=dense_endpoints.device,
        dtype=dense_endpoints.dtype,
    )
    coords = coords[None].expand(dense_endpoints.shape[0], -1, -1)
    base_fields = torch.cat((m01, m12, difference), dim=-1)
    if mode == "global":
        base_fields = base_fields.mean(dim=1, keepdim=True).expand(-1, 16, -1)
        coords = torch.zeros_like(coords)
    elif mode != "local":
        raise ValueError(f"unknown local-field mode: {mode}")
    state = torch.stack((a5_phase, a5_log_variance, sensor_support), dim=-1)
    state = state[:, None].expand(-1, 16, -1)
    features = torch.cat((base_fields, coords, state), dim=-1)
    valid = valid_pairs[0] & valid_pairs[1]
    return LocalTemporalFieldBatch(features, valid, a5_phase)


__all__ = [
    "LocalTemporalPhaseField",
    "LocalTemporalPhaseFieldConfig",
    "LocalTemporalPhaseFieldOutput",
    "build_local_temporal_field_features",
]
