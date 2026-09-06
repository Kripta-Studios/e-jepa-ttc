"""Compact causal phase refiners: masked GRU and a bounded-window Transformer.

The primary residual uses the label-free median of the CURRENT expert phases,
not an in-sample trained router. The separate cost head estimates ORIGINAL expert
costs only. It does not claim to predict the corrected predictor's cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from .phase import MAX_PHASE, MIN_PHASE, emitted_phase, pinball, relative_cost_targets


@dataclass(frozen=True)
class TemporalConfig:
    feature_count: int = 17
    hidden: int = 64
    backbone: str = "gru"
    output_mode: str = "residual"

    def __post_init__(self) -> None:
        if self.feature_count not in {17, 145} or self.hidden not in {64, 128, 160}:
            raise ValueError("unsupported frozen feature/capacity setting")
        if self.backbone not in {"gru", "transformer"} or self.output_mode not in {
            "residual",
            "free",
            "selector",
        }:
            raise ValueError("unsupported backbone or output mode")
        if self.backbone == "transformer" and self.hidden != 128:
            raise ValueError("Transformer control has width128")


class TemporalRefiner(nn.Module):
    def __init__(self, cfg: TemporalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.project = nn.Sequential(
            nn.Linear(cfg.feature_count + 4, h), nn.LayerNorm(h), nn.SiLU()
        )
        if cfg.backbone == "gru":
            self.cells = nn.ModuleList([nn.GRUCell(h, h), nn.GRUCell(h, h)])
        else:
            block = nn.TransformerEncoderLayer(
                h,
                4,
                dim_feedforward=2 * h,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.location = nn.Linear(h, 1)
        self.width = nn.Linear(h, 2)
        self.cost = nn.Linear(h, 2)
        for head in (self.location, self.width, self.cost):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, features: Tensor, times: Tensor, valid: Tensor, experts: Tensor
    ) -> dict[str, Tensor]:
        if features.ndim != 3:
            raise ValueError("features must be [B,L,F]")
        b, length, f = features.shape
        if b == 0 or length < 1 or length > 16 or f != self.cfg.feature_count:
            raise ValueError("input dimensions outside contract")
        if times.shape != (b, length, 4) or valid.shape != (b, length) or valid.dtype != torch.bool:
            raise ValueError("timing/mask mismatch")
        if experts.shape != (b, 3) or not torch.isfinite(experts).all():
            raise ValueError("finite current expert phases required")
        if not valid[:, -1].all() or (valid[:, :-1] & ~valid[:, 1:]).any():
            raise ValueError("histories must have contiguous valid suffixes")
        if not torch.isfinite(features[valid]).all() or not torch.isfinite(times[valid]).all():
            raise ValueError("nonfinite valid features/timing")
        safe_x = torch.where(valid[..., None], features, torch.zeros_like(features))
        safe_t = torch.where(valid[..., None], times, torch.zeros_like(times))
        z = self.project(torch.cat((safe_x, safe_t), -1))
        if self.cfg.backbone == "gru":
            states = [z.new_zeros((b, self.cfg.hidden)), z.new_zeros((b, self.cfg.hidden))]
            for t in range(length):
                proposal = self.cells[0](z[:, t], states[0])
                states[0] = torch.where(valid[:, t, None], proposal, states[0])
                proposal = self.cells[1](states[0], states[1])
                states[1] = torch.where(valid[:, t, None], proposal, states[1])
            hidden = states[-1]
        else:
            # Move valid suffixes to the left: no attention query has all keys masked.
            counts = valid.sum(1)
            position = torch.arange(length, device=z.device)[None]
            take = (position + (length - counts)[:, None]).clamp_max(length - 1)
            packed = z.gather(1, take[..., None].expand(-1, -1, z.shape[-1]))
            pad = position >= counts[:, None]
            packed = torch.where(pad[..., None], torch.zeros_like(packed), packed)
            causal = torch.ones((length, length), device=z.device, dtype=torch.bool).triu(1)
            encoded = self.encoder(packed, mask=causal, src_key_padding_mask=pad)
            hidden = encoded[torch.arange(b, device=z.device), counts - 1]
        anchor = experts.median(-1).values
        delta = 0.03 * self.location(hidden).squeeze(-1)
        location = delta if self.cfg.output_mode == "free" else anchor + delta
        widths = 0.03 * functional.softplus(self.width(hidden))
        risk = torch.cat((experts.new_zeros((b, 1)), self.cost(hidden)), -1)
        selected = risk.argmin(-1)
        point = emitted_phase(location)
        if self.cfg.output_mode == "selector":
            point = experts.gather(1, selected[:, None]).squeeze(1)
        return {
            "point_phase": point,
            "raw_location": location,
            "raw_residual": delta,
            "q10": (location - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
            "q90": (location + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
            "relative_cost": risk,
            "expert_index": selected,
        }


def training_loss(
    output: dict[str, Tensor],
    truth: Tensor,
    experts: Tensor,
    mass: Tensor,
    population: int,
    *,
    selector_only: bool = False,
) -> Tensor:
    """Unbiased uniform-query minibatch estimate; NO batch mass renormalization.

    Learned quantiles describe phase, not calibrated collision probabilities.
    MSE on relative-cost targets estimates mean original-expert cost differences.
    """
    if truth.ndim != 1 or mass.shape != truth.shape or population <= 0:
        raise ValueError("invalid truth/mass/population")
    if not torch.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("invalid global query mass")
    cost_target = relative_cost_targets(experts, truth)
    cost_loss = (output["relative_cost"][:, 1:] - cost_target).square().mean(-1)
    if selector_only:
        per_row = cost_loss
    else:
        l1 = (output["point_phase"] - truth).abs() / 0.03
        quant = (pinball(output["q10"], truth, 0.1) + pinball(output["q90"], truth, 0.9)) / 0.03
        per_row = l1 + 0.1 * quant + 0.01 * cost_loss
    return (population * mass * per_row).mean()
