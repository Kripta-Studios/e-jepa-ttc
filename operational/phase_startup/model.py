"""Sign-equivariant four-expert routing with explicit availability masks."""

import torch
from torch import nn


class ExpertRouter(nn.Module):
    """Learn reliability, allowing recovery from either neural or geometric errors."""

    center: torch.Tensor
    scale: torch.Tensor

    def __init__(self, mode: str = "quality", coordinate: str = "seconds") -> None:
        super().__init__()
        if mode not in ("quality", "native"):
            raise ValueError("Expected quality or native control")
        if coordinate not in ("seconds", "inverse"):
            raise ValueError("Expected seconds or inverse coordinate")
        self.inverse_coordinate = coordinate == "inverse"
        self.quality_mode = mode == "quality"
        dim, hidden = (33, 18) if self.quality_mode else (13, 38)
        self.register_buffer("center", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))
        first, last = nn.Linear(dim, hidden), nn.Linear(hidden, 4)
        self.gate = nn.Sequential(first, nn.Tanh(), last)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([2.0, 0.0, 0.0, 0.0]))

    @torch.jit.export
    def inputs(
        self, native: torch.Tensor, geometry: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        available = torch.isfinite(native)
        geo_available = torch.isfinite(geometry) & torch.isfinite(quality).all(1)
        safe = torch.where(available, native, torch.zeros_like(native))
        geo = torch.where(geo_available, geometry, torch.zeros_like(geometry))
        signs = torch.sign(safe)
        pairs = torch.stack(
            (signs[:, 0] * signs[:, 1], signs[:, 0] * signs[:, 2], signs[:, 1] * signs[:, 2]), 1
        )
        features = torch.cat(
            (
                torch.log1p(safe.abs()),
                available.to(safe.dtype),
                pairs,
                geo_available.to(safe.dtype)[:, None],
                signs * torch.sign(geo)[:, None],
            ),
            1,
        )
        if self.quality_mode:
            q = torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
            q = torch.cat((q[:, :5].abs(), q[:, 5:]), 1)
            return torch.cat((features, q), 1)
        return features

    def forward(
        self, native: torch.Tensor, geometry: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        """Return signed seconds; no supplied history preserves exact native F0."""
        experts = torch.cat((native, geometry[:, None]), 1)
        available = torch.cat(
            (
                torch.isfinite(native),
                (torch.isfinite(geometry) & torch.isfinite(quality).all(1))[:, None],
            ),
            1,
        )
        if self.inverse_coordinate:
            available = available & (experts.abs() > 0)
        safe = torch.where(available, experts, torch.zeros_like(experts)).clamp(-60.0, 60.0)
        x = ((self.inputs(native, geometry, quality) - self.center) / self.scale).clamp(-8.0, 8.0)
        logits = self.gate(x).masked_fill(~available, -10000.0)
        weights = torch.softmax(logits, 1) * available.to(logits.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp(min=1e-12)
        if self.inverse_coordinate:
            bounded = torch.where(safe < 0, -safe.abs().clamp(min=0.1), safe.abs().clamp(min=0.1))
            rate = (weights / bounded).sum(1)
            bounded_rate = torch.where(
                rate < 0, -rate.abs().clamp(min=1 / 60), rate.abs().clamp(min=1 / 60)
            )
            # Exact cancellation carries no approach direction; retain native F0.
            prediction = torch.where(rate == 0, native[:, 0], bounded_rate.reciprocal())
        else:
            prediction = (weights * safe).sum(1)
        has_history = torch.isfinite(quality[:, 9]) & (quality[:, 9] > 0)
        return torch.where(has_history & available.any(1), prediction, native[:, 0])
