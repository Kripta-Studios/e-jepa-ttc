"""IQ inverse routing with explicit native clocks/counts or a larger base-only gate."""

import torch
from torch import nn


class NativeSupportRouter(nn.Module):
    """F2 adds ten causal support coordinates; Fcap adds hidden units only."""

    center: torch.Tensor
    scale: torch.Tensor
    support_mask: torch.Tensor

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in ("F2", "Fcap"):
            raise ValueError("Expected F2 or Fcap")
        self.use_support = mode == "F2"
        dimension, hidden = (43, 18) if self.use_support else (33, 23)
        self.register_buffer("center", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))
        self.register_buffer("support_mask", torch.ones(10))
        first, last = nn.Linear(dimension, hidden), nn.Linear(hidden, 4)
        self.gate = nn.Sequential(first, nn.Tanh(), last)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([2.0, 0.0, 0.0, 0.0]))

    @torch.jit.export
    def base_inputs(
        self, native: torch.Tensor, geometry: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        """The unchanged33 IQ coordinates, including its geometric quality treatment."""
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
        q = torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)
        q = torch.cat((q[:, :5].abs(), q[:, 5:]), 1)
        return torch.cat((features, q), 1)

    @torch.jit.export
    def inputs(
        self,
        native: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """Keep missingness metadata outside the ten declared model coordinates."""
        base = self.base_inputs(native, geometry, quality)
        if self.use_support:
            if support.ndim != 2 or support.shape[0] != len(native) or support.shape[1] != 10:
                raise ValueError("Expected aligned native support[N,10]")
            if not torch.isfinite(support).all():
                raise ValueError("Support must be finite; unavailable context uses audited zeros")
            return torch.cat((base, support), 1)
        return base

    @torch.jit.export
    def normalized_inputs(
        self,
        native: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the train-only mask to new coordinates after original normalization."""
        x = ((self.inputs(native, geometry, quality, support) - self.center) / self.scale).clamp(
            -8.0, 8.0
        )
        if self.use_support:
            x = torch.cat((x[:, :33], x[:, 33:] * self.support_mask), 1)
        return x

    def forward(
        self,
        native: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        """Preserve original inverse fusion, sign handling, limits and exact F0 fallback."""
        experts = torch.cat((native, geometry[:, None]), 1)
        available = torch.cat(
            (
                torch.isfinite(native),
                (torch.isfinite(geometry) & torch.isfinite(quality).all(1))[:, None],
            ),
            1,
        )
        available = available & (experts.abs() > 0)
        safe = torch.where(available, experts, torch.zeros_like(experts)).clamp(-60.0, 60.0)
        x = self.normalized_inputs(native, geometry, quality, support)
        logits = self.gate(x).masked_fill(~available, -10000.0)
        weights = torch.softmax(logits, 1) * available.to(logits.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp(min=1e-12)
        bounded = torch.where(safe < 0, -safe.abs().clamp(min=0.1), safe.abs().clamp(min=0.1))
        rate = (weights / bounded).sum(1)
        bounded_rate = torch.where(
            rate < 0, -rate.abs().clamp(min=1 / 60), rate.abs().clamp(min=1 / 60)
        )
        prediction = torch.where(rate == 0, native[:, 0], bounded_rate.reciprocal())
        has_history = torch.isfinite(quality[:, 9]) & (quality[:, 9] > 0)
        return torch.where(has_history & available.any(1), prediction, native[:, 0])
