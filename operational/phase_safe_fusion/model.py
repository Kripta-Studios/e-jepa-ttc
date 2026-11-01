"""Select a direction before combining magnitudes; never cancel signed rates."""

import math

import torch
from torch import nn

from operational.phase_startup.model import ExpertRouter


class SafeFusion(nn.Module):
    """Fuse same-sign experts or learn a bounded correction to valid geometry.

    Missing/stale/invalid history returns native F0 bitwise. No target, sequence
    identity, absolute clock or domain name is an inference feature.
    """

    def __init__(self, mode: str = "log") -> None:
        super().__init__()
        if mode not in ("log", "residual"):
            raise ValueError("Expected log or residual")
        self.residual = mode == "residual"
        self.features = ExpertRouter("quality", "inverse")
        # Reuse the existing 33-feature contract, not its signed-rate forward.
        self.correction = nn.Linear(33, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    @torch.jit.export
    def inputs(
        self, native: torch.Tensor, geometry: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        """Return inference-only reliability features for train-only normalization."""
        return self.features.inputs(native, geometry, quality)

    def forward(
        self, native: torch.Tensor, geometry: torch.Tensor, quality: torch.Tensor
    ) -> torch.Tensor:
        """Return signed TTC seconds without opposite-sign rate cancellation."""
        experts = torch.cat((native, geometry[:, None]), 1)
        quality_ok = torch.isfinite(quality).all(1)
        available = torch.isfinite(experts) & (experts.abs() > 0)
        available[:, 3] = available[:, 3] & quality_ok
        safe = torch.where(available, experts, torch.ones_like(experts))
        signs = torch.sign(safe)
        x = (
            (self.inputs(native, geometry, quality) - self.features.center) / self.features.scale
        ).clamp(-8.0, 8.0)
        logits = self.features.gate(x).masked_fill(~available, -10000.0)
        weights = torch.softmax(logits, 1) * available.to(logits.dtype)
        vote = (weights * signs).sum(1)
        first = available.to(torch.int64).argmax(1)
        tie_sign = signs.gather(1, first[:, None]).squeeze(1)
        direction = torch.where(vote == 0, tie_sign, torch.sign(vote))
        admitted = available & (signs == direction[:, None])
        same = weights * admitted.to(weights.dtype)
        same = same / same.sum(1, keepdim=True).clamp(min=1e-12)
        magnitude = torch.exp((same * safe.abs().clamp(0.1, 60).log()).sum(1))
        prediction = direction * magnitude
        if self.residual:
            factor = torch.exp(math.log(2.0) * torch.tanh(self.correction(x).squeeze(1)))
            corrected = safe[:, 3].clamp(-60.0, 60.0) * factor
            prediction = torch.where(available[:, 3], corrected, prediction)
        history = quality_ok & (quality[:, 9] > 0) & (quality[:, 11] <= 0.15)
        return torch.where(history & available.any(1), prediction, native[:, 0])
