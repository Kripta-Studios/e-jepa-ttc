"""Smooth signed TTC regression without the reciprocal-phase discontinuity."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


class DirectTTCHead(nn.Module):
    """Small temporal head using lag and sample spacing, with masked padding.

    Availability metadata is deliberately excluded: its historical training and
    transfer adapters use different conventions. The output crosses zero
    continuously and has no reciprocal singularity. Bounds are architectural,
    and the same operational bound is reported separately for every comparator.
    """

    def __init__(self, hidden: int = 64, bound_seconds: float = 60.0) -> None:
        super().__init__()
        if hidden <= 0 or bound_seconds <= 0:
            raise ValueError("positive hidden size and TTC bound required")
        self.bound_seconds = bound_seconds
        self.project = nn.Sequential(nn.Linear(19, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.temporal = nn.GRU(hidden, hidden, batch_first=True)
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, features: Tensor, timing: Tensor, valid: Tensor) -> Tensor:
        """Predict signed seconds from chronological [B,8,17] histories."""
        x = torch.cat((features, timing[:, :, (0, 2)]), -1)
        x = self.project(torch.where(valid[..., None], x, torch.zeros_like(x)))
        # Invalid slots cannot update the recurrent state, even through biases.
        state = features.new_zeros((1, len(features), self.temporal.hidden_size))
        for slot in range(features.shape[1]):
            _, candidate = self.temporal(x[:, slot : slot + 1], state)
            state = torch.where(valid[:, slot][None, :, None], candidate, state)
        coordinate = self.output(state[0]).squeeze(-1).clamp(-10, 10)
        return self.bound_seconds * torch.tanh(torch.sinh(coordinate) / self.bound_seconds)


def symmetric_ttc_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Per-query symmetric seconds, relative and signed-asinh residual losses.

    Each term treats equal over/under residuals in its own coordinate equally;
    the asinh term is not symmetric about a nonzero target in seconds. There is
    no sign-specific penalty, no population-derived test weights or TTC cap on
    the supervision. A .5 s relative denominator floor avoids near-zero blowup.
    """
    error = prediction - target
    seconds = functional.smooth_l1_loss(error / 4, torch.zeros_like(error), reduction="none")
    relative = functional.smooth_l1_loss(
        error / target.abs().clamp_min(0.5), torch.zeros_like(error), reduction="none"
    )
    transformed = functional.smooth_l1_loss(
        torch.asinh(prediction), torch.asinh(target), reduction="none"
    )
    return seconds + relative + 0.25 * transformed
