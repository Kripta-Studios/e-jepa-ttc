"""Object Event TTC v4.20: bounded residual refiner for dense pseudo-flow."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ObjectEventV420Config:
    hidden_dim: int = 32
    residual_limit: float = 2.0
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.residual_limit <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("v4.20 numerical controls must be positive")


class BoxPseudoFlowRefiner(nn.Module):
    """Small shared residual decoder.

    Input channels are raw flow-x, raw flow-y, matching confidence,
    event-only foreground overlap and seed disagreement.  The raw flow is an
    explicit anchor; the network can only add a bounded residual.
    """

    input_channels: int = 5

    def __init__(self, config: ObjectEventV420Config | None = None) -> None:
        super().__init__()
        self.config = config or ObjectEventV420Config()
        h = self.config.hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(self.input_channels, h, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(h, h, kernel_size=3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(h, 2, kernel_size=3, padding=1, bias=False),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.zeros_(final.weight)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError("inputs must be [B,5,H,W]")
        raw_flow = inputs[:, :2]
        residual = self.config.residual_limit * torch.tanh(self.net(inputs))
        return raw_flow + residual, residual


__all__ = ["BoxPseudoFlowRefiner", "ObjectEventV420Config"]
