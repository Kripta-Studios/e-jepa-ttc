"""Tiny CNN supervised TTC regressor."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Small residual block for dense event representations."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block."""

        return self.activation(x + self.net(x))


class TinyCNNRegressor(nn.Module):
    """Compact CNN predicting log-TTC from voxel grids."""

    def __init__(self, in_channels: int, width: int = 48) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            ResidualBlock(width),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            ResidualBlock(width * 2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),
            ResidualBlock(width * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4, width * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(width * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict log-TTC for a batch of voxel grids."""

        return self.head(self.backbone(x)).squeeze(-1)
