"""Training-only foreground decoders for Garl and compact smoke arms."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class _BasicResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.second = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.relu(values + self.second(self.first(values)))


class _OfficialUpStage(nn.Module):
    def __init__(
        self,
        up_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            up_channels,
            up_channels,
            kernel_size=2,
            stride=2,
        )
        self.fuse = nn.Sequential(
            _ConvBlock(up_channels + skip_channels, out_channels),
            _ConvBlock(out_channels, out_channels),
        )
        # The released Garl config uses decoder_layers=[2,2,2,2], which adds
        # one BasicBlock after every UpBlock.
        self.residual = _BasicResidual(out_channels)

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.residual(self.fuse(torch.cat((self.upsample(values), skip), dim=1)))


class _OfficialResNet50Decoder(nn.Module):
    """Recreate the event-branch decoder topology in the public Garl source."""

    def __init__(self) -> None:
        super().__init__()
        self.up1 = _OfficialUpStage(2048, 1024, 1024)
        self.up2 = _OfficialUpStage(1024, 512, 512)
        self.up3 = _OfficialUpStage(512, 256, 256)
        self.up4 = _OfficialUpStage(256, 64, 64)
        self.output = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 4, kernel_size=2, stride=2),
        )

    def forward(self, pyramid: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(pyramid) != 5:
            raise ValueError("Official Garl decoder requires five ResNet feature maps.")
        before_pool, first, second, third, fourth = pyramid
        values = self.up1(fourth, third)
        values = self.up2(values, second)
        values = self.up3(values, first)
        values = self.up4(values, before_pool)
        return self.output(values)


class ForegroundTrainingDecoder(nn.Module):
    """Return two-class logits for both Garl endpoint foreground masks."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        output_size: int = 256,
        output_channels: int = 4,
        official_resnet50: bool = False,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.output_channels = output_channels
        self.official_resnet50 = official_resnet50
        self.official_decoder = _OfficialResNet50Decoder() if official_resnet50 else None
        self.token_norm = nn.LayerNorm(dim)
        self.compact_decoder = (
            None
            if official_resnet50
            else nn.Sequential(
                nn.Conv2d(dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, output_channels, 1),
            )
        )

    def forward(
        self,
        tokens: torch.Tensor,
        spatial_shape: tuple[int, int],
        *,
        feature_pyramid: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        if self.official_decoder is not None:
            return self.official_decoder(feature_pyramid)
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,P,D].")
        height, width = spatial_shape
        if height * width != tokens.shape[1]:
            raise ValueError("spatial_shape does not match tokens.")
        feature_map = (
            self.token_norm(tokens)
            .transpose(1, 2)
            .reshape(
                tokens.shape[0],
                tokens.shape[-1],
                height,
                width,
            )
        )
        assert self.compact_decoder is not None
        logits = self.compact_decoder(feature_map)
        return functional.interpolate(
            logits,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )


__all__ = ["ForegroundTrainingDecoder"]
