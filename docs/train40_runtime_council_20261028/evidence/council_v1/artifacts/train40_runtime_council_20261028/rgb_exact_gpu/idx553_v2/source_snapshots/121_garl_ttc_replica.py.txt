"""Independent, source-audited Garl-TTC adaptation for local EvTTC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from e_jepa_ttc.models.dense_patch_ttc import DensePatchEventBackbone
from e_jepa_ttc.models.foreground_training_decoder import ForegroundTrainingDecoder
from e_jepa_ttc.models.height_ratio_head import LearnedHeightRatioHead


@dataclass(frozen=True)
class GarlTTCConfig:
    """Ablation switches aligned with NAIL-HNU/Garl-TTC commit 2566612."""

    event_channels: int
    modality: str = "event"
    fusion: str = "late"
    objective: str = "height_ratio"
    backbone: str = "resnet50"
    dim: int = 512
    foreground_supervision: bool = True
    roi_size: int = 128
    foreground_size: int = 256
    rgb_channels: int = 6
    official_event_channels: int = 40
    source_commit: str = "256661242b8a7f5e56aa3c1c02348b30f6e89de6"
    evttc_height_adapter: str = "visible_bbox_height_after_shared_square_roi"

    def __post_init__(self) -> None:
        if self.modality not in {"event", "rgb", "rgbe"}:
            raise ValueError("modality must be event, rgb or rgbe.")
        if self.fusion not in {"early", "late"}:
            raise ValueError("fusion must be early or late.")
        if self.objective not in {"direct", "height_ratio"}:
            raise ValueError("objective must be direct or height_ratio.")
        if self.backbone not in {"resnet50", "compact"}:
            raise ValueError("backbone must be resnet50 or compact.")
        if self.roi_size != 128 or self.foreground_size != 256:
            raise ValueError("Source-audited Garl uses 128x128 inputs and 256x256 masks.")
        if self.backbone == "resnet50" and self.dim != 512:
            raise ValueError("Official Garl ResNet-50 uses a 512-dimensional FC layer.")
        if self.backbone == "resnet50" and self.event_channels != self.official_event_channels:
            raise ValueError(
                "Official Garl event input is two 20-plane time surfaces (40 channels)."
            )


@dataclass
class GarlTTCOutput:
    """Garl prediction plus source-compatible auxiliary outputs."""

    inverse_ttc: torch.Tensor
    ttc_seconds: torch.Tensor
    foreground_logits: torch.Tensor | None
    predicted_height_ratio: torch.Tensor | None
    predicted_heights: torch.Tensor | None
    object_token: torch.Tensor


@dataclass
class _EncoderOutput:
    pooled: torch.Tensor
    dense_tokens: torch.Tensor
    spatial_shape: tuple[int, int]
    feature_pyramid: tuple[torch.Tensor, ...]


class _CompactROIEncoder(nn.Module):
    """Fast smoke-only encoder; never labelled as the paper architecture."""

    def __init__(self, in_channels: int, dim: int) -> None:
        super().__init__()
        self.encoder = DensePatchEventBackbone(in_channels, dim=dim)
        self.output_channels = dim

    def forward(self, values: torch.Tensor) -> _EncoderOutput:
        encoded = self.encoder(values[:, None])
        dense = encoded.dense_tokens[:, 0]
        return _EncoderOutput(
            pooled=encoded.global_token[:, 0],
            dense_tokens=dense,
            spatial_shape=encoded.spatial_shape,
            feature_pyramid=(),
        )


class _ResNet50ROIEncoder(nn.Module):
    """ResNet-50 feature extractor matching the official 2048x4x4 output."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        from torchvision.models import resnet50

        backbone = resnet50(weights=None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.output_channels = 2048

    def forward(self, values: torch.Tensor) -> _EncoderOutput:
        before_pool = self.relu(self.bn1(self.conv1(values)))
        first = self.layer1(self.maxpool(before_pool))
        second = self.layer2(first)
        third = self.layer3(second)
        fourth = self.layer4(third)
        if fourth.shape[-2:] != (4, 4):
            raise ValueError("Official Garl ResNet-50 expects a 128x128 ROI.")
        pooled = self.pool(fourth).flatten(1)
        dense = fourth.flatten(2).transpose(1, 2)
        return _EncoderOutput(
            pooled=pooled,
            dense_tokens=dense,
            spatial_shape=(4, 4),
            feature_pyramid=(before_pool, first, second, third, fourth),
        )


def _encoder(kind: str, in_channels: int, dim: int) -> nn.Module:
    if kind == "resnet50":
        return _ResNet50ROIEncoder(in_channels)
    return _CompactROIEncoder(in_channels, dim)


class GarlTTCReplica(nn.Module):
    """Garl-TTC under EvTTC supervision with explicitly declared adapters."""

    rgb_mean = (0.485, 0.456, 0.406)
    rgb_std = (0.229, 0.224, 0.225)

    def __init__(self, config: GarlTTCConfig) -> None:
        super().__init__()
        self.config = config
        early_channels = config.event_channels + config.rgb_channels
        self.event_encoder = (
            _encoder(config.backbone, config.event_channels, config.dim)
            if config.modality == "event" or (config.modality == "rgbe" and config.fusion == "late")
            else None
        )
        self.rgb_encoder = (
            _encoder(config.backbone, config.rgb_channels, config.dim)
            if config.modality == "rgb" or (config.modality == "rgbe" and config.fusion == "late")
            else None
        )
        self.early_encoder = (
            _encoder(config.backbone, early_channels, config.dim)
            if config.modality == "rgbe" and config.fusion == "early"
            else None
        )
        encoder_width = 2048 if config.backbone == "resnet50" else config.dim
        fusion_width = (
            encoder_width * 2
            if config.modality == "rgbe" and config.fusion == "late"
            else encoder_width
        )
        # The official source has no LayerNorm or activation between these two
        # linear layers.
        self.middle_layer = nn.Linear(fusion_width, config.dim)
        self.height_head = (
            LearnedHeightRatioHead(config.dim) if config.objective == "height_ratio" else None
        )
        self.direct_ttc_head = nn.Linear(config.dim, 1) if config.objective == "direct" else None
        # The source constructs an unused RGB decoder in late fusion.  Omitting
        # that branch is prediction/loss equivalent and materially lowers VRAM.
        self.foreground = (
            ForegroundTrainingDecoder(
                encoder_width,
                output_size=config.foreground_size,
                output_channels=4,
                official_resnet50=config.backbone == "resnet50",
            )
            if config.foreground_supervision
            else None
        )
        self._initialize_like_source()

    def _initialize_like_source(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @classmethod
    def normalize_rgb_pair(cls, rgb_pair: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization independently to both endpoint ROIs."""

        mean = rgb_pair.new_tensor(cls.rgb_mean).view(1, 1, 3, 1, 1)
        std = rgb_pair.new_tensor(cls.rgb_std).view(1, 1, 3, 1, 1)
        return (rgb_pair - mean) / std

    def _encode(
        self,
        event_intervals: torch.Tensor,
        rgb_pair: torch.Tensor | None,
    ) -> tuple[torch.Tensor, _EncoderOutput]:
        if event_intervals.ndim != 4:
            raise ValueError("event_intervals must have shape [B,C,128,128].")
        if event_intervals.shape[1] != self.config.event_channels:
            raise ValueError(
                f"Garl expected {self.config.event_channels} event channels, "
                f"got {event_intervals.shape[1]}."
            )
        if event_intervals.shape[-2:] != (self.config.roi_size, self.config.roi_size):
            raise ValueError("Garl event ROI must be 128x128.")
        rgb_channels: torch.Tensor | None = None
        if rgb_pair is not None:
            if rgb_pair.ndim != 5 or rgb_pair.shape[1:3] != (2, 3):
                raise ValueError("rgb_pair must have shape [B,2,3,128,128].")
            if rgb_pair.shape[-2:] != (self.config.roi_size, self.config.roi_size):
                raise ValueError("Garl RGB ROIs must be 128x128.")
            rgb_channels = self.normalize_rgb_pair(rgb_pair).flatten(1, 2)
        if self.config.modality == "event":
            assert self.event_encoder is not None
            encoded = self.event_encoder(event_intervals)
            return self.middle_layer(encoded.pooled), encoded
        if rgb_channels is None:
            raise ValueError(f"RGB pair is required for modality={self.config.modality}.")
        if self.config.modality == "rgb":
            assert self.rgb_encoder is not None
            encoded = self.rgb_encoder(rgb_channels)
            return self.middle_layer(encoded.pooled), encoded
        if self.config.fusion == "early":
            assert self.early_encoder is not None
            encoded = self.early_encoder(torch.cat((rgb_channels, event_intervals), dim=1))
            return self.middle_layer(encoded.pooled), encoded
        assert self.event_encoder is not None and self.rgb_encoder is not None
        rgb_output = self.rgb_encoder(rgb_channels)
        event_output = self.event_encoder(event_intervals)
        fused = torch.cat((rgb_output.pooled, event_output.pooled), dim=-1)
        return self.middle_layer(fused), event_output

    def forward(
        self,
        event_intervals: torch.Tensor,
        elapsed_s: torch.Tensor,
        *,
        rgb_pair: torch.Tensor | None = None,
    ) -> GarlTTCOutput:
        """Predict from the two endpoint RGB ROIs and two event intervals."""

        token, event_or_single_output = self._encode(event_intervals, rgb_pair)
        if self.config.objective == "height_ratio":
            assert self.height_head is not None
            inverse_ttc, height_ratio, predicted_heights = self.height_head(
                token,
                elapsed_s,
            )
            ttc_seconds = inverse_ttc.reciprocal()
        else:
            assert self.direct_ttc_head is not None
            ttc_seconds = self.direct_ttc_head(token).squeeze(-1)
            inverse_ttc = torch.where(
                ttc_seconds.abs() >= 1e-6,
                ttc_seconds.reciprocal(),
                torch.full_like(ttc_seconds, 1e6),
            )
            height_ratio = None
            predicted_heights = None
        foreground_logits = (
            self.foreground(
                event_or_single_output.dense_tokens,
                event_or_single_output.spatial_shape,
                feature_pyramid=event_or_single_output.feature_pyramid,
            )
            if self.foreground is not None
            else None
        )
        return GarlTTCOutput(
            inverse_ttc=inverse_ttc,
            ttc_seconds=ttc_seconds,
            foreground_logits=foreground_logits,
            predicted_height_ratio=height_ratio,
            predicted_heights=predicted_heights,
            object_token=token,
        )


__all__ = ["GarlTTCConfig", "GarlTTCOutput", "GarlTTCReplica"]
