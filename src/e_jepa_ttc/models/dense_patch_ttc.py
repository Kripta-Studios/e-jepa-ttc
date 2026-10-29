"""Dense multi-scale event backbone without premature global pooling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.token_transformer import EventTubeletTransformerEncoder


@dataclass
class DenseEventFeatures:
    """Per-layer maps, aligned dense tokens and a global diagnostic token."""

    layer_maps: tuple[torch.Tensor, ...]
    layer_tokens: tuple[torch.Tensor, ...]
    dense_tokens: torch.Tensor
    global_token: torch.Tensor
    spatial_shape: tuple[int, int]


class DensePatchEventBackbone(nn.Module):
    """Small 2-D-per-frame backbone sized for a 12 GB laptop GPU."""

    def __init__(
        self,
        in_channels: int,
        *,
        dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
        downsample_stages: int = 3,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or dim < 32 or depth < 2:
            raise ValueError("in_channels, dim and depth must define a non-trivial backbone.")
        if not 1 <= downsample_stages <= depth:
            raise ValueError("downsample_stages must lie within backbone depth.")
        widths = [max(32, dim // 2)] + [dim] * (depth - 1)
        stages: list[nn.Module] = []
        current = in_channels
        for index, width in enumerate(widths):
            stride = 2 if index < downsample_stages else 1
            groups = min(8, width)
            stages.append(
                nn.Sequential(
                    nn.Conv2d(current, width, 3, stride=stride, padding=1, bias=False),
                    nn.GroupNorm(groups, width),
                    nn.GELU(),
                    nn.Conv2d(width, width, 3, padding=1, groups=1, bias=False),
                    nn.GroupNorm(groups, width),
                    nn.GELU(),
                    nn.Dropout2d(dropout),
                )
            )
            current = width
        self.stages = nn.ModuleList(stages)
        self.projections = nn.ModuleList(nn.Conv2d(width, dim, 1) for width in widths)
        self.output_norm = nn.LayerNorm(dim)
        self.output_dim = dim

    def forward(self, events: torch.Tensor) -> DenseEventFeatures:
        """Encode ``[B,T,C,H,W]`` and retain every stage at a shared patch grid."""

        if events.ndim != 5:
            raise ValueError("events must have shape [B,T,C,H,W].")
        batch, steps, channels, height, width = events.shape
        flat = events.reshape(batch * steps, channels, height, width)
        maps: list[torch.Tensor] = []
        value = flat
        for stage in self.stages:
            value = stage(value)
            maps.append(value)
        target_shape = maps[-1].shape[-2:]
        aligned_maps = [
            projection(
                functional.interpolate(
                    layer,
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            for layer, projection in zip(maps, self.projections, strict=True)
        ]
        temporal_maps = tuple(
            layer.reshape(batch, steps, self.output_dim, *target_shape) for layer in aligned_maps
        )
        tokens = tuple(
            self.output_norm(layer.flatten(-2).transpose(-1, -2)) for layer in temporal_maps
        )
        dense = tokens[-1]
        return DenseEventFeatures(
            layer_maps=temporal_maps,
            layer_tokens=tokens,
            dense_tokens=dense,
            global_token=dense.mean(dim=2),
            spatial_shape=(int(target_shape[0]), int(target_shape[1])),
        )


class BaseEventTubeletBackbone(nn.Module):
    """Expose dense/intermediate tokens from the audited BASE SSL encoder."""

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        *,
        allow_random_initialization: bool = False,
    ) -> None:
        super().__init__()
        if checkpoint_path is None:
            if not allow_random_initialization:
                raise ValueError("Random BASE initialization requires explicit authorization.")
            self.encoder = EventTubeletTransformerEncoder(
                21,
                embed_dim=192,
                event_bins=5,
                patch_size=16,
                temporal_patch_size=1,
                depth=6,
                num_heads=6,
                position_encoding="additive",
            )
            self.output_dim = 192
            self.depth = 6
            self.in_channels = 21
            self.event_bins = 5
            self.patch_size = 16
            self.checkpoint_path = None
            self.initialization = "random_grouped_cv_control"
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != "event-tubelet-transformer":
            raise ValueError("BASE checkpoint must contain an event-tubelet-transformer.")
        state = checkpoint.get("encoder_state_dict")
        if not isinstance(state, dict):
            raise ValueError("BASE checkpoint has no encoder_state_dict.")
        in_channels = int(checkpoint["in_channels"])
        event_weight = state["event_embed.weight"]
        embed_dim = int(event_weight.shape[0])
        event_bins = int(checkpoint.get("bins", 5))
        patch_size = int(event_weight.shape[-1])
        temporal_patch_size = int(event_weight.shape[-3])
        layer_indices = sorted(
            {
                int(name.split(".")[1])
                for name in state
                if name.startswith("layers.") and name.split(".")[1].isdigit()
            }
        )
        if not layer_indices or layer_indices != list(range(len(layer_indices))):
            raise ValueError("BASE checkpoint transformer layers are not contiguous.")
        depth = len(layer_indices)
        if embed_dim % 6:
            raise ValueError("Audited BASE embedding dimension is incompatible with six heads.")
        self.encoder = EventTubeletTransformerEncoder(
            in_channels,
            embed_dim=embed_dim,
            event_bins=event_bins,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            depth=depth,
            num_heads=6,
            position_encoding="additive",
        )
        self.encoder.load_state_dict(state, strict=True)
        self.output_dim = embed_dim
        self.depth = depth
        self.in_channels = in_channels
        self.event_bins = event_bins
        self.patch_size = patch_size
        self.checkpoint_path = Path(checkpoint_path)
        self.initialization = "audited_ssl_checkpoint"

    def forward(self, events: torch.Tensor) -> DenseEventFeatures:
        """Encode ``[B,T,21,H,W]`` without discarding BASE patch tokens."""

        if events.ndim != 5 or events.shape[2] != self.in_channels:
            raise ValueError(
                f"BASE EventTubelet input must have shape [B,T,{self.in_channels},H,W]."
            )
        batch, steps, channels, height, width = events.shape
        flat = events.reshape(batch * steps, channels, height, width)
        outputs = self.encoder.forward_intermediate_tokens(
            flat,
            tuple(range(self.depth)),
        )
        layers = tuple(
            value.reshape(batch, steps, value.shape[1], self.output_dim) for value in outputs
        )
        grid_width = width // self.patch_size
        if grid_width <= 0 or layers[-1].shape[2] % grid_width:
            raise ValueError("BASE token grid is incompatible with the input resolution.")
        spatial_shape = (int(layers[-1].shape[2] // grid_width), int(grid_width))
        return DenseEventFeatures(
            layer_maps=(),
            layer_tokens=layers,
            dense_tokens=layers[-1],
            global_token=layers[-1].mean(dim=2),
            spatial_shape=spatial_shape,
        )


class DensePatchTTCHead(nn.Module):
    """Attention-pool dense patches into a log-TTC prediction."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool the last temporal step from ``[B,T,P,D]``."""

        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,T,P,D].")
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(query, tokens[:, -1], tokens[:, -1], need_weights=False)
        return self.head(pooled[:, 0]).squeeze(-1)


__all__ = [
    "BaseEventTubeletBackbone",
    "DenseEventFeatures",
    "DensePatchEventBackbone",
    "DensePatchTTCHead",
]
