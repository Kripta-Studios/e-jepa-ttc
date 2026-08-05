"""Event-only causal TTC diagnostic for Object Event v4.1.

The v4 fused screen learned almost exclusively from observable box motion.  This
module removes that shortcut completely and asks a bounded falsification
question: can the corrected common-coordinate ``t0/t1/t2`` event tensor support
signed-expansion learning on its own?

The model is intentionally standalone and small.  It does not import the v4
motion/fusion model, consume boxes, or receive TTC-derived geometry.  Temporal
reversal is measured and may be regularised by the trainer, but the prediction
is *not* constructed as ``forward - reverse``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.data.event_v4_geometry import EVENT_V4_CHANNEL_COUNT, EVENT_V4_STEPS


@dataclass(frozen=True)
class ObjectEventV41Config:
    in_channels: int = EVENT_V4_CHANNEL_COUNT
    temporal_steps: int = EVENT_V4_STEPS
    input_size: int = 64
    stem_dim: int = 48
    embed_dim: int = 96
    spatial_grid: int = 4
    encoded_hidden_dim: int = 384
    activity_hidden_dim: int = 128
    dropout: float = 0.0
    max_abs_expansion: float = 0.25
    ttc_clip_seconds: float = 60.0
    min_abs_expansion_for_ttc: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.in_channels != EVENT_V4_CHANNEL_COUNT:
            raise ValueError(f"v4.1 requires {EVENT_V4_CHANNEL_COUNT} event channels")
        if self.temporal_steps != EVENT_V4_STEPS:
            raise ValueError(f"v4.1 requires {EVENT_V4_STEPS} causal steps")
        if min(self.input_size, self.stem_dim, self.embed_dim, self.spatial_grid) <= 0:
            raise ValueError("v4.1 dimensions must be positive")
        if self.input_size < 16:
            raise ValueError("input_size must be at least 16")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if not 0.0 < self.max_abs_expansion < 1.0:
            raise ValueError("max_abs_expansion must lie in (0,1)")


@dataclass
class ObjectEventV41Output:
    expansion: torch.Tensor
    reverse_expansion: torch.Tensor
    encoded_expansion: torch.Tensor
    activity_expansion: torch.Tensor
    raw_score: torch.Tensor
    reverse_raw_score: torch.Tensor
    reversal_consistency_error: torch.Tensor
    endpoint_embeddings: torch.Tensor
    spatial_embeddings: torch.Tensor


def safe_ttc_from_expansion(
    expansion: torch.Tensor,
    delta_t_s: torch.Tensor,
    *,
    minimum_abs_expansion: float,
    clip_seconds: float,
) -> torch.Tensor:
    """Convert expansion to TTC for reporting only, never for training loss."""

    sign = torch.where(expansion < 0.0, -torch.ones_like(expansion), torch.ones_like(expansion))
    denominator = sign * expansion.abs().clamp_min(minimum_abs_expansion)
    return (delta_t_s / denominator).clamp(-clip_seconds, clip_seconds)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return functional.gelu(inputs + self.block(inputs))


class _FrameEventEncoder(nn.Module):
    """Shared spatial encoder applied independently to each causal event step."""

    def __init__(self, config: ObjectEventV41Config) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.stem_dim,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(config.stem_dim), config.stem_dim),
            nn.GELU(),
            nn.Conv2d(
                config.stem_dim,
                config.embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(config.embed_dim), config.embed_dim),
            nn.GELU(),
            _ResidualSpatialBlock(config.embed_dim),
            _ResidualSpatialBlock(config.embed_dim),
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.network(events)


def _temporal_levels_and_differences(values: torch.Tensor) -> torch.Tensor:
    """Build causal levels, velocities, acceleration and interactions."""

    if values.ndim != 3 or values.shape[1] != EVENT_V4_STEPS:
        raise ValueError("Temporal features must have shape [B,3,D]")
    v0, v1, v2 = values.unbind(dim=1)
    d01 = v1 - v0
    d12 = v2 - v1
    acceleration = d12 - d01
    return torch.cat(
        (
            v0,
            v1,
            v2,
            d01,
            d12,
            acceleration,
            d01.abs(),
            d12.abs(),
            acceleration.abs(),
            v0 * v1,
            v1 * v2,
            d01 * d12,
        ),
        dim=-1,
    )


def _temporal_maps(maps: torch.Tensor) -> torch.Tensor:
    if maps.ndim != 5 or maps.shape[1] != EVENT_V4_STEPS:
        raise ValueError("Encoded maps must have shape [B,3,D,H,W]")
    v0, v1, v2 = maps.unbind(dim=1)
    d01 = v1 - v0
    d12 = v2 - v1
    acceleration = d12 - d01
    return torch.cat(
        (
            v0,
            v1,
            v2,
            d01,
            d12,
            acceleration,
            d01.abs(),
            d12.abs(),
            acceleration.abs(),
        ),
        dim=1,
    )


def _initialise_small_output(head: nn.Sequential) -> None:
    final = head[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("Expected a linear output layer")
    nn.init.normal_(final.weight, mean=0.0, std=1.0e-3)
    nn.init.zeros_(final.bias)


class ObjectEventTTCV41(nn.Module):
    """Small direct event-only model used as a learnability gate."""

    def __init__(self, config: ObjectEventV41Config | None = None) -> None:
        super().__init__()
        self.config = config or ObjectEventV41Config()
        self.encoder = _FrameEventEncoder(self.config)

        # Nine map blocks (levels/differences) pooled to a small spatial grid,
        # plus twelve endpoint-statistic blocks.  Keeping a spatial grid is
        # essential: global averages alone discarded the expansion pattern in v4.
        encoded_dim = (
            9 * self.config.embed_dim * self.config.spatial_grid**2
            + 12 * self.config.embed_dim * 2
        )
        self.encoded_head = nn.Sequential(
            nn.LayerNorm(encoded_dim),
            nn.Linear(encoded_dim, self.config.encoded_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.encoded_hidden_dim, self.config.encoded_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.encoded_hidden_dim // 2, 1),
        )

        # Per-channel mean/std for t0, first difference and second difference.
        activity_dim = self.config.in_channels * 6
        self.activity_head = nn.Sequential(
            nn.LayerNorm(activity_dim),
            nn.Linear(activity_dim, self.config.activity_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.activity_hidden_dim, 1),
        )
        self.branch_scale = nn.Parameter(torch.zeros(2))
        _initialise_small_output(self.encoded_head)
        _initialise_small_output(self.activity_head)

    def _resize(self, events: torch.Tensor) -> torch.Tensor:
        expected = (self.config.temporal_steps, self.config.in_channels)
        if events.ndim != 5 or events.shape[1:3] != expected:
            raise ValueError(
                f"events must be [B,{expected[0]},{expected[1]},H,W], got {tuple(events.shape)}"
            )
        if events.shape[-2:] == (self.config.input_size, self.config.input_size):
            return events
        batch, steps, channels, height, width = events.shape
        flat = events.reshape(batch * steps, channels, height, width)
        resized = functional.interpolate(
            flat.float(),
            size=(self.config.input_size, self.config.input_size),
            mode="area",
        ).to(events.dtype)
        return resized.reshape(
            batch,
            steps,
            channels,
            self.config.input_size,
            self.config.input_size,
        )

    def _activity_feature(self, events: torch.Tensor) -> torch.Tensor:
        mean = events.float().mean(dim=(-2, -1)).to(events.dtype)
        std = events.float().std(dim=(-2, -1), unbiased=False).to(events.dtype)
        m0, m1, m2 = mean.unbind(dim=1)
        s0, s1, s2 = std.unbind(dim=1)
        return torch.cat((m0, m1 - m0, m2 - m1, s0, s1 - s0, s2 - s1), dim=-1)

    def _forward_one_order(
        self, events: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        resized = self._resize(events)
        batch, steps, channels, height, width = resized.shape
        encoded = self.encoder(resized.reshape(batch * steps, channels, height, width))
        _, embedding_dim, encoded_height, encoded_width = encoded.shape
        maps = encoded.reshape(batch, steps, embedding_dim, encoded_height, encoded_width)

        spatial_mean = maps.mean(dim=(-2, -1))
        spatial_std = maps.float().std(dim=(-2, -1), unbiased=False).to(maps.dtype)
        pooled_maps = functional.adaptive_avg_pool2d(
            _temporal_maps(maps),
            output_size=(self.config.spatial_grid, self.config.spatial_grid),
        ).flatten(start_dim=1)
        encoded_feature = torch.cat(
            (
                pooled_maps,
                _temporal_levels_and_differences(spatial_mean),
                _temporal_levels_and_differences(spatial_std),
            ),
            dim=-1,
        )
        encoded_raw = self.encoded_head(encoded_feature).squeeze(-1)
        activity_raw = self.activity_head(self._activity_feature(resized)).squeeze(-1)
        scale = functional.softplus(self.branch_scale)
        raw = scale[0] * encoded_raw + scale[1] * activity_raw
        return raw, encoded_raw, activity_raw, spatial_mean, spatial_std

    def forward(self, events: torch.Tensor) -> ObjectEventV41Output:
        raw, encoded_raw, activity_raw, endpoints, spatial = self._forward_one_order(events)
        reverse_raw, _, _, _, _ = self._forward_one_order(torch.flip(events, dims=(1,)))
        maximum = self.config.max_abs_expansion
        expansion = maximum * torch.tanh(raw)
        reverse_expansion = maximum * torch.tanh(reverse_raw)
        encoded_expansion = maximum * torch.tanh(encoded_raw)
        activity_expansion = maximum * torch.tanh(activity_raw)
        return ObjectEventV41Output(
            expansion=expansion,
            reverse_expansion=reverse_expansion,
            encoded_expansion=encoded_expansion,
            activity_expansion=activity_expansion,
            raw_score=raw,
            reverse_raw_score=reverse_raw,
            reversal_consistency_error=(expansion + reverse_expansion).abs(),
            endpoint_embeddings=endpoints,
            spatial_embeddings=spatial,
        )


__all__ = [
    "ObjectEventTTCV41",
    "ObjectEventV41Config",
    "ObjectEventV41Output",
    "safe_ttc_from_expansion",
]
