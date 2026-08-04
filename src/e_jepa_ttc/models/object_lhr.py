"""Object-centric visible-height-ratio TTC model.

The model consumes two ROI event representations for one tracked object. It
predicts one positive visible height per endpoint and derives signed TTC through

    tau = delta_t / (1 - h1 / h2).

No TTC label, bounding box, 3-D geometry, or category enters the network.
Bounding boxes are used only offline to construct the object ROI cache.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.highres_factorized import (
    EJEPATubeletLHR,
    EJEPATubeletLHRConfig,
    HighResFeatures,
)


@dataclass(frozen=True)
class ObjectCentricLHRConfig:
    """Architecture controls for the object-centric LHR candidate."""

    in_channels: int = 21
    embed_dim: int = 192
    patch_size: int = 16
    spatial_window: int = 8
    heads: int = 6
    spatial_depth: int = 1
    temporal_depth: int = 2
    temporal_mixer: str = "block_causal"
    merge_2x2: bool = False
    global_attention: bool = False
    memory_budget_gb: float = 12.0
    pooling: str = "query"
    query_count: int = 8
    head_hidden_dim: int = 192
    mask_decoder: bool = True
    mask_size: int = 128
    initial_height_px: float = 96.0
    min_height_px: float = 1.0
    max_height_px: float = 1024.0
    ttc_clip_seconds: float = 60.0
    denominator_epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.embed_dim <= 0 or self.head_hidden_dim <= 0:
            raise ValueError("Object-LHR channel and hidden dimensions must be positive")
        if self.mask_size <= 0:
            raise ValueError("mask_size must be positive")
        if not 0 < self.min_height_px < self.max_height_px:
            raise ValueError("Height bounds must satisfy 0 < min < max")
        if not self.min_height_px <= self.initial_height_px <= self.max_height_px:
            raise ValueError("initial_height_px must lie inside the configured bounds")
        if self.ttc_clip_seconds <= 0 or self.denominator_epsilon <= 0:
            raise ValueError("TTC clipping and denominator epsilon must be positive")

    def backbone_config(self) -> EJEPATubeletLHRConfig:
        """Return the exact transferable JEPA backbone configuration."""

        return EJEPATubeletLHRConfig(
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            patch_size=self.patch_size,
            spatial_window=self.spatial_window,
            heads=self.heads,
            spatial_depth=self.spatial_depth,
            temporal_depth=self.temporal_depth,
            temporal_mixer=self.temporal_mixer,
            merge_2x2=self.merge_2x2,
            global_attention=self.global_attention,
            memory_budget_gb=self.memory_budget_gb,
            pooling=self.pooling,
            query_count=self.query_count,
        )


@dataclass
class ObjectCentricLHROutput:
    """Geometric predictions and dense diagnostics."""

    log_visible_heights: torch.Tensor
    visible_heights_px: torch.Tensor
    log_height_ratio: torch.Tensor
    height_ratio: torch.Tensor
    ttc_mean_seconds: torch.Tensor
    direction_logits: torch.Tensor
    endpoint_embeddings: torch.Tensor
    pair_embedding: torch.Tensor
    mask_logits: torch.Tensor | None
    features: HighResFeatures


def stable_ttc_from_log_ratio(
    log_ratio: torch.Tensor,
    delta_t_s: torch.Tensor,
    *,
    denominator_epsilon: float,
    clip_seconds: float,
) -> torch.Tensor:
    """Convert log(h1/h2) to signed TTC without division singularities."""

    if log_ratio.shape != delta_t_s.shape:
        raise ValueError("log_ratio and delta_t_s must have identical shapes")
    ratio = torch.exp(log_ratio.clamp(-12.0, 12.0))
    denominator = 1.0 - ratio
    sign = torch.where(
        denominator >= 0,
        torch.ones_like(denominator),
        -torch.ones_like(denominator),
    )
    safe = sign * denominator.abs().clamp_min(denominator_epsilon)
    return (delta_t_s / safe).clamp(-clip_seconds, clip_seconds)


class ObjectCentricLHR(nn.Module):
    """JEPA-compatible object ROI encoder with a geometric LHR readout."""

    def __init__(self, config: ObjectCentricLHRConfig | None = None) -> None:
        super().__init__()
        self.config = config or ObjectCentricLHRConfig()
        self.encoder = EJEPATubeletLHR(self.config.backbone_config())
        # The legacy direct-TTC/collision readouts belong to the historical
        # full-frame model. They are intentionally frozen and never optimized
        # by the object-centric route.
        for legacy_head in (self.encoder.ttc_head, self.encoder.collision_head):
            for parameter in legacy_head.parameters():
                parameter.requires_grad_(False)
        hidden = self.config.head_hidden_dim
        self.height_head = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim),
            nn.Linear(self.config.embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        pair_dim = self.config.embed_dim * 3
        self.pair_projector = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.direction_head = nn.Linear(hidden, 2)
        if self.config.mask_decoder:
            self.mask_decoder: nn.Module | None = nn.Sequential(
                nn.Conv2d(self.config.embed_dim, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, kernel_size=1),
            )
        else:
            self.mask_decoder = None
        self._initialize_height_head()

    def _initialize_height_head(self) -> None:
        final = self.height_head[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("height_head must end with a Linear layer")
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        log_min = math.log(self.config.min_height_px)
        log_max = math.log(self.config.max_height_px)
        fraction = (math.log(self.config.initial_height_px) - log_min) / (log_max - log_min)
        fraction = min(max(fraction, 1e-6), 1.0 - 1e-6)
        nn.init.constant_(final.bias, math.log(fraction / (1.0 - fraction)))

    def load_exact_backbone_state_dict(
        self,
        source_state: Mapping[str, Any],
        source_config: EJEPATubeletLHRConfig | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Load only an exact label-free JEPA encoder state."""

        return self.encoder.load_exact_backbone_state_dict(source_state, source_config)

    def backbone_structural_config(self) -> dict[str, Any]:
        """Return the exact architecture identity required by SSL transfer."""

        return self.encoder.backbone_structural_config()

    def _decode_masks(self, features: HighResFeatures) -> torch.Tensor | None:
        if self.mask_decoder is None:
            return None
        batch, steps, patches, dim = features.tokens.shape
        grid_h = features.encoded_grid_height
        grid_w = features.encoded_grid_width
        if grid_h * grid_w != patches:
            raise ValueError("Dense token count does not match encoded grid geometry")
        grid = features.tokens.reshape(batch, steps, grid_h, grid_w, dim)
        grid = grid.permute(0, 1, 4, 2, 3).reshape(batch * steps, dim, grid_h, grid_w)
        logits = self.mask_decoder(grid)
        logits = functional.interpolate(
            logits,
            size=(self.config.mask_size, self.config.mask_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits.reshape(batch, steps, 1, self.config.mask_size, self.config.mask_size)

    def forward(
        self,
        events: torch.Tensor,
        delta_t_s: torch.Tensor,
    ) -> ObjectCentricLHROutput:
        """Predict endpoint heights and derive signed TTC."""

        if events.ndim != 5 or events.shape[1] != 2:
            raise ValueError("events must have shape [B,2,C,H,W]")
        if events.shape[2] != self.config.in_channels:
            raise ValueError(
                f"events have {events.shape[2]} channels; expected {self.config.in_channels}"
            )
        if delta_t_s.shape != (events.shape[0],):
            raise ValueError("delta_t_s must have shape [B]")
        if bool((delta_t_s <= 0).any()) or not torch.isfinite(delta_t_s).all():
            raise ValueError("delta_t_s must be finite and strictly positive")

        features = self.encoder.forward_features(events)
        endpoint_embeddings = self.encoder.pool_temporal_steps(features)
        raw_height_logits = self.height_head(endpoint_embeddings).squeeze(-1)
        log_min = math.log(self.config.min_height_px)
        log_max = math.log(self.config.max_height_px)
        # A sigmoid parameterization keeps heights positive and bounded without
        # the zero-gradient saturation introduced by hard clamping.
        log_heights = log_min + (log_max - log_min) * torch.sigmoid(raw_height_logits)
        heights = torch.exp(log_heights)
        log_ratio = log_heights[:, 0] - log_heights[:, 1]
        ratio = torch.exp(log_ratio.clamp(-12.0, 12.0))
        ttc = stable_ttc_from_log_ratio(
            log_ratio,
            delta_t_s,
            denominator_epsilon=self.config.denominator_epsilon,
            clip_seconds=self.config.ttc_clip_seconds,
        )
        first, second = endpoint_embeddings.unbind(dim=1)
        pair_embedding = self.pair_projector(torch.cat([first, second, second - first], dim=-1))
        return ObjectCentricLHROutput(
            log_visible_heights=log_heights,
            visible_heights_px=heights,
            log_height_ratio=log_ratio,
            height_ratio=ratio,
            ttc_mean_seconds=ttc,
            direction_logits=self.direction_head(pair_embedding),
            endpoint_embeddings=endpoint_embeddings,
            pair_embedding=pair_embedding,
            mask_logits=self._decode_masks(features),
            features=features,
        )

    def checkpoint_config(self) -> dict[str, Any]:
        """Return a JSON-safe architecture record."""

        return asdict(self.config)


__all__ = [
    "ObjectCentricLHR",
    "ObjectCentricLHRConfig",
    "ObjectCentricLHROutput",
    "stable_ttc_from_log_ratio",
]
