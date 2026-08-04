"""Stable object-centric TTC prediction in inverse-time coordinates.

The historical Object-LHR v1 head predicts two absolute visible heights and
then divides by ``1 - h1 / h2``.  That route is preserved for reproducibility.
This v2 candidate instead predicts:

* ``log |1 / TTC|`` -- a bounded magnitude that cannot cross the singularity;
* the TTC sign with a dedicated two-class head;
* the implied LHR log-ratio ``log(1 - delta_t / TTC)`` as a differentiable
  geometry consistency signal.

Bounding boxes and TTC labels never enter the encoder.  The cache still uses
official boxes offline to construct one ROI per tracked object.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from e_jepa_ttc.models.highres_factorized import (
    EJEPATubeletLHR,
    EJEPATubeletLHRConfig,
    HighResFeatures,
)


@dataclass(frozen=True)
class ObjectExpansionConfig:
    """Architecture and stable inverse-TTC parameterization controls."""

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
    adapter_hidden_dim: int = 192
    pair_hidden_dim: int = 192
    dropout: float = 0.1
    min_abs_inverse_ttc: float = 1.0 / 60.0
    max_abs_inverse_ttc: float = 2.0
    initial_abs_inverse_ttc: float = 1.0 / 3.0
    initial_negative_probability: float = 0.2
    ttc_clip_seconds: float = 60.0
    log_ratio_epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        integers = (
            self.in_channels,
            self.embed_dim,
            self.patch_size,
            self.spatial_window,
            self.heads,
            self.adapter_hidden_dim,
            self.pair_hidden_dim,
            self.query_count,
        )
        if min(integers) <= 0:
            raise ValueError("Object-expansion dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if not 0.0 < self.min_abs_inverse_ttc < self.max_abs_inverse_ttc:
            raise ValueError(
                "Inverse-TTC bounds must satisfy 0 < min_abs < max_abs"
            )
        if not (
            self.min_abs_inverse_ttc
            <= self.initial_abs_inverse_ttc
            <= self.max_abs_inverse_ttc
        ):
            raise ValueError("initial_abs_inverse_ttc must lie inside the bounds")
        if not 0.0 < self.initial_negative_probability < 1.0:
            raise ValueError("initial_negative_probability must lie in (0,1)")
        if self.ttc_clip_seconds <= 0.0 or not 0.0 < self.log_ratio_epsilon < 1.0:
            raise ValueError("TTC clipping and log-ratio epsilon are invalid")

    def backbone_config(self) -> EJEPATubeletLHRConfig:
        """Return the exact JEPA encoder identity required for transfer."""

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
class ObjectExpansionOutput:
    """Stable TTC outputs and representation diagnostics."""

    log_abs_inverse_ttc: torch.Tensor
    abs_inverse_ttc: torch.Tensor
    direction_logits: torch.Tensor
    negative_probability: torch.Tensor
    signed_inverse_ttc_soft: torch.Tensor
    signed_inverse_ttc_hard: torch.Tensor
    log_height_ratio_soft: torch.Tensor
    ttc_soft_seconds: torch.Tensor
    ttc_mean_seconds: torch.Tensor
    endpoint_embeddings: torch.Tensor
    adapted_endpoint_embeddings: torch.Tensor
    pair_embedding: torch.Tensor
    features: HighResFeatures


def bounded_log_magnitude(
    raw: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
) -> torch.Tensor:
    """Map unconstrained logits to a bounded log-positive magnitude."""

    if not 0.0 < minimum < maximum:
        raise ValueError("Magnitude bounds must satisfy 0 < minimum < maximum")
    log_min = math.log(minimum)
    log_max = math.log(maximum)
    return log_min + (log_max - log_min) * torch.sigmoid(raw)


def soft_direction_sign(direction_logits: torch.Tensor) -> torch.Tensor:
    """Return E[sign] where class 0 is positive TTC and class 1 is negative."""

    if direction_logits.ndim != 2 or direction_logits.shape[1] != 2:
        raise ValueError("direction_logits must have shape [B,2]")
    negative_probability = torch.softmax(direction_logits, dim=-1)[:, 1]
    return 1.0 - 2.0 * negative_probability


def hard_direction_sign(direction_logits: torch.Tensor) -> torch.Tensor:
    """Return the discrete TTC sign using the direction argmax."""

    if direction_logits.ndim != 2 or direction_logits.shape[1] != 2:
        raise ValueError("direction_logits must have shape [B,2]")
    negative = direction_logits.argmax(dim=-1) == 1
    return torch.where(
        negative,
        -torch.ones_like(negative, dtype=direction_logits.dtype),
        torch.ones_like(negative, dtype=direction_logits.dtype),
    )


def log_ratio_from_signed_inverse_ttc(
    signed_inverse_ttc: torch.Tensor,
    delta_t_s: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Compute ``log(1 - delta_t / TTC)`` in a stable differentiable form."""

    if signed_inverse_ttc.shape != delta_t_s.shape:
        raise ValueError("signed inverse TTC and delta_t_s must share shape")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0,1)")
    normalized_expansion = delta_t_s * signed_inverse_ttc
    normalized_expansion = normalized_expansion.clamp(max=1.0 - epsilon)
    return torch.log1p(-normalized_expansion)


class ObjectCentricExpansionTTC(nn.Module):
    """JEPA-compatible object ROI encoder with stable inverse-TTC heads."""

    def __init__(self, config: ObjectExpansionConfig | None = None) -> None:
        super().__init__()
        self.config = config or ObjectExpansionConfig()
        self.encoder = EJEPATubeletLHR(self.config.backbone_config())
        for legacy_head in (self.encoder.ttc_head, self.encoder.collision_head):
            for parameter in legacy_head.parameters():
                parameter.requires_grad_(False)

        self.endpoint_adapter = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim),
            nn.Linear(self.config.embed_dim, self.config.adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.adapter_hidden_dim, self.config.embed_dim),
        )
        pair_input_dim = self.config.embed_dim * 5
        self.pair_projector = nn.Sequential(
            nn.LayerNorm(pair_input_dim),
            nn.Linear(pair_input_dim, self.config.pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.log_magnitude_head = nn.Linear(self.config.pair_hidden_dim, 1)
        self.direction_head = nn.Linear(self.config.pair_hidden_dim, 2)
        self._initialize_new_modules()

    def _initialize_new_modules(self) -> None:
        adapter_last = self.endpoint_adapter[-1]
        if not isinstance(adapter_last, nn.Linear):
            raise TypeError("endpoint_adapter must end with Linear")
        nn.init.zeros_(adapter_last.weight)
        nn.init.zeros_(adapter_last.bias)

        nn.init.normal_(self.log_magnitude_head.weight, mean=0.0, std=1.0e-3)
        log_min = math.log(self.config.min_abs_inverse_ttc)
        log_max = math.log(self.config.max_abs_inverse_ttc)
        target = math.log(self.config.initial_abs_inverse_ttc)
        fraction = (target - log_min) / (log_max - log_min)
        fraction = min(max(fraction, 1.0e-6), 1.0 - 1.0e-6)
        nn.init.constant_(
            self.log_magnitude_head.bias,
            math.log(fraction / (1.0 - fraction)),
        )

        nn.init.normal_(self.direction_head.weight, mean=0.0, std=1.0e-3)
        probability = self.config.initial_negative_probability
        with torch.no_grad():
            self.direction_head.bias[0] = 0.0
            self.direction_head.bias[1] = math.log(probability / (1.0 - probability))

    def load_exact_backbone_state_dict(
        self,
        source_state: Mapping[str, Any],
        source_config: EJEPATubeletLHRConfig | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Load only an exact label-free JEPA encoder state."""

        return self.encoder.load_exact_backbone_state_dict(
            source_state,
            source_config,
        )

    def backbone_structural_config(self) -> dict[str, Any]:
        """Return the exact architecture identity required by SSL transfer."""

        return self.encoder.backbone_structural_config()

    def forward(
        self,
        events: torch.Tensor,
        delta_t_s: torch.Tensor,
    ) -> ObjectExpansionOutput:
        """Predict stable TTC sign and inverse-time magnitude from two ROIs."""

        if events.ndim != 5 or events.shape[1] != 2:
            raise ValueError("events must have shape [B,2,C,H,W]")
        if events.shape[2] != self.config.in_channels:
            raise ValueError(
                f"events have {events.shape[2]} channels; "
                f"expected {self.config.in_channels}"
            )
        if delta_t_s.shape != (events.shape[0],):
            raise ValueError("delta_t_s must have shape [B]")
        if bool((delta_t_s <= 0).any()) or not torch.isfinite(delta_t_s).all():
            raise ValueError("delta_t_s must be finite and strictly positive")

        features = self.encoder.forward_features(events)
        endpoint_embeddings = self.encoder.pool_temporal_steps(features)
        adapted = endpoint_embeddings + self.endpoint_adapter(endpoint_embeddings)
        first, second = adapted.unbind(dim=1)
        difference = second - first
        pair_features = torch.cat(
            [
                first,
                second,
                difference,
                difference.abs(),
                first * second,
            ],
            dim=-1,
        )
        pair_embedding = self.pair_projector(pair_features)
        raw_log_magnitude = self.log_magnitude_head(pair_embedding).squeeze(-1)
        log_abs_inverse_ttc = bounded_log_magnitude(
            raw_log_magnitude,
            minimum=self.config.min_abs_inverse_ttc,
            maximum=self.config.max_abs_inverse_ttc,
        )
        abs_inverse_ttc = torch.exp(log_abs_inverse_ttc)
        direction_logits = self.direction_head(pair_embedding)
        negative_probability = torch.softmax(direction_logits, dim=-1)[:, 1]
        soft_sign = 1.0 - 2.0 * negative_probability
        hard_sign = hard_direction_sign(direction_logits)
        signed_inverse_soft = soft_sign * abs_inverse_ttc
        signed_inverse_hard = hard_sign * abs_inverse_ttc
        log_ratio_soft = log_ratio_from_signed_inverse_ttc(
            signed_inverse_soft,
            delta_t_s,
            epsilon=self.config.log_ratio_epsilon,
        )
        ttc_soft = (soft_sign / abs_inverse_ttc).clamp(
            -self.config.ttc_clip_seconds,
            self.config.ttc_clip_seconds,
        )
        ttc_hard = (hard_sign / abs_inverse_ttc).clamp(
            -self.config.ttc_clip_seconds,
            self.config.ttc_clip_seconds,
        )
        return ObjectExpansionOutput(
            log_abs_inverse_ttc=log_abs_inverse_ttc,
            abs_inverse_ttc=abs_inverse_ttc,
            direction_logits=direction_logits,
            negative_probability=negative_probability,
            signed_inverse_ttc_soft=signed_inverse_soft,
            signed_inverse_ttc_hard=signed_inverse_hard,
            log_height_ratio_soft=log_ratio_soft,
            ttc_soft_seconds=ttc_soft,
            ttc_mean_seconds=ttc_hard,
            endpoint_embeddings=endpoint_embeddings,
            adapted_endpoint_embeddings=adapted,
            pair_embedding=pair_embedding,
            features=features,
        )


__all__ = [
    "ObjectCentricExpansionTTC",
    "ObjectExpansionConfig",
    "ObjectExpansionOutput",
    "bounded_log_magnitude",
    "hard_direction_sign",
    "log_ratio_from_signed_inverse_ttc",
    "soft_direction_sign",
]
