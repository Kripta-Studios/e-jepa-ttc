"""Geometry-conditioned JEPA for continuous signed TTC expansion.

Object-expansion v2 separated TTC into a binary sign and a positive magnitude.
The validation audit showed that this decomposition was fragile under the
sequence-prior shift: a threshold that recovered negative TTC destroyed the
positive buckets.  This v3 candidate instead predicts the continuous,
dimensionless expansion coordinate

    g = delta_t / TTC.

The sign is therefore part of the regressed physical variable and no inference
threshold is required.  A strictly observable box-motion prior is corrected by
an antisymmetric event residual.  The JEPA encoder is also kept active through a
latent endpoint predictor, rather than being used only as initialization.
"""

from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.highres_factorized import (
    EJEPATubeletLHR,
    EJEPATubeletLHRConfig,
    HighResFeatures,
)

# Cache contract from garlttc_lhr_cache.py.  These are observable 2-D quantities;
# TTC, depth, box3d and category are deliberately absent.
OBSERVABLE_MOTION_DIM = 18
LOG_HEIGHT_RATE_INDEX = 7
SIGNED_MOTION_INDICES = (4, 5, 6, 7, 15, 16)


@dataclass(frozen=True)
class ObjectSignedExpansionConfig:
    """Architecture and bounded signed-expansion controls."""

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
    motion_dim: int = OBSERVABLE_MOTION_DIM
    adapter_hidden_dim: int = 192
    motion_hidden_dim: int = 128
    predictor_hidden_dim: int = 256
    pair_hidden_dim: int = 256
    dropout: float = 0.1
    max_abs_expansion: float = 0.25
    ttc_clip_seconds: float = 60.0
    min_abs_expansion_for_ttc: float = 1.0e-4
    log_ratio_epsilon: float = 1.0e-6
    box_log_rate_scale: float = 5.0

    def __post_init__(self) -> None:
        integers = (
            self.in_channels,
            self.embed_dim,
            self.patch_size,
            self.spatial_window,
            self.heads,
            self.query_count,
            self.motion_dim,
            self.adapter_hidden_dim,
            self.motion_hidden_dim,
            self.predictor_hidden_dim,
            self.pair_hidden_dim,
        )
        if min(integers) <= 0:
            raise ValueError("Signed-expansion dimensions must be positive")
        if self.motion_dim != OBSERVABLE_MOTION_DIM:
            raise ValueError(
                f"motion_dim must match the cache contract ({OBSERVABLE_MOTION_DIM})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if not 0.0 < self.max_abs_expansion < 1.0:
            raise ValueError("max_abs_expansion must lie in (0,1)")
        if self.ttc_clip_seconds <= 0.0:
            raise ValueError("ttc_clip_seconds must be positive")
        if not 0.0 < self.min_abs_expansion_for_ttc < self.max_abs_expansion:
            raise ValueError("min_abs_expansion_for_ttc is outside the valid range")
        if not 0.0 < self.log_ratio_epsilon < 1.0:
            raise ValueError("log_ratio_epsilon must lie in (0,1)")
        if self.box_log_rate_scale <= 0.0:
            raise ValueError("box_log_rate_scale must be positive")

    def backbone_config(self) -> EJEPATubeletLHRConfig:
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
class ObjectSignedExpansionOutput:
    """Continuous physical prediction plus JEPA/geometry diagnostics."""

    signed_expansion: torch.Tensor
    signed_inverse_ttc: torch.Tensor
    ttc_mean_seconds: torch.Tensor
    log_height_ratio: torch.Tensor
    geometry_prior_expansion: torch.Tensor
    geometry_prior_log_ratio: torch.Tensor
    learned_residual_expansion: torch.Tensor
    ordered_score_forward: torch.Tensor
    ordered_score_reverse: torch.Tensor
    endpoint_embeddings: torch.Tensor
    activity_endpoint_embeddings: torch.Tensor
    adapted_endpoint_embeddings: torch.Tensor
    predicted_second_embedding: torch.Tensor
    predicted_first_embedding: torch.Tensor
    target_endpoint_embeddings: torch.Tensor
    pair_embedding: torch.Tensor
    activity_logits: torch.Tensor
    activity_targets: torch.Tensor
    features: HighResFeatures


def reverse_observable_motion(motion: torch.Tensor) -> torch.Tensor:
    """Return an involutive time-reversal approximation for cached motion.

    Endpoint-static quantities remain unchanged because the lightweight cache
    stores only the second endpoint geometry.  Every signed derivative is
    negated.  Applying this function twice returns the original tensor exactly.
    """

    if motion.ndim != 2 or motion.shape[1] != OBSERVABLE_MOTION_DIM:
        raise ValueError(
            f"motion must have shape [B,{OBSERVABLE_MOTION_DIM}]"
        )
    reversed_motion = motion.clone()
    reversed_motion[:, list(SIGNED_MOTION_INDICES)] *= -1.0
    return reversed_motion


def antisymmetric_pair_score(
    forward_score: torch.Tensor,
    reverse_score: torch.Tensor,
) -> torch.Tensor:
    """Project two ordered scores onto their antisymmetric component."""

    if forward_score.shape != reverse_score.shape:
        raise ValueError("forward and reverse scores must share shape")
    return 0.5 * (forward_score - reverse_score)


def expansion_to_log_ratio(
    expansion: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Map ``g = dt/TTC`` to the Garl visible-height log-ratio."""

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie in (0,1)")
    return torch.log1p(-expansion.clamp(max=1.0 - epsilon))


def safe_ttc_from_expansion(
    expansion: torch.Tensor,
    delta_t_s: torch.Tensor,
    *,
    minimum_abs_expansion: float,
    clip_seconds: float,
) -> torch.Tensor:
    """Derive finite signed TTC without altering the learned expansion."""

    if expansion.shape != delta_t_s.shape:
        raise ValueError("expansion and delta_t_s must share shape")
    if minimum_abs_expansion <= 0.0 or clip_seconds <= 0.0:
        raise ValueError("safe TTC controls must be positive")
    sign = torch.where(
        expansion < 0.0,
        -torch.ones_like(expansion),
        torch.ones_like(expansion),
    )
    denominator = sign * expansion.abs().clamp_min(minimum_abs_expansion)
    return (delta_t_s / denominator).clamp(-clip_seconds, clip_seconds)


def geometry_prior_from_motion(
    observable_motion: torch.Tensor,
    delta_t_s: torch.Tensor,
    *,
    log_rate_scale: float,
    max_abs_expansion: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a causal box-height LHR prior from the cache motion contract.

    The cache stores ``log_height_rate / 5`` at index 7, where the unscaled
    rate is ``(log h2 - log h1) / dt``.  Hence

        log(h1 / h2) = -rate * dt,
        g_box = 1 - exp(log(h1 / h2)).

    This uses only the same observed boxes already required to define the ROI.
    """

    if (
        observable_motion.ndim != 2
        or observable_motion.shape[1] != OBSERVABLE_MOTION_DIM
    ):
        raise ValueError("observable_motion has an invalid shape")
    if delta_t_s.shape != (observable_motion.shape[0],):
        raise ValueError("delta_t_s must have shape [B]")
    log_rate = observable_motion[:, LOG_HEIGHT_RATE_INDEX] * log_rate_scale
    log_ratio = -log_rate * delta_t_s
    expansion = 1.0 - torch.exp(log_ratio.clamp(min=-4.0, max=4.0))
    return expansion.clamp(-max_abs_expansion, max_abs_expansion), log_ratio


class ObjectCentricSignedExpansionTTC(nn.Module):
    """JEPA ROI encoder with continuous geometry-conditioned expansion."""

    def __init__(self, config: ObjectSignedExpansionConfig | None = None) -> None:
        super().__init__()
        self.config = config or ObjectSignedExpansionConfig()
        self.encoder = EJEPATubeletLHR(self.config.backbone_config())
        for legacy_head in (self.encoder.ttc_head, self.encoder.collision_head):
            for parameter in legacy_head.parameters():
                parameter.requires_grad_(False)
        self.target_encoder = deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()

        self.endpoint_adapter = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim),
            nn.Linear(self.config.embed_dim, self.config.adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.adapter_hidden_dim, self.config.embed_dim),
        )
        self.activity_fusion = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim * 2),
            nn.Linear(self.config.embed_dim * 2, self.config.embed_dim),
            nn.GELU(),
        )
        self.motion_encoder = nn.Sequential(
            nn.LayerNorm(self.config.motion_dim * 2 + 2),
            nn.Linear(self.config.motion_dim * 2 + 2, self.config.motion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.motion_hidden_dim, self.config.embed_dim),
        )
        predictor_input = self.config.embed_dim * 2 + 2
        self.latent_predictor = nn.Sequential(
            nn.LayerNorm(predictor_input),
            nn.Linear(predictor_input, self.config.predictor_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.predictor_hidden_dim, self.config.embed_dim),
        )
        ordered_input = self.config.embed_dim * 7 + 2
        self.ordered_projector = nn.Sequential(
            nn.LayerNorm(ordered_input),
            nn.Linear(ordered_input, self.config.pair_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.ordered_score = nn.Linear(self.config.pair_hidden_dim, 1, bias=False)
        self.geometry_residual = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim),
            nn.Linear(self.config.embed_dim, self.config.motion_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.motion_hidden_dim, 1, bias=False),
        )
        self.activity_head = nn.Linear(self.config.embed_dim, 1)
        self._initialize_new_modules()

    def _initialize_new_modules(self) -> None:
        adapter_last = self.endpoint_adapter[-1]
        if not isinstance(adapter_last, nn.Linear):
            raise TypeError("endpoint_adapter must end with Linear")
        nn.init.zeros_(adapter_last.weight)
        nn.init.zeros_(adapter_last.bias)
        nn.init.zeros_(self.ordered_score.weight)
        geometry_last = self.geometry_residual[-1]
        if not isinstance(geometry_last, nn.Linear):
            raise TypeError("geometry_residual must end with Linear")
        nn.init.zeros_(geometry_last.weight)
        nn.init.zeros_(self.activity_head.bias)

    def load_exact_backbone_state_dict(
        self,
        source_state: Mapping[str, Any],
        source_config: EJEPATubeletLHRConfig | Mapping[str, Any],
    ) -> dict[str, Any]:
        return self.encoder.load_exact_backbone_state_dict(source_state, source_config)

    def backbone_structural_config(self) -> dict[str, Any]:
        return self.encoder.backbone_structural_config()

    @torch.no_grad()
    def reset_target_encoder(self) -> None:
        """Synchronize the EMA target encoder with the online encoder."""

        self.target_encoder.load_state_dict(self.encoder.state_dict(), strict=True)
        self.target_encoder.eval()

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Apply one exponential-moving-average target update."""

        if not 0.0 <= momentum < 1.0:
            raise ValueError("target EMA momentum must lie in [0,1)")
        for target, online in zip(
            self.target_encoder.parameters(),
            self.encoder.parameters(),
            strict=True,
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        for target, online in zip(
            self.target_encoder.buffers(),
            self.encoder.buffers(),
            strict=True,
        ):
            target.copy_(online)
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> ObjectCentricSignedExpansionTTC:
        super().train(mode)
        self.target_encoder.eval()
        return self

    @staticmethod
    def _dt_features(delta_t_s: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                delta_t_s,
                torch.log(delta_t_s.clamp_min(1.0e-6)),
            ),
            dim=-1,
        )

    def _activity_pool(
        self,
        features: HighResFeatures,
        events: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, _, height, width = events.shape
        activity = events.abs().mean(dim=2, keepdim=True)
        pooled_activity = functional.adaptive_avg_pool2d(
            activity.reshape(batch * steps, 1, height, width),
            (features.encoded_grid_height, features.encoded_grid_width),
        ).reshape(batch, steps, -1)
        valid = features.valid_patch_mask
        logits = torch.log1p(pooled_activity).masked_fill(~valid, -1.0e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        embeddings = torch.sum(features.tokens * weights.unsqueeze(-1), dim=2)
        max_value = pooled_activity.amax(dim=-1, keepdim=True).clamp_min(1.0e-6)
        targets = (pooled_activity / max_value).clamp(0.0, 1.0)
        activity_logits = self.activity_head(features.tokens).squeeze(-1)
        return embeddings, activity_logits, targets

    def _predict_endpoint(
        self,
        source: torch.Tensor,
        motion_embedding: torch.Tensor,
        dt_features: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.latent_predictor(
            torch.cat((source, motion_embedding, dt_features), dim=-1)
        )
        return source + residual

    def _ordered_pair_score(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        predicted_second: torch.Tensor,
        motion_embedding: torch.Tensor,
        dt_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        difference = second - first
        prediction_error = second - predicted_second
        value = torch.cat(
            (
                first,
                second,
                difference,
                difference.abs(),
                first * second,
                prediction_error,
                motion_embedding,
                dt_features,
            ),
            dim=-1,
        )
        embedding = self.ordered_projector(value)
        score = self.ordered_score(embedding).squeeze(-1)
        return score, embedding

    def forward(
        self,
        events: torch.Tensor,
        delta_t_s: torch.Tensor,
        observable_motion: torch.Tensor,
        jepa_context_motion: torch.Tensor,
        precontext_motion_valid: torch.Tensor,
    ) -> ObjectSignedExpansionOutput:
        if events.ndim != 5 or events.shape[1] != 2:
            raise ValueError("events must have shape [B,2,C,H,W]")
        if events.shape[2] != self.config.in_channels:
            raise ValueError("events channel count does not match the JEPA backbone")
        batch = events.shape[0]
        if delta_t_s.shape != (batch,) or bool((delta_t_s <= 0).any()):
            raise ValueError("delta_t_s must be positive with shape [B]")
        if observable_motion.shape != (batch, self.config.motion_dim):
            raise ValueError("observable_motion has an invalid shape")
        if jepa_context_motion.shape != (batch, self.config.motion_dim):
            raise ValueError("jepa_context_motion has an invalid shape")
        if precontext_motion_valid.shape != (batch,):
            raise ValueError("precontext_motion_valid must have shape [B]")

        features = self.encoder.forward_features(events)
        query_endpoints = self.encoder.pool_temporal_steps(features)
        activity_endpoints, activity_logits, activity_targets = self._activity_pool(
            features, events
        )
        with torch.no_grad():
            target_features = self.target_encoder.forward_features(events)
            target_endpoints, _, _ = self._activity_pool(target_features, events)
        endpoints = self.activity_fusion(
            torch.cat((query_endpoints, activity_endpoints), dim=-1)
        )
        adapted = endpoints + self.endpoint_adapter(endpoints)
        dt_features = self._dt_features(delta_t_s)

        context_valid = precontext_motion_valid.to(observable_motion.dtype).unsqueeze(-1)
        context_motion = jepa_context_motion * context_valid
        motion_embedding = self.motion_encoder(
            torch.cat(
                (
                    observable_motion,
                    context_motion,
                    context_valid,
                    delta_t_s.unsqueeze(-1),
                ),
                dim=-1,
            )
        )
        reverse_motion = reverse_observable_motion(observable_motion)
        reverse_context = reverse_observable_motion(context_motion)
        reverse_motion_embedding = self.motion_encoder(
            torch.cat(
                (
                    reverse_motion,
                    reverse_context,
                    context_valid,
                    delta_t_s.unsqueeze(-1),
                ),
                dim=-1,
            )
        )

        first, second = adapted[:, 0], adapted[:, 1]
        predicted_second = self._predict_endpoint(
            first, motion_embedding, dt_features
        )
        predicted_first = self._predict_endpoint(
            second, reverse_motion_embedding, dt_features
        )
        forward_score, forward_pair = self._ordered_pair_score(
            first,
            second,
            predicted_second,
            motion_embedding,
            dt_features,
        )
        reverse_score, reverse_pair = self._ordered_pair_score(
            second,
            first,
            predicted_first,
            reverse_motion_embedding,
            dt_features,
        )
        antisymmetric_event_score = antisymmetric_pair_score(forward_score, reverse_score)
        geometry_score = self.geometry_residual(motion_embedding).squeeze(-1)

        prior_expansion, prior_log_ratio = geometry_prior_from_motion(
            observable_motion,
            delta_t_s,
            log_rate_scale=self.config.box_log_rate_scale,
            max_abs_expansion=self.config.max_abs_expansion,
        )
        normalized_prior = (
            prior_expansion / self.config.max_abs_expansion
        ).clamp(-0.999, 0.999)
        prior_raw = torch.atanh(normalized_prior)
        learned_raw = antisymmetric_event_score + geometry_score
        signed_expansion = self.config.max_abs_expansion * torch.tanh(
            prior_raw + learned_raw
        )
        learned_residual = signed_expansion - prior_expansion
        signed_inverse = signed_expansion / delta_t_s
        ttc = safe_ttc_from_expansion(
            signed_expansion,
            delta_t_s,
            minimum_abs_expansion=self.config.min_abs_expansion_for_ttc,
            clip_seconds=self.config.ttc_clip_seconds,
        )
        log_ratio = expansion_to_log_ratio(
            signed_expansion,
            epsilon=self.config.log_ratio_epsilon,
        )
        pair_embedding = 0.5 * (forward_pair + reverse_pair)
        return ObjectSignedExpansionOutput(
            signed_expansion=signed_expansion,
            signed_inverse_ttc=signed_inverse,
            ttc_mean_seconds=ttc,
            log_height_ratio=log_ratio,
            geometry_prior_expansion=prior_expansion,
            geometry_prior_log_ratio=prior_log_ratio,
            learned_residual_expansion=learned_residual,
            ordered_score_forward=forward_score,
            ordered_score_reverse=reverse_score,
            endpoint_embeddings=endpoints,
            activity_endpoint_embeddings=activity_endpoints,
            adapted_endpoint_embeddings=adapted,
            predicted_second_embedding=predicted_second,
            predicted_first_embedding=predicted_first,
            target_endpoint_embeddings=target_endpoints,
            pair_embedding=pair_embedding,
            activity_logits=activity_logits,
            activity_targets=activity_targets,
            features=features,
        )


__all__ = [
    "LOG_HEIGHT_RATE_INDEX",
    "OBSERVABLE_MOTION_DIM",
    "antisymmetric_pair_score",
    "ObjectCentricSignedExpansionTTC",
    "ObjectSignedExpansionConfig",
    "ObjectSignedExpansionOutput",
    "expansion_to_log_ratio",
    "geometry_prior_from_motion",
    "reverse_observable_motion",
    "safe_ttc_from_expansion",
]
