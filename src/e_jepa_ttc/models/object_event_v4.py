"""Event-required Object TTC v4 with common-coordinate temporal ROIs.

V4 removes the v3 shortcut in which the JEPA predictor and ordered score both
received box-motion embeddings.  The event branch here accepts only event
voxels and time.  A separate motion branch is fused late through a bounded gate,
so validation can measure each modality independently and training cannot drive
the event contribution to zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from e_jepa_ttc.data.event_v4_geometry import (
    EVENT_V4_CHANNEL_COUNT,
    EVENT_V4_STEPS,
)
from e_jepa_ttc.models.highres_factorized import (
    BACKBONE_STRUCTURAL_CONFIG_FIELDS,
    EJEPATubeletLHR,
    EJEPATubeletLHRConfig,
    HighResFeatures,
)

OBSERVABLE_MOTION_DIM = 18


@dataclass(frozen=True)
class ObjectEventV4Config:
    in_channels: int = EVENT_V4_CHANNEL_COUNT
    temporal_steps: int = EVENT_V4_STEPS
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
    predictor_hidden_dim: int = 256
    event_head_hidden_dim: int = 256
    motion_hidden_dim: int = 128
    fusion_hidden_dim: int = 128
    dropout: float = 0.1
    max_abs_expansion: float = 0.25
    ttc_clip_seconds: float = 60.0
    min_abs_expansion_for_ttc: float = 1.0e-4
    minimum_event_gate: float = 0.35
    maximum_event_gate: float = 0.90

    def __post_init__(self) -> None:
        integers = (
            self.in_channels,
            self.temporal_steps,
            self.embed_dim,
            self.patch_size,
            self.spatial_window,
            self.heads,
            self.predictor_hidden_dim,
            self.event_head_hidden_dim,
            self.motion_hidden_dim,
            self.fusion_hidden_dim,
        )
        if min(integers) <= 0:
            raise ValueError("Object Event v4 dimensions must be positive")
        if self.in_channels != EVENT_V4_CHANNEL_COUNT:
            raise ValueError(f"v4 requires exactly {EVENT_V4_CHANNEL_COUNT} active channels")
        if self.temporal_steps != EVENT_V4_STEPS:
            raise ValueError(f"v4 requires exactly {EVENT_V4_STEPS} causal steps")
        if self.embed_dim % self.heads:
            raise ValueError("embed_dim must be divisible by heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if not 0.0 < self.minimum_event_gate <= self.maximum_event_gate <= 1.0:
            raise ValueError("event-gate bounds must satisfy 0 < min <= max <= 1")
        if not 0.0 < self.max_abs_expansion < 1.0:
            raise ValueError("max_abs_expansion must lie in (0,1)")
        if self.ttc_clip_seconds <= 0.0 or self.min_abs_expansion_for_ttc <= 0.0:
            raise ValueError("TTC safety controls must be positive")

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
class ObjectEventV4Output:
    signed_expansion: torch.Tensor
    ttc_mean_seconds: torch.Tensor
    event_expansion: torch.Tensor
    event_ttc_seconds: torch.Tensor
    motion_expansion: torch.Tensor
    motion_ttc_seconds: torch.Tensor
    event_gate: torch.Tensor
    event_raw_forward: torch.Tensor
    event_raw_reverse: torch.Tensor
    reversal_error: torch.Tensor
    endpoint_embeddings: torch.Tensor
    reversed_endpoint_embeddings: torch.Tensor
    predicted_future_tokens: torch.Tensor
    target_future_tokens: torch.Tensor
    future_token_mask: torch.Tensor
    online_features: HighResFeatures


def safe_ttc_from_expansion(
    expansion: torch.Tensor,
    delta_t_s: torch.Tensor,
    *,
    minimum_abs_expansion: float,
    clip_seconds: float,
) -> torch.Tensor:
    if expansion.shape != delta_t_s.shape:
        raise ValueError("expansion and delta_t_s must share shape")
    sign = torch.where(expansion < 0.0, -torch.ones_like(expansion), torch.ones_like(expansion))
    denominator = sign * expansion.abs().clamp_min(minimum_abs_expansion)
    return (delta_t_s / denominator).clamp(-clip_seconds, clip_seconds)


def expansion_to_log_ratio(expansion: torch.Tensor, *, epsilon: float = 1.0e-6) -> torch.Tensor:
    return torch.log1p(-expansion.clamp(max=1.0 - epsilon))



class ObjectEventTTCV4(nn.Module):
    """Three-step event JEPA plus explicitly separated late motion fusion."""

    def __init__(self, config: ObjectEventV4Config | None = None) -> None:
        super().__init__()
        self.config = config or ObjectEventV4Config()
        self.encoder = EJEPATubeletLHR(self.config.backbone_config())
        for legacy_head in (self.encoder.ttc_head, self.encoder.collision_head):
            for parameter in legacy_head.parameters():
                parameter.requires_grad_(False)
        self.target_encoder = deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.target_encoder.eval()

        self.local_predictor = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim * 2),
            nn.Linear(self.config.embed_dim * 2, self.config.predictor_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.predictor_hidden_dim, self.config.embed_dim),
        )
        # This scorer is strictly event-only. Motion is intentionally absent.
        event_feature_dim = self.config.embed_dim * 7
        self.event_order_scorer = nn.Sequential(
            nn.LayerNorm(event_feature_dim),
            nn.Linear(event_feature_dim, self.config.event_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.event_head_hidden_dim, 1, bias=False),
        )
        self.motion_encoder = nn.Sequential(
            nn.LayerNorm(OBSERVABLE_MOTION_DIM),
            nn.Linear(OBSERVABLE_MOTION_DIM, self.config.motion_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.motion_hidden_dim, self.config.motion_hidden_dim),
            nn.GELU(),
        )
        self.motion_head = nn.Linear(self.config.motion_hidden_dim, 1, bias=False)
        self.event_gate_head = nn.Sequential(
            nn.LayerNorm(self.config.embed_dim * 3),
            nn.Linear(self.config.embed_dim * 3, self.config.fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.fusion_hidden_dim, 1),
        )

    def train(self, mode: bool = True) -> "ObjectEventTTCV4":
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def reset_target_encoder(self) -> None:
        self.target_encoder.load_state_dict(self.encoder.state_dict(), strict=True)
        self.target_encoder.eval()

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("EMA momentum must lie in [0,1)")
        for target, online in zip(
            self.target_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)
        for target, online in zip(
            self.target_encoder.buffers(), self.encoder.buffers(), strict=True
        ):
            target.copy_(online)

    def _validate_inputs(
        self,
        events: torch.Tensor,
        delta_t_s: torch.Tensor,
        observable_motion: torch.Tensor,
    ) -> None:
        expected = (
            events.shape[0],
            self.config.temporal_steps,
            self.config.in_channels,
        )
        if events.ndim != 5 or events.shape[:3] != expected:
            raise ValueError(
                f"events must have shape [B,{self.config.temporal_steps},"
                f"{self.config.in_channels},H,W], got {tuple(events.shape)}"
            )
        if events.shape[-1] != events.shape[-2]:
            raise ValueError("Object Event v4 requires square common ROIs")
        if delta_t_s.shape != (events.shape[0],):
            raise ValueError("delta_t_s must have shape [B]")
        if observable_motion.shape != (events.shape[0], OBSERVABLE_MOTION_DIM):
            raise ValueError(
                f"observable_motion must have shape [B,{OBSERVABLE_MOTION_DIM}]"
            )

    def _event_feature(self, endpoints: torch.Tensor) -> torch.Tensor:
        if endpoints.ndim != 3 or endpoints.shape[1] != EVENT_V4_STEPS:
            raise ValueError("endpoint embeddings must have shape [B,3,D]")
        z0, z1, z2 = endpoints.unbind(dim=1)
        return torch.cat(
            (
                z0,
                z1,
                z2,
                z1 - z0,
                z2 - z1,
                (z1 - z0).abs(),
                (z2 - z1).abs(),
            ),
            dim=-1,
        )

    def _raw_event_score(self, endpoints: torch.Tensor) -> torch.Tensor:
        return self.event_order_scorer(self._event_feature(endpoints)).squeeze(-1)

    def _encode_both_orders(
        self,
        events: torch.Tensor,
    ) -> tuple[HighResFeatures, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        online = self.encoder.forward_features(events)
        endpoints = self.encoder.pool_temporal_steps(online)
        reversed_events = torch.flip(events, dims=(1,))
        reversed_features = self.encoder.forward_features(reversed_events)
        reversed_endpoints = self.encoder.pool_temporal_steps(reversed_features)
        raw_forward = self._raw_event_score(endpoints)
        raw_reverse = self._raw_event_score(reversed_endpoints)
        return online, endpoints, reversed_endpoints, raw_forward, raw_reverse

    def _local_jepa(
        self,
        online: HighResFeatures,
        events: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if online.tokens.shape[1] != EVENT_V4_STEPS:
            raise ValueError("Online encoder did not preserve three temporal steps")
        context = torch.cat((online.tokens[:, 0], online.tokens[:, 1]), dim=-1)
        predicted = self.local_predictor(context)
        with torch.no_grad():
            target = self.target_encoder.forward_features(events)
            target_future = target.tokens[:, 2].detach()
            target_mask = target.valid_patch_mask[:, 2].detach()
        if predicted.shape != target_future.shape:
            raise RuntimeError(
                f"Local JEPA token mismatch {tuple(predicted.shape)} != {tuple(target_future.shape)}"
            )
        return predicted, target_future, target_mask

    def forward(
        self,
        events: torch.Tensor,
        delta_t_s: torch.Tensor,
        observable_motion: torch.Tensor | None = None,
    ) -> ObjectEventV4Output:
        if observable_motion is None:
            observable_motion = events.new_zeros((events.shape[0], OBSERVABLE_MOTION_DIM))
        self._validate_inputs(events, delta_t_s, observable_motion)
        online, endpoints, reversed_endpoints, raw_forward, raw_reverse = (
            self._encode_both_orders(events)
        )
        # Swapping the input exchanges the two raw terms, so this coordinate is
        # exactly odd even though the causal encoder itself is order-sensitive.
        antisymmetric_raw = 0.5 * (raw_forward - raw_reverse)
        event_expansion = self.config.max_abs_expansion * torch.tanh(antisymmetric_raw)

        motion_embedding = self.motion_encoder(observable_motion)
        motion_expansion = self.config.max_abs_expansion * torch.tanh(
            self.motion_head(motion_embedding).squeeze(-1)
        )
        z0, z1, z2 = endpoints.unbind(dim=1)
        gate_features = torch.cat((z2 - z1, (z2 - z1).abs(), z1 * z2), dim=-1)
        unit_gate = torch.sigmoid(self.event_gate_head(gate_features).squeeze(-1))
        event_gate = self.config.minimum_event_gate + (
            self.config.maximum_event_gate - self.config.minimum_event_gate
        ) * unit_gate
        fused = event_gate * event_expansion + (1.0 - event_gate) * motion_expansion
        fused = fused.clamp(-self.config.max_abs_expansion, self.config.max_abs_expansion)

        predicted_future, target_future, future_mask = self._local_jepa(online, events)
        reverse_check = self.config.max_abs_expansion * torch.tanh(
            0.5 * (raw_reverse - raw_forward)
        )
        reversal_error = (event_expansion + reverse_check).abs()
        return ObjectEventV4Output(
            signed_expansion=fused,
            ttc_mean_seconds=safe_ttc_from_expansion(
                fused,
                delta_t_s,
                minimum_abs_expansion=self.config.min_abs_expansion_for_ttc,
                clip_seconds=self.config.ttc_clip_seconds,
            ),
            event_expansion=event_expansion,
            event_ttc_seconds=safe_ttc_from_expansion(
                event_expansion,
                delta_t_s,
                minimum_abs_expansion=self.config.min_abs_expansion_for_ttc,
                clip_seconds=self.config.ttc_clip_seconds,
            ),
            motion_expansion=motion_expansion,
            motion_ttc_seconds=safe_ttc_from_expansion(
                motion_expansion,
                delta_t_s,
                minimum_abs_expansion=self.config.min_abs_expansion_for_ttc,
                clip_seconds=self.config.ttc_clip_seconds,
            ),
            event_gate=event_gate,
            event_raw_forward=raw_forward,
            event_raw_reverse=raw_reverse,
            reversal_error=reversal_error,
            endpoint_embeddings=endpoints,
            reversed_endpoint_embeddings=reversed_endpoints,
            predicted_future_tokens=predicted_future,
            target_future_tokens=target_future,
            future_token_mask=future_mask,
            online_features=online,
        )

    def event_only_forward(
        self,
        events: torch.Tensor,
        delta_t_s: torch.Tensor,
    ) -> ObjectEventV4Output:
        return self(events, delta_t_s, events.new_zeros((events.shape[0], OBSERVABLE_MOTION_DIM)))

    def load_adapted_pretrained_backbone(
        self,
        source_state: Mapping[str, Any],
        source_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Load a 21-channel Level encoder by slicing its active first 12 channels.

        Every structural field except ``in_channels`` must match exactly.  Only
        ``patch_embed.weight`` is adapted; all other backbone keys and shapes are
        fail-closed.
        """

        expected_config = self.encoder.backbone_structural_config()
        mismatches = {}
        for key in BACKBONE_STRUCTURAL_CONFIG_FIELDS:
            if key == "in_channels":
                continue
            if source_config.get(key) != expected_config[key]:
                mismatches[key] = {
                    "expected": expected_config[key],
                    "received": source_config.get(key),
                }
        if int(source_config.get("in_channels", -1)) < self.config.in_channels:
            mismatches["in_channels"] = {
                "expected_at_least": self.config.in_channels,
                "received": source_config.get("in_channels"),
            }
        if mismatches:
            raise ValueError(f"Adapted backbone structural mismatch: {mismatches}")

        expected_state = self.encoder.backbone_state_dict()
        received_keys = set(source_state)
        expected_keys = set(expected_state)
        if received_keys != expected_keys:
            raise ValueError(
                "Adapted backbone keys must match exactly; "
                f"missing={sorted(expected_keys - received_keys)}, "
                f"extra={sorted(received_keys - expected_keys)}"
            )
        adapted: dict[str, torch.Tensor] = {}
        for key, expected in expected_state.items():
            value = source_state[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Backbone value is not a tensor: {key}")
            if key == "patch_embed.weight":
                if value.ndim != 4 or value.shape[1] < self.config.in_channels:
                    raise ValueError(
                        f"Cannot adapt patch_embed.weight {tuple(value.shape)} to "
                        f"{self.config.in_channels} channels"
                    )
                value = value[:, : self.config.in_channels].contiguous()
            if value.shape != expected.shape:
                raise ValueError(
                    f"Backbone shape mismatch for {key}: {tuple(value.shape)} != "
                    f"{tuple(expected.shape)}"
                )
            adapted[key] = value
        result = self.encoder.load_state_dict(adapted, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"Unexpected adapted keys: {result.unexpected_keys}")
        self.reset_target_encoder()
        return {
            "source_in_channels": int(source_config["in_channels"]),
            "target_in_channels": self.config.in_channels,
            "patch_embed_adaptation": "slice_first_active_channels",
            "structural_config": expected_config,
            "transferred_keys": sorted(adapted),
            "key_count": len(adapted),
            "missing_non_backbone_keys": sorted(result.missing_keys),
        }


__all__ = [
    "OBSERVABLE_MOTION_DIM",
    "ObjectEventTTCV4",
    "ObjectEventV4Config",
    "ObjectEventV4Output",
    "expansion_to_log_ratio",
    "safe_ttc_from_expansion",
]
