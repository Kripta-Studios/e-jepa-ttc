"""Foreground/height-bypassing endpoint features for E-Clock X0."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from e_jepa_ttc.models.collision_clock_motion import GLOBAL_TRANSPORT_FEATURE_NAMES


@dataclass(frozen=True)
class HeightBypassEncoderConfig:
    in_channels: int = 12
    hidden_dim: int = 64
    token_dim: int = 128
    residual_depth: int = 2
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if min(self.in_channels, self.hidden_dim, self.token_dim, self.residual_depth) <= 0:
            raise ValueError("height-bypass encoder dimensions and depth must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")


class _TrunkResidualBlock(nn.Module):
    """Topology-equivalent copy of the sanctioned A5 trunk residual block."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = math.gcd(channels, 8)
        self.network = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class HeightBypassEndpointEncoder(nn.Module):
    """A5-equivalent raw trunk/token with no registered foreground path."""

    def __init__(self, config: HeightBypassEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeightBypassEncoderConfig()
        half = max(self.config.hidden_dim // 2, 8)
        self.features = nn.Sequential(
            nn.Conv2d(
                self.config.in_channels,
                half,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(math.gcd(half, 8), half),
            nn.GELU(),
            nn.Conv2d(
                half,
                self.config.hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(math.gcd(self.config.hidden_dim, 8), self.config.hidden_dim),
            nn.GELU(),
            *[
                _TrunkResidualBlock(self.config.hidden_dim, self.config.dropout)
                for _ in range(self.config.residual_depth)
            ],
        )
        self.token = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.config.hidden_dim, self.config.token_dim),
            nn.LayerNorm(self.config.token_dim),
        )
        assert_height_bypass_module_tree(self)

    @classmethod
    def from_sanctioned_modules(
        cls,
        *,
        features: nn.Module,
        token: nn.Module,
        config: HeightBypassEncoderConfig,
    ) -> HeightBypassEndpointEncoder:
        """Deep-copy only the sanctioned A5 modules into a clean container."""

        instance = cls(config)
        instance.features = copy.deepcopy(features)
        instance.token = copy.deepcopy(token)
        assert_height_bypass_module_tree(instance)
        return instance

    @classmethod
    def from_causal_scale_topology(
        cls,
        source: nn.Module,
        *,
        config: HeightBypassEncoderConfig,
    ) -> HeightBypassEndpointEncoder:
        """Extract only ``encoder.features`` and ``encoder.token`` from an A5 topology."""

        encoder = getattr(source, "encoder", None)
        features = getattr(encoder, "features", None)
        token = getattr(encoder, "token", None)
        if not isinstance(features, nn.Module) or not isinstance(token, nn.Module):
            raise TypeError("source does not expose the sanctioned A5 encoder modules")
        return cls.from_sanctioned_modules(features=features, token=token, config=config)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dense = self.features(values)
        return dense, self.token(dense)


@dataclass(frozen=True)
class ClockFeatureSchema:
    """Frozen field order and dimensionality for the T=3 X0 clock head."""

    token_dim: int
    motion_names: tuple[str, ...] = GLOBAL_TRANSPORT_FEATURE_NAMES
    dt_features: tuple[str, ...] = ("dt", "log_dt", "reciprocal_dt")
    version: str = "x0_t3_global_v1"

    def __post_init__(self) -> None:
        if self.token_dim <= 0:
            raise ValueError("clock token dimension must be positive")
        if self.motion_names != GLOBAL_TRANSPORT_FEATURE_NAMES:
            raise ValueError("clock motion feature names or order drifted")
        if self.dt_features != ("dt", "log_dt", "reciprocal_dt"):
            raise ValueError("clock delta-time feature schema drifted")

    @property
    def input_dim(self) -> int:
        return 7 * self.token_dim + 4 * len(self.motion_names) + 3 * len(self.dt_features) + 5

    def manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "token_dim": self.token_dim,
            "motion_names": list(self.motion_names),
            "dt_features": list(self.dt_features),
            "input_dim": self.input_dim,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["schema_sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload


def _dt_embed(delta: torch.Tensor) -> torch.Tensor:
    return torch.stack((delta, torch.log(delta + 1.0e-8), torch.reciprocal(delta)), dim=-1)


def assemble_x0_clock_features(
    tokens: torch.Tensor,
    motion01: torch.Tensor,
    motion12: torch.Tensor,
    delta_t_s: torch.Tensor,
    support: torch.Tensor,
    *,
    schema: ClockFeatureSchema,
) -> torch.Tensor:
    """Validate and concatenate the frozen X0 temporal vector in canonical order."""

    if tokens.ndim != 3 or tokens.shape[1:] != (3, schema.token_dim):
        raise ValueError("clock tokens must have shape [B,3,K]")
    batch = tokens.shape[0]
    motion_shape = (batch, len(schema.motion_names))
    if motion01.shape != motion_shape or motion12.shape != motion_shape:
        raise ValueError("clock motion features must have shape [B,9]")
    if delta_t_s.shape != (batch, 2) or support.shape != (batch, 3):
        raise ValueError("clock timing/support shapes drifted")
    values = (tokens, motion01, motion12, delta_t_s, support)
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("clock feature inputs must be finite")
    z0, z1, z2 = tokens.unbind(dim=1)
    d01 = z1 - z0
    d12 = z2 - z1
    d02 = z2 - z0
    motion_delta = motion12 - motion01
    dt01, dt12 = delta_t_s.unbind(dim=1)
    vector = torch.cat(
        (
            z0,
            z1,
            z2,
            d01,
            d12,
            d02,
            d12.abs(),
            motion01,
            motion12,
            motion_delta,
            motion_delta.abs(),
            _dt_embed(dt01),
            _dt_embed(dt12),
            _dt_embed(dt01 + dt12),
            support,
            torch.minimum(support[:, 0], support[:, 1]).unsqueeze(-1),
            torch.minimum(support[:, 1], support[:, 2]).unsqueeze(-1),
        ),
        dim=-1,
    )
    if vector.shape != (batch, schema.input_dim):
        raise RuntimeError("collision-clock feature vector schema drifted")
    return vector


FORBIDDEN_HEIGHT_MODULE_NAMES = (
    "foreground",
    "endpoint_projector",
    "height_correction_head",
    "pair_projector",
    "uncertainty_head",
    "auxiliary_inverse_ttc_head",
    "transport_projector",
    "transport_router",
)


def assert_height_bypass_module_tree(module: nn.Module) -> None:
    """Fail closed if a geometry/foreground interface is registered."""

    names = tuple(name.lower() for name, _child in module.named_modules())
    for forbidden in FORBIDDEN_HEIGHT_MODULE_NAMES:
        if any(forbidden in name for name in names):
            raise RuntimeError(f"forbidden height-interface module registered: {forbidden}")


def event_sensor_support(values: torch.Tensor) -> torch.Tensor:
    """Return the historical event support fraction using event tensors only."""

    if values.ndim != 4:
        raise ValueError("event endpoint must have shape [B,C,H,W]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("event endpoint contains non-finite values")
    return (values.abs() > 1.0e-8).to(values.dtype).mean(dim=(-3, -2, -1))


__all__ = [
    "ClockFeatureSchema",
    "FORBIDDEN_HEIGHT_MODULE_NAMES",
    "HeightBypassEncoderConfig",
    "HeightBypassEndpointEncoder",
    "assert_height_bypass_module_tree",
    "assemble_x0_clock_features",
    "event_sensor_support",
]
