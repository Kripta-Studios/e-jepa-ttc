"""E-Clock X0 direct benchmark-phase models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCOutput
from e_jepa_ttc.models.collision_clock_features import (
    ClockFeatureSchema,
    HeightBypassEncoderConfig,
    HeightBypassEndpointEncoder,
    assert_height_bypass_module_tree,
    assemble_x0_clock_features,
    event_sensor_support,
)
from e_jepa_ttc.models.collision_clock_math import (
    benchmark_phase_to_inverse_ttc,
    finite_ttc_from_inverse,
    neutral_raw_phase,
    phase_lower_bound,
)
from e_jepa_ttc.models.collision_clock_motion import height_free_global_transport_features

MotionFeatureMode = Literal[
    "embedded_a5",
    "global_uniform_zeroed_control",
    "global_uniform",
]


@dataclass(frozen=True)
class CollisionClockConfig:
    """Frozen X0 model contract."""

    in_channels: int = 12
    encoder_hidden_dim: int = 64
    encoder_token_dim: int = 128
    residual_depth: int = 2
    dropout: float = 0.05
    clock_hidden_dim: int = 128
    metric_delta_t_s: float = 0.1
    minimum_abs_prediction_ttc_s: float = 0.1
    ttc_clip_seconds: float = 60.0
    min_sensor_support: float = 1.0e-4
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    feature_source: Literal["a5_pair", "raw_endpoint"] = "raw_endpoint"
    motion_feature_mode: MotionFeatureMode = "global_uniform"
    transport_radius: int = 1
    transport_temperature: float = 0.02
    phase_weighting: Literal["uniform", "official_macro_mid"] = "uniform"
    learn_phase_variance: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk_thresholds_s", tuple(self.risk_thresholds_s))
        if min(
            self.in_channels,
            self.encoder_hidden_dim,
            self.encoder_token_dim,
            self.residual_depth,
            self.clock_hidden_dim,
        ) <= 0:
            raise ValueError("collision-clock dimensions and depth must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        scalar_positive = (
            self.metric_delta_t_s,
            self.minimum_abs_prediction_ttc_s,
            self.ttc_clip_seconds,
            self.min_sensor_support,
            self.transport_temperature,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in scalar_positive):
            raise ValueError("physical and transport constants must be finite and positive")
        if self.transport_radius < 1:
            raise ValueError("transport_radius must be positive")
        if not self.risk_thresholds_s or any(
            not math.isfinite(value) or value <= self.metric_delta_t_s
            for value in self.risk_thresholds_s
        ):
            raise ValueError("risk thresholds must be finite and exceed metric delta")
        if tuple(sorted(set(self.risk_thresholds_s))) != self.risk_thresholds_s:
            raise ValueError("risk thresholds must be unique and increasing")
        if self.feature_source == "a5_pair" and self.motion_feature_mode != "embedded_a5":
            raise ValueError("a5_pair requires embedded_a5 motion")
        if self.feature_source == "raw_endpoint" and self.motion_feature_mode == "embedded_a5":
            raise ValueError("raw_endpoint cannot use embedded_a5 motion")
        if self.learn_phase_variance:
            raise ValueError("X0-U does not authorize learned phase variance")


@dataclass
class CollisionClockTTCOutput:
    benchmark_phase_mean: torch.Tensor
    ttc_mean_seconds: torch.Tensor
    inverse_ttc_mean: torch.Tensor
    known_mask: torch.Tensor
    sensor_support: torch.Tensor
    global_clock_token: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    benchmark_phase_log_variance: torch.Tensor | None = None
    collision_logits: torch.Tensor | None = None


def _validate_inputs(
    inputs: torch.Tensor,
    delta_t_s: torch.Tensor,
    *,
    in_channels: int,
) -> tuple[int, int, int, int]:
    if inputs.ndim != 5 or inputs.shape[1] != 3:
        raise ValueError("X0 inputs must have shape [B,3,C,H,W]")
    batch, _steps, channels, height, width = inputs.shape
    if channels != in_channels:
        raise ValueError(f"inputs have {channels} channels; expected {in_channels}")
    if delta_t_s.shape != (batch, 2):
        raise ValueError("delta_t_s must have shape [B,2]")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("inputs must be finite")
    if not bool(torch.isfinite(delta_t_s).all()) or bool((delta_t_s <= 0.0).any()):
        raise ValueError("delta_t_s must be finite and strictly positive")
    return batch, channels, height, width


def _pair_dt_embed(delta: torch.Tensor) -> torch.Tensor:
    return torch.stack((delta, torch.log(delta + 1.0e-8), torch.reciprocal(delta)), dim=-1)


class _DirectPhaseHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        config: CollisionClockConfig,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("direct phase head must end with Linear")
        nn.init.zeros_(final.weight)
        nn.init.constant_(
            final.bias,
            neutral_raw_phase(
                metric_delta_t_s=config.metric_delta_t_s,
                minimum_abs_prediction_ttc_s=config.minimum_abs_prediction_ttc_s,
            ),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(values).squeeze(-1)
        return raw, values


def _physical_output(
    raw_phase: torch.Tensor,
    clock_token: torch.Tensor,
    support: torch.Tensor,
    config: CollisionClockConfig,
    diagnostics: dict[str, torch.Tensor],
) -> CollisionClockTTCOutput:
    lower = phase_lower_bound(
        metric_delta_t_s=config.metric_delta_t_s,
        minimum_abs_prediction_ttc_s=config.minimum_abs_prediction_ttc_s,
    )
    phase = lower + functional.softplus(raw_phase)
    inverse = benchmark_phase_to_inverse_ttc(
        phase,
        metric_delta_t_s=config.metric_delta_t_s,
    )
    ttc = finite_ttc_from_inverse(inverse, clip_seconds=config.ttc_clip_seconds)
    known = (
        torch.isfinite(phase)
        & torch.isfinite(inverse)
        & torch.isfinite(ttc)
        & (inverse.abs() >= 1.0 / config.ttc_clip_seconds)
        & (support >= config.min_sensor_support)
    )
    diagnostics = {
        **diagnostics,
        "raw_benchmark_phase": raw_phase,
        "distance_to_phase_lower_bound": phase - lower,
        "official_failure_region_excluded_by_parameterization": torch.ones_like(phase),
    }
    return CollisionClockTTCOutput(
        benchmark_phase_mean=phase,
        ttc_mean_seconds=ttc,
        inverse_ttc_mean=inverse,
        known_mask=known,
        sensor_support=support,
        global_clock_token=clock_token,
        diagnostics=diagnostics,
    )


class X0HeightBypassDirectPhase(nn.Module):
    """Matched BASE/DYN topology with the sole scientific switch at motion slots."""

    def __init__(
        self,
        encoder: HeightBypassEndpointEncoder,
        config: CollisionClockConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or CollisionClockConfig()
        if self.config.feature_source != "raw_endpoint":
            raise ValueError("height-bypass model requires feature_source=raw_endpoint")
        if encoder.config != HeightBypassEncoderConfig(
            in_channels=self.config.in_channels,
            hidden_dim=self.config.encoder_hidden_dim,
            token_dim=self.config.encoder_token_dim,
            residual_depth=self.config.residual_depth,
            dropout=self.config.dropout,
        ):
            raise ValueError("encoder and collision-clock configs disagree")
        self.encoder = encoder
        self.feature_schema = ClockFeatureSchema(self.config.encoder_token_dim)
        self.input_dim = self.feature_schema.input_dim
        self.clock_head = _DirectPhaseHead(
            self.input_dim,
            self.config.clock_hidden_dim,
            self.config.dropout,
            self.config,
        )
        assert_height_bypass_module_tree(self)

    def checkpoint_config(self) -> dict[str, object]:
        return asdict(self.config)

    def forward(self, inputs: torch.Tensor, delta_t_s: torch.Tensor) -> CollisionClockTTCOutput:
        batch, channels, height, width = _validate_inputs(
            inputs,
            delta_t_s,
            in_channels=self.config.in_channels,
        )
        flat = inputs.reshape(batch * 3, channels, height, width)
        dense_flat, token_flat = self.encoder(flat)
        dense = dense_flat.reshape(batch, 3, *dense_flat.shape[1:])
        tokens = token_flat.reshape(batch, 3, -1)
        support = event_sensor_support(flat).reshape(batch, 3)

        transport01 = height_free_global_transport_features(
            dense[:, 0],
            dense[:, 1],
            radius=self.config.transport_radius,
            temperature=self.config.transport_temperature,
        ).features
        transport12 = height_free_global_transport_features(
            dense[:, 1],
            dense[:, 2],
            radius=self.config.transport_radius,
            temperature=self.config.transport_temperature,
        ).features
        observed01 = transport01
        observed12 = transport12
        if self.config.motion_feature_mode == "global_uniform_zeroed_control":
            transport01 = transport01 * 0.0
            transport12 = transport12 * 0.0
        elif self.config.motion_feature_mode != "global_uniform":
            raise RuntimeError("invalid height-bypass motion mode")

        feature_vector = assemble_x0_clock_features(
            tokens,
            transport01,
            transport12,
            delta_t_s,
            support,
            schema=self.feature_schema,
        )
        raw_phase, clock_token = self.clock_head(feature_vector)
        return _physical_output(
            raw_phase,
            clock_token,
            support[:, 2],
            self.config,
            {
                "global_transport_01_observed": observed01,
                "global_transport_12_observed": observed12,
                "global_transport_01_consumed": transport01,
                "global_transport_12_consumed": transport12,
            },
        )


class X0PairDirectPhase(nn.Module):
    """Geometry-infused readout diagnostic over a frozen fold-local A5."""

    def __init__(self, source_a5: CausalScaleTTC, config: CollisionClockConfig) -> None:
        super().__init__()
        if config.feature_source != "a5_pair" or config.motion_feature_mode != "embedded_a5":
            raise ValueError("PAIR requires a5_pair/embedded_a5 config")
        self.config = config
        self.source_a5 = source_a5.eval()
        for parameter in self.source_a5.parameters():
            parameter.requires_grad_(False)
        input_dim = config.encoder_token_dim + 5
        self.clock_head = _DirectPhaseHead(
            input_dim,
            config.clock_hidden_dim,
            config.dropout,
            config,
        )

    def train(self, mode: bool = True) -> X0PairDirectPhase:
        super().train(mode)
        self.source_a5.eval()
        return self

    def forward(self, inputs: torch.Tensor, delta_t_s: torch.Tensor) -> CollisionClockTTCOutput:
        _validate_inputs(inputs, delta_t_s, in_channels=self.config.in_channels)
        with torch.no_grad():
            source = self.source_a5(inputs, delta_t_s)
        pair_token = source.pair_tokens[:, -1]
        dt12 = delta_t_s[:, 1]
        support = source.sensor_support
        feature_vector = torch.cat(
            (
                pair_token,
                _pair_dt_embed(dt12),
                support[:, -1:].reshape(-1, 1),
                torch.minimum(support[:, -2], support[:, -1]).unsqueeze(-1),
            ),
            dim=-1,
        )
        raw_phase, clock_token = self.clock_head(feature_vector)
        return _physical_output(
            raw_phase,
            clock_token,
            support[:, -1],
            self.config,
            {
                "source_a5_ttc_seconds": source.ttc_mean_seconds,
                "pair_is_geometry_infused": torch.ones_like(raw_phase),
            },
        )


class X0A5Replay(nn.Module):
    """Exact frozen A5 control wrapper; it is not a height-bypass arm."""

    def __init__(self, source_a5: CausalScaleTTC) -> None:
        super().__init__()
        self.source_a5 = source_a5.eval()
        for parameter in self.source_a5.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> X0A5Replay:
        super().train(False)
        self.source_a5.eval()
        return self

    def forward(self, inputs: torch.Tensor, delta_t_s: torch.Tensor) -> CausalScaleTTCOutput:
        with torch.no_grad():
            return self.source_a5(inputs, delta_t_s)


__all__ = [
    "CollisionClockConfig",
    "CollisionClockTTCOutput",
    "MotionFeatureMode",
    "X0A5Replay",
    "X0HeightBypassDirectPhase",
    "X0PairDirectPhase",
]
