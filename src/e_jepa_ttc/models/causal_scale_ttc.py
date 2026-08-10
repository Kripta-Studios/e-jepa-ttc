"""Shared causal foreground-scale model for event, RGB, and late-fusion TTC arms.

The primary prediction is deliberately geometry-bound.  Each endpoint produces a
foreground probability map, its differentiable vertical extent, and a geometry
token.  Consecutive extents define

``r = log(h_current / h_previous)`` and ``inverse_ttc = expm1(r) / delta_t``.

No bounding-box coordinate, category, sequence identifier, or direct TTC feature is
accepted by this module.  A learned correction is bounded and exactly antisymmetric
under endpoint reversal.  The direct inverse-TTC readout is auxiliary and never
feeds the primary TTC or risk outputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class CausalScaleTTCConfig:
    """Architecture and physical guardrails shared by the three v5 modality arms."""

    modality: Literal["event", "rgb"] = "event"
    in_channels: int = 12
    hidden_dim: int = 64
    geometry_dim: int = 128
    residual_depth: int = 2
    dropout: float = 0.05
    foreground_decoder: Literal[
        "bilinear",
        "deconv",
        "resize_conv",
        "equivariant_fullres",
        "equivariant_separable",
    ] = "bilinear"
    foreground_fullres_dim: int = 16
    foreground_temperature: float = 1.0
    foreground_temporal_smoothing: float = 0.0
    max_abs_log_ratio_residual: float = 0.05
    max_abs_log_height_correction: float = 0.0
    temporal_inverse_ttc_blend: float = 1.0
    min_abs_log_ratio: float = 2.0e-3
    min_sensor_support: float = 1.0e-4
    ttc_clip_seconds: float = 60.0
    initial_log_ratio_std: float = 0.03
    log_ratio_log_variance_min: float = -12.0
    log_ratio_log_variance_max: float = 2.0
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)

    def __post_init__(self) -> None:
        if self.modality not in {"event", "rgb"}:
            raise ValueError("modality must be 'event' or 'rgb'")
        if min(self.in_channels, self.hidden_dim, self.geometry_dim, self.residual_depth) <= 0:
            raise ValueError("causal-scale dimensions and depth must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if self.foreground_decoder not in {
            "bilinear",
            "deconv",
            "resize_conv",
            "equivariant_fullres",
            "equivariant_separable",
        }:
            raise ValueError(
                "foreground_decoder must be bilinear, deconv, resize_conv, "
                "equivariant_fullres, or equivariant_separable"
            )
        if self.foreground_fullres_dim <= 0:
            raise ValueError("foreground_fullres_dim must be positive")
        if self.foreground_temperature <= 0.0:
            raise ValueError("foreground_temperature must be positive")
        if not 0.0 <= self.foreground_temporal_smoothing <= 0.4:
            raise ValueError("foreground_temporal_smoothing must lie in [0,0.4]")
        if not 0.0 <= self.max_abs_log_ratio_residual <= 0.25:
            raise ValueError("max_abs_log_ratio_residual must lie in [0,0.25]")
        if not 0.0 <= self.max_abs_log_height_correction <= 0.5:
            raise ValueError("max_abs_log_height_correction must lie in [0,0.5]")
        if not 0.0 <= self.temporal_inverse_ttc_blend <= 1.0:
            raise ValueError("temporal_inverse_ttc_blend must lie in [0,1]")
        if self.min_abs_log_ratio <= 0.0 or self.min_sensor_support < 0.0:
            raise ValueError("physical support thresholds must be non-negative")
        if self.ttc_clip_seconds <= 0.0 or self.initial_log_ratio_std <= 0.0:
            raise ValueError("TTC clipping and initial uncertainty must be positive")
        if self.log_ratio_log_variance_min >= self.log_ratio_log_variance_max:
            raise ValueError("log-ratio variance bounds are reversed")
        if not self.risk_thresholds_s or any(value <= 0.0 for value in self.risk_thresholds_s):
            raise ValueError("risk thresholds must be non-empty and positive")
        if tuple(sorted(set(self.risk_thresholds_s))) != self.risk_thresholds_s:
            raise ValueError("risk thresholds must be unique and strictly increasing")


@dataclass
class SoftScaleObservation:
    """Differentiable foreground geometry for one or more endpoint maps."""

    height_normalized: torch.Tensor
    width_normalized: torch.Tensor
    centroid_x_normalized: torch.Tensor
    centroid_y_normalized: torch.Tensor
    foreground_fraction: torch.Tensor


@dataclass
class CausalScaleTTCOutput:
    """Physical TTC distribution plus auditable intermediate predictions."""

    ttc_mean_seconds: torch.Tensor
    ttc_log_variance: torch.Tensor
    inverse_ttc_mean: torch.Tensor
    inverse_ttc_log_variance: torch.Tensor
    collision_logits: torch.Tensor
    known_mask: torch.Tensor
    log_height_ratio: torch.Tensor
    pair_log_height_ratio: torch.Tensor
    analytic_log_height_ratio: torch.Tensor
    residual_log_height_ratio: torch.Tensor
    log_ratio_log_variance: torch.Tensor
    pair_ttc_seconds: torch.Tensor
    pair_inverse_ttc: torch.Tensor
    visible_height_normalized: torch.Tensor
    visible_width_normalized: torch.Tensor
    foreground_logits: torch.Tensor
    geometry_tokens: torch.Tensor
    pair_tokens: torch.Tensor
    auxiliary_inverse_ttc: torch.Tensor
    sensor_support: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    endpoint_dense_features: torch.Tensor | None = None


def soft_vertical_extent_from_logits(
    foreground_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> SoftScaleObservation:
    """Measure translation-invariant soft 2-D extent from foreground logits.

    The moment correction by one pixel squared makes the estimate exact for an
    ideal discrete uniform rectangle: ``extent = sqrt(12 * variance + pixel**2)``.
    Coordinates are normalized, so ratios are independent of input resolution.
    """

    if foreground_logits.ndim < 4 or foreground_logits.shape[-3] != 1:
        raise ValueError("foreground_logits must end in [1,H,W]")
    if foreground_logits.shape[-2] < 2 or foreground_logits.shape[-1] < 2:
        raise ValueError("foreground maps must be at least 2x2")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    probabilities = torch.sigmoid(foreground_logits / temperature)
    row_mass = probabilities.sum(dim=-1).squeeze(-2)
    column_mass = probabilities.sum(dim=-2).squeeze(-2)
    total = row_mass.sum(dim=-1).clamp_min(torch.finfo(probabilities.dtype).eps)
    height = foreground_logits.shape[-2]
    width = foreground_logits.shape[-1]
    y_coordinates = (
        torch.arange(height, device=foreground_logits.device, dtype=foreground_logits.dtype) + 0.5
    ) / float(height)
    x_coordinates = (
        torch.arange(width, device=foreground_logits.device, dtype=foreground_logits.dtype) + 0.5
    ) / float(width)
    centroid_y = (row_mass * y_coordinates).sum(dim=-1) / total
    centroid_x = (column_mass * x_coordinates).sum(dim=-1) / total
    variance_y = (
        row_mass * (y_coordinates - centroid_y.unsqueeze(-1)).square()
    ).sum(dim=-1) / total
    variance_x = (
        column_mass * (x_coordinates - centroid_x.unsqueeze(-1)).square()
    ).sum(dim=-1) / total
    pixel_height = 1.0 / float(height)
    pixel_width = 1.0 / float(width)
    height_extent = torch.sqrt(
        (12.0 * variance_y + pixel_height**2).clamp_min(pixel_height**2)
    )
    width_extent = torch.sqrt(
        (12.0 * variance_x + pixel_width**2).clamp_min(pixel_width**2)
    )
    return SoftScaleObservation(
        height_normalized=height_extent,
        width_normalized=width_extent,
        centroid_x_normalized=centroid_x,
        centroid_y_normalized=centroid_y,
        foreground_fraction=probabilities.mean(dim=(-3, -2, -1)),
    )


def target_log_ratio_from_ttc(
    ttc_seconds: torch.Tensor,
    delta_t_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the current/previous scale log-ratio and its physical validity mask."""

    if ttc_seconds.shape != delta_t_s.shape:
        raise ValueError("ttc_seconds and delta_t_s must share shape")
    if bool((delta_t_s <= 0.0).any()) or not bool(torch.isfinite(delta_t_s).all()):
        raise ValueError("delta_t_s must be finite and strictly positive")
    safe_ttc = torch.where(ttc_seconds.abs() > 1.0e-8, ttc_seconds, torch.ones_like(ttc_seconds))
    ratio = 1.0 + delta_t_s / safe_ttc
    valid = torch.isfinite(ttc_seconds) & (ttc_seconds.abs() > 1.0e-8) & (ratio > 0.0)
    log_ratio = torch.where(valid, torch.log(ratio.clamp_min(1.0e-8)), torch.zeros_like(ratio))
    return log_ratio, valid


def log_ratio_to_inverse_ttc(
    log_height_ratio: torch.Tensor,
    delta_t_s: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact constant-velocity current-endpoint scale identity."""

    if log_height_ratio.shape != delta_t_s.shape:
        raise ValueError("log_height_ratio and delta_t_s must share shape")
    if bool((delta_t_s <= 0.0).any()) or not bool(torch.isfinite(delta_t_s).all()):
        raise ValueError("delta_t_s must be finite and strictly positive")
    return torch.expm1(log_height_ratio.clamp(-12.0, 12.0)) / delta_t_s


def finite_ttc_from_inverse(
    inverse_ttc: torch.Tensor,
    *,
    minimum_abs_inverse_ttc: float,
    clip_seconds: float,
) -> torch.Tensor:
    """Return a finite transport value; callers must still respect ``known_mask``."""

    if minimum_abs_inverse_ttc <= 0.0 or clip_seconds <= 0.0:
        raise ValueError("finite TTC controls must be positive")
    sign = torch.where(
        inverse_ttc < 0.0,
        -torch.ones_like(inverse_ttc),
        torch.ones_like(inverse_ttc),
    )
    safe = sign * inverse_ttc.abs().clamp_min(minimum_abs_inverse_ttc)
    return torch.reciprocal(safe).clamp(-clip_seconds, clip_seconds)


def blend_current_inverse_ttc(
    pair_inverse_ttc: torch.Tensor,
    pair_known: torch.Tensor,
    current_delta_t_s: torch.Tensor,
    *,
    current_pair_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport the previous-pair estimate and blend it at the current endpoint."""

    if pair_inverse_ttc.ndim != 2 or pair_known.shape != pair_inverse_ttc.shape:
        raise ValueError("pair inverse TTC and known mask must share shape [B,P]")
    if current_delta_t_s.shape != pair_inverse_ttc.shape[:1]:
        raise ValueError("current delta_t_s must have shape [B]")
    if not 0.0 <= current_pair_weight <= 1.0:
        raise ValueError("current_pair_weight must lie in [0,1]")
    current = pair_inverse_ttc[:, -1]
    used = torch.zeros_like(current, dtype=torch.bool)
    if pair_inverse_ttc.shape[1] < 2 or current_pair_weight >= 1.0:
        return current, used
    previous = pair_inverse_ttc[:, -2]
    denominator = 1.0 - current_delta_t_s * previous
    transported = previous / denominator.clamp_min(1.0e-4)
    blended = current_pair_weight * current + (1.0 - current_pair_weight) * transported
    ratio_argument = 1.0 + current_delta_t_s * blended
    used = (
        pair_known[:, -2]
        & torch.isfinite(denominator)
        & (denominator > 1.0e-4)
        & torch.isfinite(blended)
        & torch.isfinite(ratio_argument)
        & (ratio_argument > 0.0)
    )
    return torch.where(used, blended, current), used


def smooth_temporal_foreground_logits(
    logits: torch.Tensor,
    *,
    neighbor_weight: float,
) -> torch.Tensor:
    """Apply reversal-equivariant temporal consensus without future stream leakage."""

    if logits.ndim != 5 or logits.shape[1] < 2:
        raise ValueError("foreground logits must have shape [B,T>=2,1,H,W]")
    if not 0.0 <= neighbor_weight <= 0.4:
        raise ValueError("neighbor_weight must lie in [0,0.4]")
    if neighbor_weight == 0.0:
        return logits
    padded = torch.cat((logits[:, :1], logits, logits[:, -1:]), dim=1)
    return (
        (1.0 - 2.0 * neighbor_weight) * padded[:, 1:-1]
        + neighbor_weight * padded[:, :-2]
        + neighbor_weight * padded[:, 2:]
    )


class _ResidualBlock(nn.Module):
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


class _EquivariantForegroundHead(nn.Module):
    """Stride-free depthwise foreground path with low compute and no phase aliasing."""

    def __init__(self, in_channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        groups = math.gcd(hidden, 8)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, groups=hidden, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _DilatedAxisHead(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = math.gcd(channels, 8)
        layers: list[nn.Module] = []
        for dilation in (1, 2, 4, 8):
            layers.extend(
                (
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=channels,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, channels),
                    nn.GELU(),
                    nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                )
            )
        layers.append(nn.Conv1d(channels, 1, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _SeparableEquivariantForegroundHead(nn.Module):
    """Infer filled row/column occupancy without striding or coordinate features."""

    def __init__(self, in_channels: int, hidden: int) -> None:
        super().__init__()
        groups = math.gcd(hidden, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
        )
        self.row_head = _DilatedAxisHead(hidden)
        self.column_head = _DilatedAxisHead(hidden)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.stem(values)
        row_logits = self.row_head(features.amax(dim=-1)).unsqueeze(-1)
        column_logits = self.column_head(features.amax(dim=-2)).unsqueeze(-2)
        return row_logits + column_logits


class _EndpointEncoder(nn.Module):
    def __init__(self, config: CausalScaleTTCConfig) -> None:
        super().__init__()
        half = max(config.hidden_dim // 2, 8)
        half_groups = math.gcd(half, 8)
        hidden_groups = math.gcd(config.hidden_dim, 8)
        self.features = nn.Sequential(
            nn.Conv2d(config.in_channels, half, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(half_groups, half),
            nn.GELU(),
            nn.Conv2d(half, config.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(hidden_groups, config.hidden_dim),
            nn.GELU(),
            *[
                _ResidualBlock(config.hidden_dim, config.dropout)
                for _ in range(config.residual_depth)
            ],
        )
        self.foreground_from_input = config.foreground_decoder in {
            "equivariant_fullres",
            "equivariant_separable",
        }
        if self.foreground_from_input:
            if config.foreground_decoder == "equivariant_separable":
                self.foreground = _SeparableEquivariantForegroundHead(
                    config.in_channels,
                    config.foreground_fullres_dim,
                )
            else:
                self.foreground = _EquivariantForegroundHead(
                    config.in_channels,
                    config.foreground_fullres_dim,
                    config.dropout,
                )
        elif config.foreground_decoder in {"deconv", "resize_conv"}:
            decoder_channels = max(config.hidden_dim // 2, 8)
            decoder_groups = math.gcd(decoder_channels, 8)
            if config.foreground_decoder == "deconv":
                upsample: nn.Module = nn.ConvTranspose2d(
                    config.hidden_dim,
                    decoder_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            else:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
                    nn.Conv2d(
                        config.hidden_dim,
                        decoder_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                )
            self.foreground = nn.Sequential(
                upsample,
                nn.GroupNorm(decoder_groups, decoder_channels),
                nn.GELU(),
                nn.Conv2d(decoder_channels, 1, kernel_size=3, padding=1),
            )
        else:
            self.foreground = nn.Conv2d(config.hidden_dim, 1, kernel_size=1)
        self.token = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.hidden_dim, config.geometry_dim),
            nn.LayerNorm(config.geometry_dim),
        )

    def forward(
        self,
        values: torch.Tensor,
        *,
        return_dense_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features = self.features(values)
        foreground_input = values if self.foreground_from_input else features
        return (
            self.foreground(foreground_input),
            self.token(features),
            features if return_dense_features else None,
        )


class _AntisymmetricResidual(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float, maximum: float) -> None:
        super().__init__()
        self.maximum = maximum
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        final = self.scorer[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("antisymmetric scorer must end with Linear")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _ordered(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        return self.scorer(torch.cat([previous, current, current - previous], dim=-1)).squeeze(-1)

    def forward(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        forward = self._ordered(previous, current)
        reverse = self._ordered(current, previous)
        return self.maximum * torch.tanh(0.5 * (forward - reverse))


class CausalScaleTTC(nn.Module):
    """Efficient event/RGB endpoint encoder with a geometry-bound TTC readout."""

    def __init__(self, config: CausalScaleTTCConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalScaleTTCConfig()
        self.encoder = _EndpointEncoder(self.config)
        endpoint_input = self.config.geometry_dim + 4
        self.endpoint_projector = nn.Sequential(
            nn.LayerNorm(endpoint_input),
            nn.Linear(endpoint_input, self.config.geometry_dim),
            nn.GELU(),
        )
        if self.config.max_abs_log_height_correction > 0.0:
            self.height_correction_head: nn.Module | None = nn.Sequential(
                nn.LayerNorm(self.config.geometry_dim),
                nn.Linear(self.config.geometry_dim, 1),
            )
            correction_final = self.height_correction_head[-1]
            if not isinstance(correction_final, nn.Linear):
                raise TypeError("height correction head must end with Linear")
            nn.init.zeros_(correction_final.weight)
            nn.init.zeros_(correction_final.bias)
        else:
            self.height_correction_head = None
        self.residual = _AntisymmetricResidual(
            self.config.geometry_dim,
            self.config.geometry_dim,
            self.config.dropout,
            self.config.max_abs_log_ratio_residual,
        )
        pair_input = self.config.geometry_dim * 3
        self.pair_projector = nn.Sequential(
            nn.LayerNorm(pair_input),
            nn.Linear(pair_input, self.config.geometry_dim),
            nn.GELU(),
        )
        uncertainty_input = self.config.geometry_dim * 2 + 1
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(uncertainty_input),
            nn.Linear(uncertainty_input, self.config.geometry_dim),
            nn.GELU(),
            nn.Linear(self.config.geometry_dim, 1),
        )
        uncertainty_final = self.uncertainty_head[-1]
        if not isinstance(uncertainty_final, nn.Linear):
            raise TypeError("uncertainty head must end with Linear")
        nn.init.zeros_(uncertainty_final.weight)
        nn.init.constant_(
            uncertainty_final.bias,
            2.0 * math.log(self.config.initial_log_ratio_std),
        )
        self.auxiliary_inverse_ttc_head = nn.Sequential(
            nn.LayerNorm(self.config.geometry_dim),
            nn.Linear(self.config.geometry_dim, 1),
        )
        self.log_ratio_variance_offset: torch.Tensor
        self.register_buffer("log_ratio_variance_offset", torch.tensor(0.0))

    def set_log_ratio_variance_offset(self, value: float) -> None:
        """Set a validation-fitted scalar variance offset without changing means."""

        if not math.isfinite(value):
            raise ValueError("log-ratio variance offset must be finite")
        self.log_ratio_variance_offset[...] = value

    def _sensor_support(self, values: torch.Tensor) -> torch.Tensor:
        if self.config.modality == "event":
            return (values.abs() > 1.0e-8).to(values.dtype).mean(dim=(-3, -2, -1))
        spatial_std = values.float().std(dim=(-3, -2, -1), unbiased=False).to(values.dtype)
        mean = values.float().mean(dim=(-3, -2, -1)).to(values.dtype)
        exposure = (4.0 * mean.clamp(0.0, 1.0) * (1.0 - mean.clamp(0.0, 1.0))).sqrt()
        return (spatial_std / 0.1).clamp(0.0, 1.0) * exposure

    def forward(
        self,
        inputs: torch.Tensor,
        delta_t_s: torch.Tensor,
        *,
        return_dense_features: bool = False,
    ) -> CausalScaleTTCOutput:
        """Predict current TTC from causal endpoint tensors ``[B,T,C,H,W]``."""

        if inputs.ndim != 5 or inputs.shape[1] < 2:
            raise ValueError("inputs must have shape [B,T>=2,C,H,W]")
        if inputs.shape[2] != self.config.in_channels:
            raise ValueError(
                f"inputs have {inputs.shape[2]} channels; expected {self.config.in_channels}"
            )
        batch, steps, channels, height, width = inputs.shape
        if delta_t_s.shape != (batch, steps - 1):
            raise ValueError("delta_t_s must have shape [B,T-1]")
        if bool((delta_t_s <= 0.0).any()) or not bool(torch.isfinite(delta_t_s).all()):
            raise ValueError("delta_t_s must be finite and strictly positive")
        flat = inputs.reshape(batch * steps, channels, height, width)
        lowres_logits, base_tokens, flat_dense = self.encoder(
            flat, return_dense_features=return_dense_features,
        )
        low_h, low_w = lowres_logits.shape[-2:]
        lowres_logits = lowres_logits.reshape(batch, steps, 1, low_h, low_w)
        foreground_logits = functional.interpolate(
            lowres_logits.reshape(batch * steps, 1, low_h, low_w),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, steps, 1, height, width)
        foreground_logits = smooth_temporal_foreground_logits(
            foreground_logits,
            neighbor_weight=self.config.foreground_temporal_smoothing,
        )
        observation = soft_vertical_extent_from_logits(
            foreground_logits,
            temperature=self.config.foreground_temperature,
        )
        sensor_support = self._sensor_support(inputs)
        endpoint_scalars = torch.stack(
            (
                observation.height_normalized.clamp_min(1.0e-6).log(),
                observation.centroid_y_normalized,
                observation.foreground_fraction,
                sensor_support,
            ),
            dim=-1,
        )
        geometry_tokens = self.endpoint_projector(
            torch.cat([base_tokens.reshape(batch, steps, -1), endpoint_scalars], dim=-1)
        )
        raw_log_height = observation.height_normalized.clamp_min(1.0e-6).log()
        if self.height_correction_head is None:
            log_height_correction = torch.zeros_like(raw_log_height)
        else:
            log_height_correction = self.config.max_abs_log_height_correction * torch.tanh(
                self.height_correction_head(geometry_tokens).squeeze(-1)
            )
        corrected_log_height = raw_log_height + log_height_correction
        visible_height = corrected_log_height.exp()
        previous = geometry_tokens[:, :-1]
        current = geometry_tokens[:, 1:]
        raw_foreground_ratio = raw_log_height[:, 1:] - raw_log_height[:, :-1]
        raw_log_width = observation.width_normalized.clamp_min(1.0e-6).log()
        raw_foreground_width_ratio = raw_log_width[:, 1:] - raw_log_width[:, :-1]
        analytic = corrected_log_height[:, 1:] - corrected_log_height[:, :-1]
        residual = self.residual(previous, current)
        log_ratio = analytic + residual
        pair_tokens = self.pair_projector(
            torch.cat([previous, current, current - previous], dim=-1)
        )
        pair_support = torch.minimum(sensor_support[:, :-1], sensor_support[:, 1:])
        symmetric = torch.cat(
            [previous + current, (current - previous).abs(), pair_support.unsqueeze(-1)],
            dim=-1,
        )
        log_ratio_log_variance = (
            self.uncertainty_head(symmetric)
            .squeeze(-1)
            .add(self.log_ratio_variance_offset)
            .clamp(
                self.config.log_ratio_log_variance_min,
                self.config.log_ratio_log_variance_max,
            )
        )
        pair_inverse_ttc = log_ratio_to_inverse_ttc(log_ratio, delta_t_s)
        minimum_inverse = 1.0 / self.config.ttc_clip_seconds
        pair_ttc = finite_ttc_from_inverse(
            pair_inverse_ttc,
            minimum_abs_inverse_ttc=minimum_inverse,
            clip_seconds=self.config.ttc_clip_seconds,
        )
        pair_known = (
            (log_ratio.abs() >= self.config.min_abs_log_ratio)
            & (pair_support >= self.config.min_sensor_support)
            & torch.isfinite(log_ratio)
        )
        current_dt = delta_t_s[:, -1]
        inverse_ttc, temporal_blend_used = blend_current_inverse_ttc(
            pair_inverse_ttc,
            pair_known,
            current_dt,
            current_pair_weight=self.config.temporal_inverse_ttc_blend,
        )
        current_ratio = torch.log1p((current_dt * inverse_ttc).clamp_min(-1.0 + 1.0e-6))
        effective_log_ratio = torch.cat(
            (log_ratio[:, :-1], current_ratio.unsqueeze(-1)),
            dim=1,
        )
        ttc = finite_ttc_from_inverse(
            inverse_ttc,
            minimum_abs_inverse_ttc=minimum_inverse,
            clip_seconds=self.config.ttc_clip_seconds,
        )
        known = pair_known[:, -1] & torch.isfinite(current_ratio)
        current_log_variance = log_ratio_log_variance[:, -1]
        inverse_log_variance = (
            current_log_variance + 2.0 * current_ratio.clamp(-12.0, 12.0) - 2.0 * current_dt.log()
        ).clamp(-16.0, 16.0)
        denominator = torch.expm1(current_ratio.clamp(-12.0, 12.0)).abs().clamp_min(1.0e-6)
        ttc_log_variance = (
            current_log_variance
            + 2.0 * current_dt.log()
            + 2.0 * current_ratio.clamp(-12.0, 12.0)
            - 4.0 * denominator.log()
        ).clamp(-16.0, 16.0)
        inverse_std = torch.exp(0.5 * inverse_log_variance).clamp_min(1.0e-6)
        thresholds = inverse_ttc.new_tensor(self.config.risk_thresholds_s)
        standardized = (inverse_ttc[:, None] - thresholds.reciprocal()[None, :]) / inverse_std[
            :, None
        ]
        risk_probability = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
        risk_probability = risk_probability.clamp(1.0e-6, 1.0 - 1.0e-6)
        collision_logits = torch.logit(risk_probability)
        collision_logits = torch.where(
            known[:, None],
            collision_logits,
            torch.zeros_like(collision_logits),
        )
        auxiliary_inverse_ttc = self.auxiliary_inverse_ttc_head(pair_tokens).squeeze(-1)
        endpoint_dense_features = (
            flat_dense.reshape(
                batch, steps, flat_dense.shape[1], flat_dense.shape[2], flat_dense.shape[3]
            )
            if flat_dense is not None
            else None
        )
        return CausalScaleTTCOutput(
            ttc_mean_seconds=ttc,
            ttc_log_variance=ttc_log_variance,
            inverse_ttc_mean=inverse_ttc,
            inverse_ttc_log_variance=inverse_log_variance,
            collision_logits=collision_logits,
            known_mask=known,
            log_height_ratio=effective_log_ratio,
            pair_log_height_ratio=log_ratio,
            analytic_log_height_ratio=analytic,
            residual_log_height_ratio=residual,
            log_ratio_log_variance=log_ratio_log_variance,
            pair_ttc_seconds=pair_ttc,
            pair_inverse_ttc=pair_inverse_ttc,
            visible_height_normalized=visible_height,
            visible_width_normalized=observation.width_normalized,
            foreground_logits=foreground_logits,
            geometry_tokens=geometry_tokens,
            pair_tokens=pair_tokens,
            auxiliary_inverse_ttc=auxiliary_inverse_ttc,
            sensor_support=sensor_support,
            diagnostics={
                "foreground_centroid_y": observation.centroid_y_normalized,
                "foreground_centroid_x": observation.centroid_x_normalized,
                "foreground_fraction": observation.foreground_fraction,
                "raw_foreground_height_normalized": observation.height_normalized,
                "raw_foreground_width_normalized": observation.width_normalized,
                "raw_foreground_log_height_ratio": raw_foreground_ratio,
                "raw_foreground_log_width_ratio": raw_foreground_width_ratio,
                "log_height_correction": log_height_correction,
                "pair_sensor_support": pair_support,
                "pair_known": pair_known,
                "temporal_blend_used": temporal_blend_used,
                "foreground_temporal_smoothing": foreground_logits.new_full(
                    (batch,), self.config.foreground_temporal_smoothing
                ),
            },
            endpoint_dense_features=endpoint_dense_features,
        )

    def checkpoint_config(self) -> dict[str, object]:
        """Return the exact JSON-safe architecture configuration."""

        return asdict(self.config)


__all__ = [
    "CausalScaleTTC",
    "CausalScaleTTCConfig",
    "CausalScaleTTCOutput",
    "SoftScaleObservation",
    "blend_current_inverse_ttc",
    "finite_ttc_from_inverse",
    "log_ratio_to_inverse_ttc",
    "soft_vertical_extent_from_logits",
    "target_log_ratio_from_ttc",
]
