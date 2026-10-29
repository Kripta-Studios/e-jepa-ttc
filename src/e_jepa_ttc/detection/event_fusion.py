"""Ungated residual event fusion for the controlled S4 detector matrix.

The module consumes the already normalized ten-channel causal voxel produced by
``event_data``. It does not inspect labels, fit a normalizer, or alter the RGB
backend. Its four outputs match the spatial shapes and 256-channel contract of
``ProjectedFeatureBackend`` features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn
from torch.nn import functional

EventFusionControl = Literal["F00", "F10", "F01", "F11"]


@dataclass(frozen=True)
class EventFusionSpec:
    """Frozen small-CNN and projected-feature contract."""

    input_channels: int = 10
    input_height: int = 384
    input_width: int = 640
    projected_channels: tuple[int, ...] = (256, 256, 256, 256)
    encoder_channels: tuple[int, ...] = (16, 32, 64, 64)
    fusion_channels: int = 32

    def __post_init__(self) -> None:
        integer_fields = (
            self.input_channels,
            self.input_height,
            self.input_width,
            self.fusion_channels,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields):
            raise ValueError("Event fusion dimensions must be positive integers")
        if len(self.projected_channels) != 4 or any(
            type(value) is not int or value != 256 for value in self.projected_channels
        ):
            raise ValueError("Event fusion requires exactly four C=256 projected scales")
        if len(self.encoder_channels) != 4 or any(
            type(value) is not int or value <= 0 for value in self.encoder_channels
        ):
            raise ValueError("Event encoder requires four positive channel widths")


@dataclass(frozen=True)
class EventFusionObservables:
    """Label-free event availability and density inputs for a future gate."""

    raw_event_count: torch.Tensor
    window_duration_us: torch.Tensor

    def validate(self, *, batch: int, device: torch.device) -> None:
        """Validate raw observables without fitting a learned normalizer."""
        for name, value in (
            ("raw_event_count", self.raw_event_count),
            ("window_duration_us", self.window_duration_us),
        ):
            if value.shape != (batch,) or value.device != device:
                raise ValueError(f"{name} must have shape [B] on the feature device")
            if value.dtype == torch.bool or not value.is_floating_point():
                raise ValueError(f"{name} must be floating point")
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"{name} must be finite and nonnegative")

    def fixed_gate_inputs(self) -> torch.Tensor:
        """Return detached fixed transforms; no target or learned statistics are used."""
        return torch.stack(
            (
                torch.log1p(self.raw_event_count),
                self.window_duration_us / 100_000.0,
            ),
            dim=1,
        ).detach()


class _EventEncoder(nn.Module):
    """Small stride pyramid from a ten-channel voxel to four projected scales."""

    def __init__(self, spec: EventFusionSpec) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        projections: list[nn.Module] = []
        source_channels = spec.input_channels
        for index, (encoded, projected) in enumerate(
            zip(spec.encoder_channels, spec.projected_channels, strict=True)
        ):
            stages.append(
                nn.Sequential(
                    nn.Conv2d(
                        source_channels,
                        encoded,
                        kernel_size=5 if index == 0 else 3,
                        stride=4 if index == 0 else 2,
                        padding=2 if index == 0 else 1,
                    ),
                    nn.SiLU(),
                    nn.Conv2d(encoded, encoded, kernel_size=3, padding=1),
                    nn.SiLU(),
                )
            )
            projections.append(nn.Conv2d(encoded, projected, kernel_size=1))
            source_channels = encoded
        self.stages = nn.ModuleList(stages)
        self.projections = nn.ModuleList(projections)

    def forward(
        self, voxel: torch.Tensor, output_shapes: Sequence[tuple[int, int]]
    ) -> tuple[torch.Tensor, ...]:
        encoded = voxel
        projected_scales = []
        for stage, projection, output_shape in zip(
            self.stages, self.projections, output_shapes, strict=True
        ):
            encoded = stage(encoded)
            projected = projection(encoded)
            if projected.shape[-2:] != output_shape:
                projected = functional.interpolate(
                    projected,
                    size=output_shape,
                    mode="bilinear",
                    align_corners=False,
                )
            projected_scales.append(projected)
        return tuple(projected_scales)


class _ScaleEventResidual(nn.Module):
    """Zero-initialized residual mixer for one projected RGB scale."""

    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Conv2d(2 * channels, bottleneck, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(bottleneck, channels, kernel_size=1),
        )
        output = cast(nn.Conv2d, self.mixer[-1])
        nn.init.zeros_(output.weight)
        if output.bias is not None:
            nn.init.zeros_(output.bias)

    def forward(
        self, current: torch.Tensor, event: torch.Tensor, effective_mask: torch.Tensor
    ) -> torch.Tensor:
        residual = self.mixer(torch.cat((current, event), dim=1))
        mask = effective_mask.to(dtype=current.dtype).view(-1, 1, 1, 1)
        return current + residual * mask


class CausalEventFeatureFusion(nn.Module):
    """Fuse causal event voxels into four projected RGB features.

    F00/F10 are RGB-only controls and reject event inputs. F01/F11 enable the
    same event branch with single-frame or temporal RGB respectively. A missing
    event batch (``voxel=None, mask=None``), a false mask, or a valid empty
    window (raw count zero) returns the exact RGB baseline for that sample.
    """

    def __init__(self, spec: EventFusionSpec | None = None) -> None:
        super().__init__()
        self.spec = EventFusionSpec() if spec is None else spec
        self.encoder = _EventEncoder(self.spec)
        self.residuals = nn.ModuleList(
            _ScaleEventResidual(channels, self.spec.fusion_channels)
            for channels in self.spec.projected_channels
        )

    def _validate_current(self, current_features: Sequence[torch.Tensor]) -> int:
        if len(current_features) != 4:
            raise ValueError("Event fusion requires four current projected scales")
        batch: int | None = None
        reference: torch.Tensor | None = None
        for index, (feature, channels) in enumerate(
            zip(current_features, self.spec.projected_channels, strict=True)
        ):
            if feature.ndim != 4 or feature.shape[1] != channels:
                raise ValueError(f"current_features[{index}] must be [B,256,H,W]")
            if feature.shape[0] <= 0 or min(feature.shape[-2:]) <= 0:
                raise ValueError("Current projected features must be nonempty")
            if not feature.is_floating_point() or not torch.isfinite(feature).all():
                raise ValueError("Current projected features must be finite floating point")
            if reference is None:
                reference = feature
                batch = feature.shape[0]
            elif (
                feature.shape[0] != batch
                or feature.device != reference.device
                or feature.dtype != reference.dtype
            ):
                raise ValueError("Current projected scales must share batch, device and dtype")
        assert batch is not None
        return batch

    def forward(
        self,
        current_features: Sequence[torch.Tensor],
        *,
        control: EventFusionControl,
        event_voxel: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
        observables: EventFusionObservables | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return fused scales under one explicit F00/F10/F01/F11 condition."""
        batch = self._validate_current(current_features)
        if control not in ("F00", "F10", "F01", "F11"):
            raise ValueError("Unknown event fusion control")
        if control in ("F00", "F10"):
            if event_voxel is not None or event_mask is not None or observables is not None:
                raise ValueError("RGB-only controls must not receive event inputs")
            return tuple(current_features)
        if event_voxel is None and event_mask is None and observables is None:
            return tuple(current_features)
        if event_voxel is None or event_mask is None or observables is None:
            raise ValueError("Event-enabled controls require voxel, mask and observables together")
        reference = current_features[0]
        expected = (
            batch,
            self.spec.input_channels,
            self.spec.input_height,
            self.spec.input_width,
        )
        if event_voxel.shape != expected:
            raise ValueError(f"event_voxel must have shape {expected}")
        if (
            event_voxel.device != reference.device
            or event_voxel.dtype != reference.dtype
            or not event_voxel.is_floating_point()
            or not torch.isfinite(event_voxel).all()
        ):
            raise ValueError("event_voxel must be finite and share feature device/dtype")
        if event_mask.shape != (batch,) or event_mask.dtype != torch.bool:
            raise ValueError("event_mask must be boolean [B]")
        if event_mask.device != reference.device:
            raise ValueError("event_mask must share the feature device")
        observables.validate(batch=batch, device=reference.device)
        if (observables.window_duration_us.masked_select(event_mask) <= 0).any():
            raise ValueError("Present event windows require positive duration")
        if (observables.raw_event_count.masked_select(~event_mask) != 0).any() or (
            observables.window_duration_us.masked_select(~event_mask) != 0
        ).any():
            raise ValueError("Missing event windows require zero observables")
        effective_mask = event_mask & (observables.raw_event_count > 0)
        if not effective_mask.any():
            return tuple(current_features)
        event_scales = self.encoder(
            event_voxel,
            [tuple(feature.shape[-2:]) for feature in current_features],
        )
        return tuple(
            residual(current, event, effective_mask)
            for current, event, residual in zip(
                current_features, event_scales, self.residuals, strict=True
            )
        )

    def parameter_count(self) -> int:
        """Return the exact trainable parameter count for resource planning."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def output_projections(self) -> tuple[nn.Conv2d, ...]:
        """Expose zero-initialized residual outputs for admission tests."""
        return tuple(
            cast(nn.Conv2d, cast(_ScaleEventResidual, item).mixer[-1])
            for item in self.residuals
        )
