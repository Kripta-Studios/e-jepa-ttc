"""Bounded causal RGB feature fusion for the S3 detector experiment.

The module operates on already projected image features. Historical tensors are
detached inside the module so the first experiment has bounded activation memory.
Temporal admissibility remains the responsibility of ``temporal_data``; this
module independently rejects malformed masks, ages and tensor shapes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalFusionSpec:
    """Frozen architecture choices for one temporal-fusion candidate."""

    channels: tuple[int, ...] = (256, 256, 256, 256)
    bottleneck_channels: int = 32
    age_reference_us: float = 100_000.0
    history_slots: int = 2

    def __post_init__(self) -> None:
        if not self.channels or any(
            type(value) is not int or value <= 0 for value in self.channels
        ):
            raise ValueError("channels must contain positive integers")
        if type(self.bottleneck_channels) is not int or self.bottleneck_channels <= 0:
            raise ValueError("bottleneck_channels must be a positive integer")
        if not isinstance(self.age_reference_us, (int, float)) or isinstance(
            self.age_reference_us, bool
        ):
            raise ValueError("age_reference_us must be a positive finite number")
        if (
            not torch.isfinite(torch.tensor(float(self.age_reference_us)))
            or self.age_reference_us <= 0
        ):
            raise ValueError("age_reference_us must be a positive finite number")
        if self.history_slots != 2:
            raise ValueError("The preregistered S3 candidate has exactly two history slots")


class _ScaleTemporalFusion(nn.Module):
    """Identity-initialized residual fusion for one projected feature scale."""

    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        self.current_projection = nn.Conv2d(channels, bottleneck, kernel_size=1)
        self.history_projection = nn.Conv2d(channels, bottleneck, kernel_size=1)
        self.age_embedding = nn.Sequential(
            nn.Linear(1, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),
        )
        self.mixer = nn.Sequential(
            nn.Conv2d(3 * bottleneck, bottleneck, kernel_size=1),
            nn.SiLU(),
        )
        self.output_projection = nn.Conv2d(bottleneck, channels, kernel_size=1)
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        current: torch.Tensor,
        history: torch.Tensor,
        normalized_age: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = current.shape[0]
        historical = history.detach()
        slot_features = []
        for slot in range(2):
            projected = self.history_projection(historical[:, slot])
            age = self.age_embedding(normalized_age[:, slot : slot + 1]).view(
                batch, -1, 1, 1
            )
            gate = mask[:, slot].to(dtype=current.dtype).view(batch, 1, 1, 1)
            slot_features.append((projected + age) * gate)
        mixed = torch.cat(
            [self.current_projection(current), slot_features[0], slot_features[1]], dim=1
        )
        residual = self.output_projection(self.mixer(mixed))
        any_history = mask.any(dim=1).to(dtype=current.dtype).view(batch, 1, 1, 1)
        return current + residual * any_history


class CausalTemporalFeatureFusion(nn.Module):
    """Fuse current MonoDGP feature scales with at most two causal RGB histories.

    ``history_features[scale]`` has shape ``[B, 2, C, H, W]`` in chronological
    order (older, newer). Missing slots contain zeros and are declared by the
    boolean mask. ``history_limit`` selects one, two or three total RGB frames;
    the one-history variant uses the newer (second) slot.
    """

    def __init__(self, spec: TemporalFusionSpec | None = None) -> None:
        super().__init__()
        self.spec = TemporalFusionSpec() if spec is None else spec
        self.scales = nn.ModuleList(
            _ScaleTemporalFusion(channels, self.spec.bottleneck_channels)
            for channels in self.spec.channels
        )

    def _validate_current(self, current_features: Sequence[torch.Tensor]) -> tuple[int, ...]:
        if len(current_features) != len(self.spec.channels):
            raise ValueError("Current feature scale count differs from the frozen specification")
        batch: int | None = None
        device: torch.device | None = None
        dtype: torch.dtype | None = None
        for index, (feature, channels) in enumerate(
            zip(current_features, self.spec.channels, strict=True)
        ):
            if not isinstance(feature, torch.Tensor) or feature.ndim != 4:
                raise ValueError(f"current_features[{index}] must have shape [B,C,H,W]")
            if not feature.is_floating_point() or not torch.isfinite(feature).all():
                raise ValueError(f"current_features[{index}] must be finite floating point")
            if feature.shape[1] != channels or feature.shape[2] <= 0 or feature.shape[3] <= 0:
                raise ValueError(f"current_features[{index}] has an unexpected shape")
            if batch is None:
                batch = feature.shape[0]
                device = feature.device
                dtype = feature.dtype
            elif feature.shape[0] != batch:
                raise ValueError("All current feature scales must share batch size")
            elif feature.device != device or feature.dtype != dtype:
                raise ValueError("All current feature scales must share device and dtype")
            if feature.shape[0] <= 0:
                raise ValueError("Temporal fusion does not accept an empty batch")
        assert batch is not None
        return (batch,)

    def forward(
        self,
        current_features: Sequence[torch.Tensor],
        history_features: Sequence[torch.Tensor] | None = None,
        history_ages_us: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        *,
        history_limit: int = 2,
    ) -> tuple[torch.Tensor, ...]:
        """Return fused scales while retaining exact single-frame behavior."""
        (batch,) = self._validate_current(current_features)
        if type(history_limit) is not int or history_limit not in (0, 1, 2):
            raise ValueError("history_limit must be 0, 1 or 2")
        if history_limit == 0:
            if (
                history_features is not None
                or history_ages_us is not None
                or history_mask is not None
            ):
                raise ValueError("Single-frame mode must not receive history tensors")
            return tuple(current_features)
        if history_features is None or history_ages_us is None or history_mask is None:
            raise ValueError("Temporal mode requires features, ages and masks")
        if len(history_features) != len(current_features):
            raise ValueError("History and current feature scale counts differ")
        reference = current_features[0]
        if history_mask.dtype != torch.bool or history_mask.shape != (batch, 2):
            raise ValueError("history_mask must be boolean [B,2]")
        if history_mask.device != reference.device:
            raise ValueError("history_mask must share the feature device")
        if history_ages_us.shape != (batch, 2) or history_ages_us.device != reference.device:
            raise ValueError("history_ages_us must have shape [B,2] on the feature device")
        if history_ages_us.dtype == torch.bool or not history_ages_us.is_floating_point():
            raise ValueError("history_ages_us must be floating point")
        if not torch.isfinite(history_ages_us).all() or (history_ages_us < 0).any():
            raise ValueError("history ages must be finite and nonnegative")
        if (history_ages_us.masked_select(~history_mask) != 0).any():
            raise ValueError("Missing history slots must have zero age")
        if history_limit == 1 and history_mask[:, 0].any():
            raise ValueError("One-history mode uses only the newer second slot")
        normalized_age = torch.log1p(
            history_ages_us.to(dtype=reference.dtype) / float(self.spec.age_reference_us)
        )
        fused = []
        for index, (current, history, scale) in enumerate(
            zip(current_features, history_features, self.scales, strict=True)
        ):
            expected = (batch, 2, current.shape[1], current.shape[2], current.shape[3])
            if history.shape != expected:
                raise ValueError(f"history_features[{index}] must have shape {expected}")
            if history.device != current.device or history.dtype != current.dtype:
                raise ValueError("History must share each current feature's device and dtype")
            if not history.is_floating_point() or not torch.isfinite(history).all():
                raise ValueError("History features must be finite floating point")
            fused.append(scale(current, history, normalized_age, history_mask))
        return tuple(fused)

    def parameter_count(self) -> int:
        """Return the exact number of trainable parameters for resource receipts."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def output_projections(self) -> tuple[nn.Conv2d, ...]:
        """Expose final residual projections for initialization audits."""
        return tuple(cast(_ScaleTemporalFusion, scale).output_projection for scale in self.scales)
