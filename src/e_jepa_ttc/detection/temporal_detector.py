"""Stateless adapter for a controlled MonoDGP temporal-feature variant.

The historical detector is not modified here. A future copied portable source
must implement :class:`ProjectedFeatureBackend` after passing the exact source
admission contract in this module.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from e_jepa_ttc.detection.temporal_fusion import CausalTemporalFeatureFusion

TemporalDetectorControl = Literal[
    "current_only",
    "current_repeated",
    "correct_past",
    "permuted_past",
    "delayed_past",
]

TEMPORAL_PORTABLE_VARIANT = "monodgp_projected_temporal_v1"
MONODGP_SOURCE_SHA256 = "ac16913d545b470cf1f741df2dd783f0ca82eaca515cc94a824a7074c4113c6e"
MONODGP_INTEGRATION_MARKER = "        # region enhancement\n"


def admit_monodgp_temporal_source(source_file: str | Path) -> dict[str, object]:
    """Fail closed unless the copied MonoDGP source is the inspected base."""
    path = Path(source_file)
    if not path.is_file():
        raise FileNotFoundError(f"MonoDGP temporal base source is absent: {path}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != MONODGP_SOURCE_SHA256:
        raise ValueError("MonoDGP temporal base source SHA-256 differs")
    text = data.decode("utf-8")
    if text.count(MONODGP_INTEGRATION_MARKER) != 1:
        raise ValueError("MonoDGP temporal integration marker is absent or ambiguous")
    return {
        "schema": "monodgp_temporal_source_admission_v1",
        "variant": TEMPORAL_PORTABLE_VARIANT,
        "source_file": str(path.resolve()),
        "source_sha256": digest,
        "integration_marker_sha256": hashlib.sha256(
            MONODGP_INTEGRATION_MARKER.encode("utf-8")
        ).hexdigest(),
        "integration_point": "after all input_proj scales and before region_head",
        "status": "PASS_EXACT_BASE",
    }


@dataclass(frozen=True)
class ProjectedFeatureState:
    """Projected image features plus call-local decoder context."""

    features: tuple[torch.Tensor, ...]
    decoder_context: object


class ProjectedFeatureBackend(nn.Module, ABC):
    """Explicit split required from the controlled copied MonoDGP source."""

    @abstractmethod
    def extract_projected_features(self, images: torch.Tensor) -> ProjectedFeatureState:
        """Run the shared backbone and input projections for one image batch."""

    @abstractmethod
    def decode_projected_features(
        self,
        state: ProjectedFeatureState,
        features: Sequence[torch.Tensor],
        calibs: torch.Tensor,
        targets: Sequence[Mapping[str, torch.Tensor]] | None,
        image_sizes: torch.Tensor,
        dn_args: object | None = None,
    ) -> dict[str, Any]:
        """Run the unchanged region heads, transformers and output heads."""


@dataclass(frozen=True)
class TemporalFrameIdentity:
    """Identity used to keep history inside one sequence generation."""

    sequence_id: str
    frame_id: str
    generation: int = 0

    def __post_init__(self) -> None:
        if not self.sequence_id or not self.frame_id:
            raise ValueError("Temporal frame identity fields must be nonempty")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("Temporal frame generation must be a nonnegative integer")


@dataclass(frozen=True)
class TemporalBatchContext:
    """Complete identity declaration for one stateless temporal forward."""

    producer_version: str
    control: TemporalDetectorControl
    current: tuple[TemporalFrameIdentity, ...]
    history: tuple[
        tuple[TemporalFrameIdentity | None, TemporalFrameIdentity | None], ...
    ]

    def validate(
        self,
        *,
        batch_size: int,
        producer_version: str,
        history_mask: torch.Tensor | None,
        history_ages_us: torch.Tensor | None,
        history_limit: int,
    ) -> None:
        """Reject cross-sequence, stale-generation or mislabeled controls."""
        if self.producer_version != producer_version:
            raise ValueError("Temporal context producer version differs from the adapter")
        if self.control not in (
            "current_only",
            "current_repeated",
            "correct_past",
            "permuted_past",
            "delayed_past",
        ):
            raise ValueError("Unknown temporal detector control")
        if len(self.current) != batch_size or len(self.history) != batch_size:
            raise ValueError("Temporal context batch size differs from image batch")
        if len(set(self.current)) != batch_size:
            raise ValueError("Current temporal identities must be unique within a batch")
        if history_limit == 0:
            if self.control != "current_only" or any(any(row) for row in self.history):
                raise ValueError("Single-frame context must be current_only without history")
            return
        if history_mask is None or history_ages_us is None:
            raise ValueError("Temporal context validation requires history mask and ages")
        if self.control == "current_only":
            raise ValueError("Temporal histories cannot use the current_only control")
        for batch_index, (current, histories) in enumerate(
            zip(self.current, self.history, strict=True)
        ):
            if len(histories) != 2:
                raise ValueError("Each temporal context row must contain exactly two slots")
            for slot, identity in enumerate(histories):
                valid = bool(history_mask[batch_index, slot])
                if valid != (identity is not None):
                    raise ValueError("Temporal identities and history mask disagree")
                if identity is None:
                    continue
                if (
                    identity.sequence_id != current.sequence_id
                    or identity.generation != current.generation
                ):
                    raise ValueError("Temporal history crosses sequence or generation")
                age = float(history_ages_us[batch_index, slot])
                if self.control == "current_repeated":
                    if identity != current or age != 0:
                        raise ValueError("Repeated-current history must equal current at age zero")
                elif identity == current or age <= 0:
                    raise ValueError("Past-frame controls require positive-age noncurrent history")
        if self.control == "current_repeated" and not history_mask.all():
            raise ValueError("Repeated-current control requires both history slots")


class TemporalDetectorAdapter(nn.Module):
    """Recompute bounded history and fuse it before the unchanged detector decoder."""

    def __init__(
        self,
        backend: ProjectedFeatureBackend,
        fusion: CausalTemporalFeatureFusion,
        *,
        producer_version: str,
        maximum_history_age_us: float,
    ) -> None:
        super().__init__()
        if not producer_version:
            raise ValueError("producer_version must be nonempty")
        if (
            isinstance(maximum_history_age_us, bool)
            or not isinstance(maximum_history_age_us, (int, float))
            or not math.isfinite(float(maximum_history_age_us))
            or maximum_history_age_us <= 0
        ):
            raise ValueError("maximum_history_age_us must be positive and finite")
        self.backend = backend
        self.fusion = fusion
        self.producer_version = producer_version
        self.maximum_history_age_us = float(maximum_history_age_us)

    @classmethod
    def from_strict_baseline(
        cls,
        backend: ProjectedFeatureBackend,
        baseline_state: Mapping[str, torch.Tensor],
        fusion: CausalTemporalFeatureFusion,
        *,
        producer_version: str,
        maximum_history_age_us: float,
    ) -> TemporalDetectorAdapter:
        """Load the complete single-frame producer before attaching new parameters."""
        backend.load_state_dict(dict(baseline_state), strict=True)
        return cls(
            backend,
            fusion,
            producer_version=producer_version,
            maximum_history_age_us=maximum_history_age_us,
        )

    def forward(
        self,
        current_images: torch.Tensor,
        calibs: torch.Tensor,
        targets: Sequence[Mapping[str, torch.Tensor]] | None,
        image_sizes: torch.Tensor,
        *,
        temporal_context: TemporalBatchContext,
        history_images: torch.Tensor | None = None,
        history_ages_us: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        history_limit: int = 0,
        dn_args: object | None = None,
    ) -> dict[str, Any]:
        """Run one stateless current/history request without a feature cache."""
        if current_images.ndim != 4 or not current_images.is_floating_point():
            raise ValueError("current_images must be floating point [B,C,H,W]")
        if current_images.shape[0] <= 0 or not torch.isfinite(current_images).all():
            raise ValueError("current_images must be a nonempty finite batch")
        batch = current_images.shape[0]
        temporal_context.validate(
            batch_size=batch,
            producer_version=self.producer_version,
            history_mask=history_mask,
            history_ages_us=history_ages_us,
            history_limit=history_limit,
        )
        current_state = self.backend.extract_projected_features(current_images)
        if history_limit == 0:
            fused = self.fusion(current_state.features, history_limit=0)
        else:
            if history_images is None or history_ages_us is None or history_mask is None:
                raise ValueError("Temporal forward requires history images, ages and masks")
            expected_prefix = (batch, 2, *current_images.shape[1:])
            if history_images.shape != expected_prefix:
                raise ValueError(f"history_images must have shape {expected_prefix}")
            if (
                history_images.device != current_images.device
                or history_images.dtype != current_images.dtype
            ):
                raise ValueError("History images must share current image device and dtype")
            if not torch.isfinite(history_images).all():
                raise ValueError("History images must be finite")
            valid_ages = history_ages_us.masked_select(history_mask)
            if valid_ages.numel() and (valid_ages > self.maximum_history_age_us).any():
                raise ValueError("History age exceeds the configured maximum")
            with torch.no_grad():
                flat = history_images.reshape(batch * 2, *history_images.shape[2:])
                history_state = self.backend.extract_projected_features(flat)
            history_features = tuple(
                feature.reshape(batch, 2, *feature.shape[1:])
                for feature in history_state.features
            )
            fused = self.fusion(
                current_state.features,
                history_features,
                history_ages_us,
                history_mask,
                history_limit=history_limit,
            )
        return self.backend.decode_projected_features(
            current_state,
            fused,
            calibs,
            targets,
            image_sizes,
            dn_args,
        )
