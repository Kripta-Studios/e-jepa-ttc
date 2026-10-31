"""Stateless composition of causal RGB history and an explicit event sensor input."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from e_jepa_ttc.detection.event_fusion import (
    CausalEventFeatureFusion,
    EventFusionControl,
    EventFusionObservables,
)
from e_jepa_ttc.detection.temporal_detector import (
    ProjectedFeatureBackend,
    TemporalBatchContext,
    TemporalDetectorAdapter,
)
from e_jepa_ttc.detection.temporal_fusion import CausalTemporalFeatureFusion


class EventTemporalDetectorAdapter(TemporalDetectorAdapter):
    """Fuse explicit event tensors after RGB history and before the original decoder."""

    def __init__(
        self,
        backend: ProjectedFeatureBackend,
        fusion: CausalTemporalFeatureFusion,
        event_fusion: CausalEventFeatureFusion,
        *,
        producer_version: str,
        maximum_history_age_us: float,
    ) -> None:
        super().__init__(
            backend,
            fusion,
            producer_version=producer_version,
            maximum_history_age_us=maximum_history_age_us,
        )
        self.event_fusion = event_fusion

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
        event_control: EventFusionControl = "F00",
        event_voxel: torch.Tensor | None = None,
        event_tokens: tuple[str, ...] | None = None,
        event_mask: torch.Tensor | None = None,
        event_observables: EventFusionObservables | None = None,
    ) -> dict[str, Any]:
        """Bind events to current identities; never store request state on the module."""
        if event_control not in ("F00", "F10", "F01", "F11"):
            raise ValueError("Unknown sensor control")
        if (event_control in ("F10", "F11")) != (history_limit > 0):
            raise ValueError("Sensor control and RGB history setting disagree")
        if event_control in ("F01", "F11"):
            expected = tuple(item.frame_id for item in temporal_context.current)
            if event_tokens != expected:
                raise ValueError("Event tokens do not match current RGB identities")
            if event_voxel is None or event_mask is None or event_observables is None:
                raise ValueError("Event controls require explicit tensors and availability")
        elif event_tokens is not None:
            raise ValueError("RGB controls must not receive event tokens")
        state, features = self.project_temporal(
            current_images,
            temporal_context=temporal_context,
            history_images=history_images,
            history_ages_us=history_ages_us,
            history_mask=history_mask,
            history_limit=history_limit,
        )
        fused = self.event_fusion(
            features,
            control=event_control,
            event_voxel=event_voxel,
            event_mask=event_mask,
            observables=event_observables,
        )
        return self.backend.decode_projected_features(
            state, fused, calibs, targets, image_sizes, dn_args
        )
