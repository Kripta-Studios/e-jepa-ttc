"""Typed v4 input contract shared by cache builders, adapters and models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

INPUT_SCHEMA_VERSION = "garlttc_input_v4"
NORMALIZATION_ID = "official_timevolume20_grid_sample_v1"
EVENT_CHANNEL_NAMES = tuple(f"event_plane_{index:02d}" for index in range(20))


@dataclass(frozen=True)
class GarlTTCModelInput:
    """Model-only tensors; labels are intentionally not accepted here."""

    event_roi_endpoints: torch.Tensor
    endpoint_timestamps_us: torch.Tensor
    delta_t_s: torch.Tensor
    boxes_xyxy: torch.Tensor | None
    full_event_context: torch.Tensor | None
    rgb_endpoints: torch.Tensor | None
    input_valid: torch.Tensor
    protocol_id: str

    def validate(self) -> None:
        """Validate tensor roles and shapes before a model/device is allocated."""

        value = self.event_roi_endpoints
        if (
            value.ndim != 5
            or tuple(value.shape[1:3]) != (2, 20)
            or tuple(value.shape[-2:]) != (128, 128)
        ):
            raise ValueError("event_roi_endpoints must have shape [B,2,20,128,128].")
        batch = value.shape[0]
        if self.endpoint_timestamps_us.shape != (batch, 2):
            raise ValueError("endpoint_timestamps_us must have shape [B,2].")
        if self.delta_t_s.shape not in {(batch,), (batch, 1)}:
            raise ValueError("delta_t_s must have shape [B] or [B,1].")
        if self.input_valid.shape != (batch,):
            raise ValueError("input_valid must have shape [B].")
        if self.boxes_xyxy is not None and self.boxes_xyxy.shape != (batch, 2, 4):
            raise ValueError("boxes_xyxy must have shape [B,2,4].")
        if self.rgb_endpoints is not None and self.rgb_endpoints.shape != (batch, 2, 3, 128, 128):
            raise ValueError("rgb_endpoints must have shape [B,2,3,128,128].")
        if self.protocol_id == "":
            raise ValueError("protocol_id must be explicit.")


@dataclass(frozen=True)
class GarlTTCSupervision:
    """Training-only labels kept separate from :class:`GarlTTCModelInput`."""

    ttc_s: torch.Tensor
    visible_heights_px: torch.Tensor
    foreground_mask: torch.Tensor | None
    geometry_target: torch.Tensor | None
    category_index: torch.Tensor | None


def validate_input_schema(schema: dict[str, Any]) -> None:
    """Fail with an actionable message when a cache schema is incompatible."""

    missing = [
        key
        for key in ("version", "event_roi_shape", "channel_names", "normalization")
        if key not in schema
    ]
    if missing:
        raise ValueError(f"Garl input schema is missing fields: {missing}")
    if schema["version"] != INPUT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Garl input schema "
            f"{schema['version']!r}; expected {INPUT_SCHEMA_VERSION!r}."
        )
    if tuple(schema["event_roi_shape"]) != (2, 20, 128, 128):
        raise ValueError("Garl input schema event_roi_shape must be [2,20,128,128].")
    if tuple(schema["channel_names"]) != EVENT_CHANNEL_NAMES:
        raise ValueError("Garl input schema channel_names do not match official plane order.")
    if schema["normalization"] != NORMALIZATION_ID:
        raise ValueError("Garl input schema normalization does not match the official adapter.")


def validate_cache_manifest_input_schema(manifest: dict[str, Any]) -> None:
    """Validate the manifest before any checkpoint or GPU work is started."""

    schema = manifest.get("input_schema")
    if not isinstance(schema, dict):
        raise ValueError("Cache manifest has no input_schema; rebuild it as cache v4.")
    validate_input_schema(schema)


__all__ = [
    "EVENT_CHANNEL_NAMES",
    "GarlTTCModelInput",
    "GarlTTCSupervision",
    "INPUT_SCHEMA_VERSION",
    "NORMALIZATION_ID",
    "validate_cache_manifest_input_schema",
    "validate_input_schema",
]
