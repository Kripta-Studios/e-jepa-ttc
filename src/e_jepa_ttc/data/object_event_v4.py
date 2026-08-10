"""Typed three-step common-ROI batches for Object Event TTC v4.

The model receives a causal event sequence t0/t1/t2 in one shared coordinate
frame, the measured t1->t2 interval, and an optional observable-motion branch.
Official TTC, visible heights, boxes in the crop and source crop coordinates are
kept as supervision/diagnostics and are never returned by ``event_inputs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from e_jepa_ttc.data.event_v4_geometry import (
    EVENT_V4_CHANNEL_COUNT,
    EVENT_V4_STEPS,
)

OBSERVABLE_MOTION_DIM = 18


@dataclass
class ObjectEventV4Batch:
    events: torch.Tensor
    delta_t_s: torch.Tensor
    observable_motion: torch.Tensor
    visible_heights_px: torch.Tensor
    target_ttc_s: torch.Tensor
    boxes_xyxy: torch.Tensor
    common_square_xyxy: torch.Tensor
    sequence_ids: list[str]
    sample_tokens: list[str]
    track_ids: list[str]

    def to(
        self,
        device: torch.device,
        *,
        non_blocking: bool = True,
    ) -> ObjectEventV4Batch:
        return ObjectEventV4Batch(
            events=self.events.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            delta_t_s=self.delta_t_s.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            observable_motion=self.observable_motion.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            visible_heights_px=self.visible_heights_px.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            target_ttc_s=self.target_ttc_s.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            boxes_xyxy=self.boxes_xyxy.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            common_square_xyxy=self.common_square_xyxy.to(
                device=device, dtype=torch.float32, non_blocking=non_blocking
            ),
            sequence_ids=self.sequence_ids,
            sample_tokens=self.sample_tokens,
            track_ids=self.track_ids,
        )

    def event_inputs(self) -> dict[str, torch.Tensor]:
        """Return the strictly event-only input mapping."""

        return {
            "events": self.events,
            "delta_t_s": self.delta_t_s,
        }

    def model_inputs(self) -> dict[str, torch.Tensor]:
        """Return v4 inference inputs; no labels, heights or crop boxes leak."""

        return {
            "events": self.events,
            "delta_t_s": self.delta_t_s,
            "observable_motion": self.observable_motion,
        }


@dataclass(frozen=True)
class BoxGeometryTargets:
    """Training-only visible bbox geometry in the cached common ROI frame."""

    height_normalized: torch.Tensor
    width_normalized: torch.Tensor
    centroid_x_normalized: torch.Tensor
    centroid_y_normalized: torch.Tensor
    valid: torch.Tensor


class GarlTTCObjectEventV4Dataset(Dataset[dict[str, Any]]):
    """Lazy view over the v4 common-coordinate cache extension."""

    def __init__(
        self,
        manifest_path: str,
        *,
        splits: tuple[str, ...],
        event_field: str = "event_v4_common_roi",
    ) -> None:
        from e_jepa_ttc.data.garlttc_lhr_cache import GarlTTCLHRCacheDataset

        self._dataset = GarlTTCLHRCacheDataset(manifest_path, splits=splits)
        self.manifest = self._dataset.manifest
        self.event_field = event_field
        fields = set(self.manifest.get("model_input_fields", []))
        required = {
            event_field,
            "event_v4_boxes_xyxy",
            "event_v4_common_square_xyxy",
            "event_v4_precontext_valid",
            "garl_delta_t_s",
            "observable_motion",
        }
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Cache lacks Object Event v4 fields: {missing}")
        extension = self.manifest.get("object_lhr_extension")
        if not isinstance(extension, dict) or int(extension.get("version", 0)) < 3:
            raise ValueError("Object Event v4 requires object_lhr_extension >= 3")
        shape = extension.get("event_v4_common_roi_shape")
        expected = [EVENT_V4_STEPS, EVENT_V4_CHANNEL_COUNT]
        if not isinstance(shape, list) or shape[:2] != expected:
            raise ValueError(
                f"Unexpected event_v4_common_roi_shape={shape!r}; expected prefix {expected}"
            )
        if extension.get("event_v4_independent_endpoint_resize") is not False:
            raise ValueError("V4 rejects independently resized endpoint ROIs")
        if extension.get("event_v4_preserves_absolute_scale_inside_common_roi") is not True:
            raise ValueError("V4 cache does not declare preserved scale")
        valid_fraction = float(
            self.manifest.get("event_v4_precontext_valid_fraction", 0.0)
        )
        if valid_fraction < 0.80:
            raise ValueError(
                "V4 requires real causal event t0 coverage >= 0.80, "
                f"manifest reports {valid_fraction:.6f}"
            )

    def __len__(self) -> int:
        return len(self._dataset)

    def shard_index_groups(self) -> tuple[tuple[int, ...], ...]:
        """Expose cache-local groups for I/O-efficient deterministic sampling."""

        return self._dataset.shard_index_groups()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._dataset[index]
        if not bool(record.get("event_v4_precontext_valid", False)):
            raise ValueError("V4 record is missing a valid t0 endpoint")
        return record


def _tensor(record: dict[str, Any], key: str) -> torch.Tensor:
    return torch.as_tensor(record[key], dtype=torch.float32)


def collate_object_event_v4(records: list[dict[str, Any]]) -> ObjectEventV4Batch:
    if not records:
        raise ValueError("Object Event v4 collate received an empty batch")
    events = torch.stack([_tensor(record, "event_v4_common_roi") for record in records])
    expected_prefix = (len(records), EVENT_V4_STEPS, EVENT_V4_CHANNEL_COUNT)
    if events.ndim != 5 or events.shape[:3] != expected_prefix:
        raise ValueError(
            "event_v4_common_roi must collate to [B,3,12,H,W], "
            f"got {tuple(events.shape)}"
        )
    if events.shape[-1] != events.shape[-2]:
        raise ValueError("V4 common ROI must be square")
    if not torch.isfinite(events).all():
        raise ValueError("V4 events contain non-finite values")
    channel_std = events.flatten(0, 1).flatten(1).std(dim=1)
    if bool((channel_std <= 0).all()):
        raise ValueError("Every v4 event channel is constant")

    delta_t = torch.tensor(
        [float(record["garl_delta_t_s"]) for record in records],
        dtype=torch.float32,
    )
    motion = torch.stack([_tensor(record, "observable_motion") for record in records])
    heights = torch.stack([_tensor(record, "garl_visible_heights_px") for record in records])
    targets = torch.tensor(
        [float(record["ttc_s"]) for record in records], dtype=torch.float32
    )
    boxes = torch.stack([_tensor(record, "event_v4_boxes_xyxy") for record in records])
    squares = torch.stack(
        [_tensor(record, "event_v4_common_square_xyxy") for record in records]
    )
    if motion.shape != (len(records), OBSERVABLE_MOTION_DIM):
        raise ValueError(f"observable_motion has invalid shape {tuple(motion.shape)}")
    if heights.shape != (len(records), 2):
        raise ValueError(f"garl_visible_heights_px has invalid shape {tuple(heights.shape)}")
    if boxes.shape != (len(records), EVENT_V4_STEPS, 4):
        raise ValueError(f"event_v4_boxes_xyxy has invalid shape {tuple(boxes.shape)}")
    if squares.shape != (len(records), 4):
        raise ValueError(
            f"event_v4_common_square_xyxy has invalid shape {tuple(squares.shape)}"
        )
    if bool((delta_t <= 0).any()) or not torch.isfinite(delta_t).all():
        raise ValueError("delta_t_s must be finite and positive")
    if bool((targets == 0).any()) or not torch.isfinite(targets).all():
        raise ValueError("Official TTC targets must be finite and non-zero")
    if bool((heights <= 0).any()) or not torch.isfinite(heights).all():
        raise ValueError("Visible-height targets must be finite and positive")

    return ObjectEventV4Batch(
        events=events,
        delta_t_s=delta_t,
        observable_motion=motion,
        visible_heights_px=heights,
        target_ttc_s=targets,
        boxes_xyxy=boxes,
        common_square_xyxy=squares,
        sequence_ids=[str(record["sequence_id"]) for record in records],
        sample_tokens=[str(record["sample_token"]) for record in records],
        track_ids=[str(record["track_id"]) for record in records],
    )


def weak_box_masks(
    boxes_xyxy: torch.Tensor,
    *,
    height: int,
    width: int,
    endpoint_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize disclosed weak bbox supervision in the cached ROI frame.

    The returned rectangles are training targets only.  They must never enter the
    estimator forward path and must not be described as segmentation ground truth.
    Floating point boxes use half-open pixel-centre membership, which avoids
    nondeterministic rounding while remaining differentiable only through the
    estimator (not the target construction).
    """

    if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must have shape [B,T,4]")
    if min(height, width) <= 0:
        raise ValueError("mask dimensions must be positive")
    if not torch.isfinite(boxes_xyxy).all():
        raise ValueError("boxes_xyxy must be finite")
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    geometric_valid = (x2 > x1) & (y2 > y1)
    if endpoint_valid is not None:
        if endpoint_valid.shape != geometric_valid.shape:
            raise ValueError("endpoint_valid must have shape [B,T]")
        geometric_valid = geometric_valid & endpoint_valid.bool()
    xs = torch.arange(width, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype) + 0.5
    ys = torch.arange(height, device=boxes_xyxy.device, dtype=boxes_xyxy.dtype) + 0.5
    inside_x = (xs >= x1[..., None]) & (xs < x2[..., None])
    inside_y = (ys >= y1[..., None]) & (ys < y2[..., None])
    masks = inside_y[..., :, None] & inside_x[..., None, :]
    masks = masks.unsqueeze(-3) & geometric_valid[..., None, None, None]
    return masks.to(dtype=torch.float32), geometric_valid


def box_geometry_targets(
    boxes_xyxy: torch.Tensor,
    *,
    height: int,
    width: int,
    endpoint_valid: torch.Tensor | None = None,
) -> BoxGeometryTargets:
    """Derive visible height, width and centre without rasterizing a dense target.

    Boxes are clipped to the common ROI because the event model cannot observe
    geometry outside that crop. These values are supervision only and must never
    be passed to ``CausalScaleTTC.forward``.
    """

    if boxes_xyxy.ndim != 3 or boxes_xyxy.shape[-1] != 4:
        raise ValueError("boxes_xyxy must have shape [B,T,4]")
    if min(height, width) <= 0:
        raise ValueError("geometry dimensions must be positive")
    if not torch.isfinite(boxes_xyxy).all():
        raise ValueError("boxes_xyxy must be finite")
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    x1 = x1.clamp(0.0, float(width))
    x2 = x2.clamp(0.0, float(width))
    y1 = y1.clamp(0.0, float(height))
    y2 = y2.clamp(0.0, float(height))
    target_width = x2 - x1
    target_height = y2 - y1
    valid = (target_width > 0.0) & (target_height > 0.0)
    if endpoint_valid is not None:
        if endpoint_valid.shape != valid.shape:
            raise ValueError("endpoint_valid must have shape [B,T]")
        valid = valid & endpoint_valid.bool()
    zero = torch.zeros_like(target_width)
    return BoxGeometryTargets(
        height_normalized=torch.where(
            valid, target_height / float(height), zero
        ),
        width_normalized=torch.where(valid, target_width / float(width), zero),
        centroid_x_normalized=torch.where(
            valid, 0.5 * (x1 + x2) / float(width), zero
        ),
        centroid_y_normalized=torch.where(
            valid, 0.5 * (y1 + y2) / float(height), zero
        ),
        valid=valid,
    )


__all__ = [
    "BoxGeometryTargets",
    "GarlTTCObjectEventV4Dataset",
    "OBSERVABLE_MOTION_DIM",
    "ObjectEventV4Batch",
    "collate_object_event_v4",
    "box_geometry_targets",
    "weak_box_masks",
]
