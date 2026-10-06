"""Exact Object Event v4 collator with a proven constant-check fast path."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from e_jepa_ttc.data.event_v4_geometry import EVENT_V4_STEPS
from e_jepa_ttc.data.object_event_v4 import (
    OBSERVABLE_MOTION_DIM,
    ObjectEventV4Batch,
    _tensor,
)

# PyTorch 2.11 CPU std uses Welford with double accumulation and casts its
# result to scalar_t. This source contract is deliberately version-bound:
# https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp
_PROVEN_TORCH_VERSION = "2.11.0"


def _all_event_rows_constant(
    events: torch.Tensor,
    *,
    _std_for_test: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> bool:
    """Return the original all-rows-constant decision, skipping proven-positive std.

    For a finite row of ``N`` values with range ``r``, sample SSE is at least
    ``r**2 / 2`` and sample std is at least ``r / sqrt(2*(N-1))``. A range above
    the conservative threshold below therefore guarantees that the PyTorch 2.11
    double-Welford result remains positive after its FP32 output cast. All other
    cases execute the original ``Tensor.std`` expression unchanged.
    """

    flat = events.flatten(0, 1).flatten(1)
    if (
        events.device.type == "cpu"
        and events.dtype == torch.float32
        and flat.shape[1] > 1
        and str(torch.__version__).split("+", 1)[0] == _PROVEN_TORCH_VERSION
    ):
        minimum, maximum = torch.aminmax(flat, dim=1)
        threshold = (
            4.0
            * torch.finfo(torch.float32).tiny
            * math.sqrt(2.0 * (flat.shape[1] - 1))
        )
        if bool(((maximum - minimum) > threshold).any()):
            return False
    channel_std = (
        flat.std(dim=1) if _std_for_test is None else _std_for_test(flat)
    )
    return bool((channel_std <= 0).all())


def collate_object_event_v4(records: list[dict[str, Any]]) -> ObjectEventV4Batch:
    if not records:
        raise ValueError("Object Event v4 collate received an empty batch")
    events = torch.stack([_tensor(record, "event_v4_common_roi") for record in records])
    expected_channels = {
        int(record.get("_event_v4_expected_channels", 12)) for record in records
    }
    expected_steps = {
        int(record.get("_event_v4_expected_steps", EVENT_V4_STEPS)) for record in records
    }
    if len(expected_channels) != 1 or len(expected_steps) != 1:
        raise ValueError("Object Event batch mixes temporal channel/step schemas")
    channel_count = expected_channels.pop()
    step_count = expected_steps.pop()
    if step_count not in {2, EVENT_V4_STEPS}:
        raise ValueError("Object Event batches support only 2-step V8 or historical 3-step inputs")
    expected_prefix = (len(records), step_count, channel_count)
    if events.ndim != 5 or events.shape[:3] != expected_prefix:
        raise ValueError(
            f"event_v4_common_roi must collate to [B,{step_count},{channel_count},H,W], "
            f"got {tuple(events.shape)}"
        )
    if events.shape[-1] != events.shape[-2]:
        raise ValueError("V4 common ROI must be square")
    if not torch.isfinite(events).all():
        raise ValueError("V4 events contain non-finite values")
    if _all_event_rows_constant(events):
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
    if boxes.shape != (len(records), step_count, 4):
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

    teacher_presence = ["sam_teacher_masks" in record for record in records]
    if any(teacher_presence) and not all(teacher_presence):
        raise ValueError("SAM teacher fields must be present for the entire batch")
    teacher_masks = None
    teacher_valid = None
    if all(teacher_presence):
        teacher_masks = torch.stack(
            [_tensor(record, "sam_teacher_masks") for record in records]
        )
        teacher_valid = torch.stack(
            [
                torch.as_tensor(record["sam_teacher_mask_valid"], dtype=torch.bool)
                for record in records
            ]
        )
        expected_masks = (len(records), 2, 1, events.shape[-2], events.shape[-1])
        if teacher_masks.shape != expected_masks:
            raise ValueError(
                "SAM teacher masks must have shape "
                f"{expected_masks}, got {tuple(teacher_masks.shape)}"
            )
        if teacher_valid.shape != (len(records), 2):
            raise ValueError("SAM teacher validity must have shape [B,2]")

    dino_presence = ["dinov3_relation_targets" in record for record in records]
    if any(dino_presence) and not all(dino_presence):
        raise ValueError("DINO teacher fields must be present for the entire batch")
    dino_targets = None
    dino_valid = None
    if all(dino_presence):
        dino_targets = torch.stack(
            [_tensor(record, "dinov3_relation_targets") for record in records]
        )
        dino_valid = torch.stack(
            [
                torch.as_tensor(record["dinov3_relation_valid"], dtype=torch.bool)
                for record in records
            ]
        )
        if dino_targets.shape != dino_valid.shape:
            raise ValueError(
                "DINO teacher targets and valid must share shape, "
                f"got {tuple(dino_targets.shape)} and {tuple(dino_valid.shape)}"
            )
        if (
            "dinov3_relation_valid" not in records[0]
        ):
            raise ValueError("DINO targets and valid must appear together")

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
        sam_teacher_masks=teacher_masks,
        sam_teacher_mask_valid=teacher_valid,
        dinov3_relation_targets=dino_targets,
        dinov3_relation_valid=dino_valid,
    )


__all__ = ["collate_object_event_v4"]
