"""Geometry contract for the v4 event-only TTC cache.

The v3 cache resized every endpoint from its own square bounding box.  That
operation made a small object at t1 and a large object at t2 occupy nearly the
same normalized canvas, removing the visual expansion signal.  V4 instead uses
one common square for t0/t1/t2 and stores only the twelve non-constant channels
of the existing 21-channel event representation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

EVENT_V4_ACTIVE_CHANNELS: tuple[int, ...] = tuple(range(12))
EVENT_V4_CHANNEL_NAMES: tuple[str, ...] = (
    "positive_voxel_bin_0",
    "positive_voxel_bin_1",
    "positive_voxel_bin_2",
    "positive_voxel_bin_3",
    "positive_voxel_bin_4",
    "negative_voxel_bin_0",
    "negative_voxel_bin_1",
    "negative_voxel_bin_2",
    "negative_voxel_bin_3",
    "negative_voxel_bin_4",
    "event_count_log1p",
    "event_rate_log1p",
)
EVENT_V4_CHANNEL_COUNT = len(EVENT_V4_ACTIVE_CHANNELS)
EVENT_V4_STEPS = 3


def event_v4_channel_count(bins_per_polarity: int) -> int:
    """Return voxel plus count/rate channels for one V4 endpoint."""

    if bins_per_polarity <= 0:
        raise ValueError("bins_per_polarity must be positive")
    return 2 * bins_per_polarity + 2


def event_v4_channel_names(bins_per_polarity: int) -> tuple[str, ...]:
    """Build the declared channel schema for a V4 temporal resolution."""

    event_v4_channel_count(bins_per_polarity)
    return (
        *(f"positive_voxel_bin_{index}" for index in range(bins_per_polarity)),
        *(f"negative_voxel_bin_{index}" for index in range(bins_per_polarity)),
        "event_count_log1p",
        "event_rate_log1p",
    )


def _finite_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError("A box must contain exactly four xyxy coordinates")
    x0, y0, x1, y1 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ValueError("Box coordinates must be finite")
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Degenerate xyxy box: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def common_square_from_boxes(
    boxes: Sequence[Sequence[float]],
    indices: Sequence[int],
    *,
    margin_fraction: float = 0.25,
    minimum_edge: float = 8.0,
) -> tuple[float, float, float, float]:
    """Return one square coordinate frame shared by every requested endpoint.

    The square is intentionally allowed to extend outside the source image;
    downstream voxelization simply leaves those regions empty.  This preserves
    object scale and translation rather than silently clipping the context.
    """

    if len(indices) < 2:
        raise ValueError("A common temporal ROI requires at least two indices")
    if margin_fraction < 0.0:
        raise ValueError("margin_fraction must be non-negative")
    if minimum_edge <= 0.0:
        raise ValueError("minimum_edge must be positive")
    selected = [_finite_box(boxes[index]) for index in indices]
    union_x0 = min(box[0] for box in selected)
    union_y0 = min(box[1] for box in selected)
    union_x1 = max(box[2] for box in selected)
    union_y1 = max(box[3] for box in selected)
    center_x = 0.5 * (union_x0 + union_x1)
    center_y = 0.5 * (union_y0 + union_y1)
    edge = max(union_x1 - union_x0, union_y1 - union_y0, minimum_edge)
    edge *= 1.0 + 2.0 * margin_fraction
    half = 0.5 * edge
    return center_x - half, center_y - half, center_x + half, center_y + half


def box_in_common_roi(
    box: Sequence[float],
    common_square: Sequence[float],
    *,
    roi_size: int,
) -> np.ndarray:
    """Map a source-image box into the common ROI without changing its scale."""

    if roi_size <= 0:
        raise ValueError("roi_size must be positive")
    x0, y0, x1, y1 = _finite_box(box)
    sx0, sy0, sx1, sy1 = _finite_box(common_square)
    scale_x = roi_size / (sx1 - sx0)
    scale_y = roi_size / (sy1 - sy0)
    return np.asarray(
        [
            (x0 - sx0) * scale_x,
            (y0 - sy0) * scale_y,
            (x1 - sx0) * scale_x,
            (y1 - sy0) * scale_y,
        ],
        dtype=np.float32,
    )


def shifted_precontext_window(
    first_window: Sequence[int],
    *,
    shift_s: float,
) -> tuple[int, int]:
    """Shift an observed event window causally backwards without fabricating events.

    GarlTTC rows expose the labelled t1/t2 endpoint windows but do not include
    an earlier annotated frame in the same row.  V4 still needs a real event
    context t0.  The cache therefore reads the same-duration event interval
    exactly ``shift_s`` before the t1 window.  Only the crop box is reused from
    t1 for diagnostics; the event tensor itself comes from the earlier interval.
    """

    if len(first_window) != 2:
        raise ValueError("An event window must contain [start_us, end_us]")
    start_us, end_us = (int(value) for value in first_window)
    if end_us <= start_us:
        raise ValueError("Event window must have positive duration")
    if not math.isfinite(float(shift_s)) or shift_s <= 0.0:
        raise ValueError("shift_s must be finite and positive")
    duration_us = end_us - start_us
    shift_us = max(
        int(round(float(shift_s) * 1_000_000.0)),
        duration_us,
    )
    shifted = (start_us - shift_us, end_us - shift_us)
    if shifted[0] < 0 or shifted[1] <= shifted[0]:
        raise ValueError("Shifted causal precontext lies before the event timeline")
    if shifted[1] > start_us:
        raise ValueError("Shifted precontext must end no later than t1 starts")
    return shifted

def select_active_event_channels(
    value: torch.Tensor,
    *,
    bins_per_polarity: int = 5,
) -> torch.Tensor:
    """Drop the nine guaranteed-zero compatibility channels from a voxel."""

    channel_count = event_v4_channel_count(bins_per_polarity)
    if value.ndim != 3 or value.shape[0] < channel_count:
        raise ValueError(
            "Expected a [C,H,W] voxel with at least twelve active channels, "
            f"got {tuple(value.shape)}"
        )
    selected = value[:channel_count]
    if selected.shape[0] != channel_count:
        raise RuntimeError("Active event-channel selection changed unexpectedly")
    return selected.contiguous()


__all__ = [
    "EVENT_V4_ACTIVE_CHANNELS",
    "EVENT_V4_CHANNEL_COUNT",
    "EVENT_V4_CHANNEL_NAMES",
    "EVENT_V4_STEPS",
    "box_in_common_roi",
    "common_square_from_boxes",
    "event_v4_channel_count",
    "event_v4_channel_names",
    "select_active_event_channels",
    "shifted_precontext_window",
]
