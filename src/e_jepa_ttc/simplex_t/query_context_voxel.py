"""Fixed current-query crop using the historical V4 voxel/scalar preprocessing."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE
from e_jepa_ttc.data.eap_representation import event_voxel_with_scalars
from e_jepa_ttc.data.types import EventBatch


def encode_query_window(
    raw: dict[str, np.ndarray],
    *,
    square_xyxy: tuple[float, float, float, float],
    start_us: int,
    end_us: int,
    sequence_id: str,
    roi_size: int,
    bins_per_polarity: int,
    event_pixel_diff: float,
) -> torch.Tensor:
    """Preserve V4 mapping, polarity, normalization and count/rate semantics.

    Parameters must be taken from the frozen producer cache manifest. This
    function deliberately has no target/depth/velocity or tracking parameters.
    """
    if set(raw) != {"x", "y", "t", "p"}:
        raise ValueError("only raw sensor columns are accepted")
    x0, y0, x1, y1 = square_xyxy
    if (
        not np.isfinite(square_xyxy).all()
        or not np.isfinite(event_pixel_diff)
        or x0 >= x1
        or y0 >= y1
        or roi_size < 1
        or start_us >= end_us
    ):
        raise ValueError("invalid crop or window")
    event_x = np.asarray(raw["x"], dtype=np.float64) + event_pixel_diff
    event_y = np.asarray(raw["y"], dtype=np.float64)
    event_t = np.asarray(raw["t"], dtype=np.int64)
    event_p = np.asarray(raw["p"])
    if any(
        values.ndim != 1 or values.shape != event_t.shape for values in (event_x, event_y, event_p)
    ):
        raise ValueError("sensor column shape mismatch")
    if np.any(event_t < start_us) or np.any(event_t >= end_us):
        raise ValueError("out-of-window sensor dependency")
    width, height = EAP_IMAGE_SIZE
    valid = (
        (event_x >= 0)
        & (event_x < width)
        & (event_y >= 0)
        & (event_y < height)
        & (event_x >= x0)
        & (event_x < x1)
        & (event_y >= y0)
        & (event_y < y1)
    )
    mapped_x = np.floor((event_x[valid] - x0) * roi_size / (x1 - x0)).astype(np.int32)
    mapped_y = np.floor((event_y[valid] - y0) * roi_size / (y1 - y0)).astype(np.int32)
    batch = EventBatch(
        x=np.clip(mapped_x, 0, roi_size - 1),
        y=np.clip(mapped_y, 0, roi_size - 1),
        t_us=event_t[valid],
        polarity=np.where(event_p[valid] > 0, 1, -1).astype(np.int8),
        width=roi_size,
        height=roi_size,
        sequence_id=sequence_id,
        t_start_us=start_us,
        t_end_us=end_us,
    )
    return event_voxel_with_scalars(batch, bins_per_polarity=bins_per_polarity)
