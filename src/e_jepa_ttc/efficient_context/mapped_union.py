"""Exact ROI projection once per union; unchanged per-window voxel reductions."""

import time

import numpy as np
import torch

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE, EAPEventReader
from e_jepa_ttc.data.eap_representation import event_voxel_with_scalars
from e_jepa_ttc.data.types import EventBatch


def map_roi(raw: dict, roi: tuple, size: int, offset: float) -> dict:
    """Preserve V4 float64 projection, sensor clipping and original event order."""
    if set(raw) != {"x", "y", "t", "p"} or size < 1:
        raise ValueError("raw sensor columns and positive ROI size required")
    x0, y0, x1, y1 = roi
    if not np.isfinite((*roi, offset)).all() or x0 >= x1 or y0 >= y1:
        raise ValueError("invalid ROI/offset")
    x = np.asarray(raw["x"], np.float64) + offset
    y = np.asarray(raw["y"], np.float64)
    t = np.asarray(raw["t"], np.int64)
    if any(v.ndim != 1 or v.shape != t.shape for v in (x, y, raw["p"])):
        raise ValueError("misaligned raw columns")
    if (np.diff(t) < 0).any():
        raise ValueError("timestamp rollback")
    width, height = EAP_IMAGE_SIZE
    keep = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
        & (x >= x0)
        & (x < x1)
        & (y >= y0)
        & (y < y1)
    )
    return {
        "x": np.clip(np.floor((x[keep] - x0) * size / (x1 - x0)).astype(np.int32), 0, size - 1),
        "y": np.clip(np.floor((y[keep] - y0) * size / (y1 - y0)).astype(np.int32), 0, size - 1),
        "t": t[keep],
        "p": np.where(raw["p"][keep] > 0, 1, -1).astype(np.int8),
    }


def encode_mapped(mapped: dict, start: int, end: int, size: int, sequence: str) -> torch.Tensor:
    """Use half-open searchsorted views and the canonical normalization per window."""
    if end <= start:
        raise ValueError("positive half-open interval required")
    lo, hi = np.searchsorted(mapped["t"], [start, end], side="left")
    batch = EventBatch(
        x=mapped["x"][lo:hi],
        y=mapped["y"][lo:hi],
        t_us=mapped["t"][lo:hi],
        polarity=mapped["p"][lo:hi],
        width=size,
        height=size,
        sequence_id=sequence,
        t_start_us=start,
        t_end_us=end,
    )
    return event_voxel_with_scalars(batch, bins_per_polarity=5)


def encode_union(
    reader: EAPEventReader,
    windows: np.ndarray,
    lags: np.ndarray,
    valid: np.ndarray,
    roi: tuple,
    *,
    sequence_id: str,
    roi_size: int,
    event_pixel_diff: float,
    retained_bytes_max: int = 256 * 1024**2,
    profile: dict | None = None,
) -> torch.Tensor:
    """Bounded union, exact mapping once, capacity fallback to historical path."""
    from e_jepa_ttc.simplex_t.context_raw_union import _read_crop, encode_context_union

    if windows.shape != (3, 2) or lags.shape != (16,) or valid.shape != (16,) or not valid[-1]:
        raise ValueError("canonical H16 layout and present required")
    intervals = windows[None] - lags[:, None, None]
    begin = time.perf_counter()
    raw = _read_crop(
        reader,
        int(intervals[valid].min()),
        int(intervals[valid].max()),
        roi,
        event_pixel_diff,
        retained_bytes_max,
    )
    read_end = time.perf_counter()
    if raw is None:
        if profile is not None:
            profile["fallback"] = True
        return encode_context_union(
            reader,
            windows,
            lags,
            valid,
            roi,
            sequence_id=sequence_id,
            roi_size=roi_size,
            event_pixel_diff=event_pixel_diff,
            retained_bytes_max=retained_bytes_max,
        )
    mapped = map_roi(raw, roi, roi_size, event_pixel_diff)
    map_end = time.perf_counter()
    del raw
    result = torch.zeros(16, 3, 12, roi_size, roi_size)
    for slot in np.flatnonzero(valid):
        for w, (start, end) in enumerate(intervals[slot]):
            result[slot, w] = encode_mapped(mapped, int(start), int(end), roi_size, sequence_id)
    if profile is not None:
        profile.update(
            read_crop_s=read_end - begin,
            projection_s=map_end - read_end,
            window_encode_s=time.perf_counter() - map_end,
            retained_bytes=sum(v.nbytes for v in mapped.values()),
            fallback=False,
        )
    return result
