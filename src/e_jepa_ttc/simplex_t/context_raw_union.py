"""Bounded input-only union reads; candidate I/O optimization, not yet cache default."""

from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE, EAPEventReader

from .query_context_voxel import encode_query_window


def encode_context_union(
    reader: EAPEventReader,
    base_windows_us: np.ndarray,
    lag_us: np.ndarray,
    valid: np.ndarray,
    square_xyxy: tuple[float, float, float, float],
    *,
    sequence_id: str,
    roi_size: int,
    event_pixel_diff: float,
    retained_bytes_max: int = 256 * 1024**2,
) -> torch.Tensor:
    """Read the union in250k chunks, retaining only the exact supplied crop.

    Spatial rejection is identical to encode_query_window and uses no labels.
    Input ordering and half-open time boundaries remain unchanged. Exceeding
    the absolute retained buffer cap raises before concatenation; no fallback
    expert, representation or history is substituted.
    """
    if base_windows_us.shape != (3, 2) or lag_us.shape != (16,) or valid.shape != (16,):
        raise ValueError("registered H16 window layout required")
    if not valid[-1] or retained_bytes_max < 1:
        raise ValueError("current context and positive buffer cap required")
    intervals = base_windows_us[None] - lag_us[:, None, None]
    start, end = int(intervals[valid].min()), int(intervals[valid].max())
    pieces: dict[str, list[np.ndarray]] = {key: [] for key in ("x", "y", "t", "p")}
    retained = 0
    x0, y0, x1, y1 = square_xyxy
    width, height = EAP_IMAGE_SIZE
    for chunk in reader.iter_window_chunks(start, end, chunk_events=250_000):
        x = np.asarray(chunk["x"], np.float64) + event_pixel_diff
        y = np.asarray(chunk["y"], np.float64)
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
        selected = {key: values[keep] for key, values in chunk.items()}
        retained += sum(values.nbytes for values in selected.values())
        if retained > retained_bytes_max:
            raise RuntimeError("RESOURCE_PAUSE:query union retained byte cap")
        for key, values in selected.items():
            pieces[key].append(values)
    raw = {
        key: np.concatenate(values) if values else np.empty(0, dtype=dtype)
        for (key, values), dtype in zip(
            pieces.items(), (np.int32, np.int32, np.int64, np.int8), strict=True
        )
    }
    if (np.diff(raw["t"]) < 0).any():
        raise ValueError("nonchronological raw union")
    result = torch.zeros(16, 3, 12, roi_size, roi_size)
    for slot in np.flatnonzero(valid):
        for window, (first, last) in enumerate(intervals[slot]):
            lo, hi = np.searchsorted(raw["t"], [first, last], side="left")
            result[slot, window] = encode_query_window(
                {key: values[lo:hi] for key, values in raw.items()},
                square_xyxy=square_xyxy,
                start_us=int(first),
                end_us=int(last),
                sequence_id=sequence_id,
                roi_size=roi_size,
                bins_per_polarity=5,
                event_pixel_diff=event_pixel_diff,
            )
    return result
