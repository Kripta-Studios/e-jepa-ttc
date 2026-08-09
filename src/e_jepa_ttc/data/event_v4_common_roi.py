"""Pure, label-free common-ROI representation used by the v4.31 materializer.

This module intentionally does not know about parquet rows, targets, or cache paths.
It is kept separate from the v4.30 cache writer until an independent fixture proves
bit-exact integration with that historical writer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes, select_active_event_channels
from e_jepa_ttc.data.garlttc_lhr_cache import _jepa_roi_voxel


@dataclass(frozen=True)
class CommonROIConfig:
    """Locked v4.30-compatible raster parameters."""

    size: int = 128
    bins: int = 5
    margin_fraction: float = 0.25
    minimum_edge: float = 8.0
    event_pixel_diff: int = 5


DEFAULT_CONFIG = CommonROIConfig()


def common_square(
    boxes: Sequence[Sequence[float]], config: CommonROIConfig = DEFAULT_CONFIG
) -> tuple[float, float, float, float]:
    """Return the unclipped square union of finite boxes, with the locked margin."""
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("boxes must be finite [N,4] xyxy values")
    x0, y0 = values[:, 0].min(), values[:, 1].min()
    x1, y1 = values[:, 2].max(), values[:, 3].max()
    if x1 <= x0 or y1 <= y0:
        raise ValueError("boxes must have positive union extent")
    edge = max(float(x1 - x0), float(y1 - y0), config.minimum_edge)
    edge *= 1.0 + 2.0 * config.margin_fraction
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = edge / 2.0
    return (float(cx - half), float(cy - half), float(cx + half), float(cy + half))


def _as_events(
    events: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        x = np.asarray(events["x"], dtype=np.float64)
        y = np.asarray(events["y"], dtype=np.float64)
        t = np.asarray(events["t"], dtype=np.int64)
        p = np.asarray(events["p"])
    except KeyError as exc:
        raise ValueError("events require x,y,t,p") from exc
    if not (len(x) == len(y) == len(t) == len(p)) or np.any(np.diff(t) < 0):
        raise ValueError("event arrays must be equal-length and monotonic")
    return x, y, t, p


def rasterize_common_roi(
    events: Mapping[str, np.ndarray],
    square_xyxy: Sequence[float],
    *,
    start_us: int,
    end_us: int,
    config: CommonROIConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Rasterize one absolute half-open interval into 10 voxel and 2 scalar channels.

    Channels 0:10 are polarity-separated, linearly interpolated temporal bins;
    10 is ``log1p(event_count)`` and 11 is the corresponding per-second rate.
    Scalar channels are spatial constants by contract.
    """
    if end_us <= start_us:
        raise ValueError("interval must satisfy start_us < end_us")
    sx0, sy0, sx1, sy1 = (float(v) for v in square_xyxy)
    if not np.isfinite([sx0, sy0, sx1, sy1]).all() or sx1 <= sx0 or sy1 <= sy0:
        raise ValueError("invalid common square")
    x, y, t, p = _as_events(events)
    selected = (t >= start_us) & (t < end_us)
    out = np.zeros((12, config.size, config.size), dtype=np.float32)
    if not selected.any():
        return torch.from_numpy(out)
    x = x[selected] + config.event_pixel_diff
    y, t, p = y[selected], t[selected], p[selected]
    gx = (x - sx0) * config.size / (sx1 - sx0)
    gy = (y - sy0) * config.size / (sy1 - sy0)
    valid = (gx >= 0) & (gx < config.size) & (gy >= 0) & (gy < config.size)
    gx, gy, t, p = gx[valid], gy[valid], t[valid], p[valid]
    ix, iy = gx.astype(np.int64), gy.astype(np.int64)
    # v4 uses linear temporal bin interpolation and separate polarities.
    pos = p > 0
    coordinate = (t.astype(np.float64) - start_us) / (end_us - start_us) * (config.bins - 1)
    lo = np.floor(coordinate).astype(np.int64).clip(0, config.bins - 1)
    hi = np.minimum(lo + 1, config.bins - 1)
    high_weight = coordinate - lo
    for bin_index, weight in ((lo, 1.0 - high_weight), (hi, high_weight)):
        channel = bin_index + np.where(pos, 0, config.bins)
        np.add.at(out, (channel, iy, ix), weight.astype(np.float32))
    # robust per-window normalization of voxel channels, without touching scalars.
    scale = float(np.percentile(np.abs(out[:10]), 99.0))
    if scale > 0.0:
        out[:10] /= scale
    count = float(len(t))
    out[10].fill(np.log1p(count))
    out[11].fill(np.log1p(count / max((end_us - start_us) / 1_000_000.0, 1e-9)))
    return torch.from_numpy(out)


def trajectory_common_roi(
    events: Mapping[str, np.ndarray],
    *,
    t0: tuple[int, int],
    t1: tuple[int, int],
    t2: tuple[int, int],
    box_t1: Sequence[float],
    box_t2: Sequence[float],
    config: CommonROIConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Build the locked [t0,t1,t2,12,128,128] trajectory; t0 uses t1's proxy box."""
    square = common_square_from_boxes(
        (box_t1, box_t2),
        (0, 1),
        margin_fraction=config.margin_fraction,
        minimum_edge=config.minimum_edge,
    )
    frames = []
    for start_us, end_us in (t0, t1, t2):
        timestamps = np.asarray(events["t"], dtype=np.int64)
        keep = (timestamps >= start_us) & (timestamps < end_us)
        interval_events = {key: np.asarray(value)[keep] for key, value in events.items()}
        value = _jepa_roi_voxel(
            interval_events,
            square,
            size=config.size,
            bins=config.bins,
            sequence_id="v4_31",
            start_us=start_us,
            end_us=end_us,
            event_pixel_diff=config.event_pixel_diff,
        )
        frames.append(select_active_event_channels(value))
    return torch.stack(frames)
