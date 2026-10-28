"""Exact CPU encoding primitives for the additive TRAIN40 runtime.

The original scientific encoders remain the oracle. Integer-coordinate tables
evaluate their float64 projection formula once per sensor coordinate; temporal
tables preserve the two original float64 ``bincount`` reductions and quantile.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from e_jepa_ttc.data.eap import EAP_IMAGE_SIZE
from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from e_jepa_ttc.representations.voxel_grid import robust_normalize
from e_jepa_ttc.rgb_port.data import RGB_SIZE, _effective_square


def map_roi_lut(
    raw: Mapping[str, np.ndarray],
    roi: tuple[float, float, float, float],
    size: int,
    offset: float = 0.0,
) -> dict[str, np.ndarray]:
    """Project integer sensor coordinates with the unchanged float64 formula.

    Arbitrary offsets and noninteger coordinates retain the canonical path.
    Invalid sensor coordinates are rejected before lookup, including negative
    values which would otherwise index the table from its end.
    """
    if set(raw) != {"x", "y", "t", "p"} or size < 1:
        raise ValueError("raw sensor columns and positive ROI size required")
    x0, y0, x1, y1 = roi
    if not np.isfinite((*roi, offset)).all() or x0 >= x1 or y0 >= y1:
        raise ValueError("invalid ROI/offset")
    x, y, t, p = (np.asarray(raw[key]) for key in ("x", "y", "t", "p"))
    if any(value.ndim != 1 or value.shape != t.shape for value in (x, y, p)):
        raise ValueError("misaligned raw columns")
    t = np.asarray(t, dtype=np.int64)
    if (np.diff(t) < 0).any():
        raise ValueError("timestamp rollback")
    if offset != 0.0 or x.dtype.kind not in "iu" or y.dtype.kind not in "iu":
        return map_roi(dict(raw), roi, size, offset)
    width, height = EAP_IMAGE_SIZE
    sensor_valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not sensor_valid.all():
        x, y, t, p = (value[sensor_valid] for value in (x, y, t, p))
    xt = np.arange(width, dtype=np.float64)
    yt = np.arange(height, dtype=np.float64)
    xlut = np.clip(np.floor((xt - x0) * size / (x1 - x0)).astype(np.int32), 0, size - 1)
    ylut = np.clip(np.floor((yt - y0) * size / (y1 - y0)).astype(np.int32), 0, size - 1)
    xlut[(xt < x0) | (xt >= x1)] = -1
    ylut[(yt < y0) | (yt >= y1)] = -1
    mapped_x, mapped_y = xlut[x], ylut[y]
    keep = (mapped_x >= 0) & (mapped_y >= 0)
    if not keep.all():
        mapped_x, mapped_y, t, p = (value[keep] for value in (mapped_x, mapped_y, t, p))
    return {
        "x": mapped_x,
        "y": mapped_y,
        "t": t,
        "p": np.where(p > 0, 1, -1).astype(np.int8),
    }


@lru_cache(maxsize=1)
def _native_temporal_lut() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep one bounded table for the frozen native 100 ms / five-bin ABI."""
    scaled = np.clip(np.arange(100_000, dtype=np.float64) / float(100_000) * 4, 0.0, 4)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.ceil(scaled).astype(np.int32)
    upper_weight = scaled - lower
    values = (lower, upper, 1.0 - upper_weight, upper_weight)
    for value in values:
        value.setflags(write=False)
    return values


def encode_mapped_lut(
    mapped: Mapping[str, np.ndarray], start: int, end: int, size: int, sequence: str
) -> torch.Tensor:
    """Encode one native window without changing histogram accumulation order.

    Non-native durations or timestamps beyond exact float64 integer range use
    the oracle. Normalization still uses the original NumPy float64 quantile.
    """
    if end <= start:
        raise ValueError("positive half-open interval required")
    if end - start != 100_000 or max(abs(start), abs(end)) > 2**53:
        return encode_mapped(dict(mapped), start, end, size, sequence)
    lo, hi = np.searchsorted(mapped["t"], [start, end], side="left")
    elapsed = mapped["t"][lo:hi] - start
    count = len(elapsed)
    output = np.zeros((12, size, size), dtype=np.float32)
    if count:
        lower, upper, lower_weight, upper_weight = _native_temporal_lut()
        spatial = mapped["y"][lo:hi].astype(np.int32) * size + mapped["x"][lo:hi].astype(np.int32)
        polarity = np.where(mapped["p"][lo:hi] > 0, 0, 5).astype(np.int32)
        pixels = size * size
        lower_indices = (lower[elapsed] + polarity) * pixels + spatial
        upper_indices = (upper[elapsed] + polarity) * pixels + spatial
        voxel = np.bincount(lower_indices, weights=lower_weight[elapsed], minlength=10 * pixels)
        voxel += np.bincount(upper_indices, weights=upper_weight[elapsed], minlength=10 * pixels)
        output[:10] = robust_normalize(voxel.reshape(10, size, size))
    output[10] = np.log1p(count)
    output[11] = np.log1p(count / max((end - start) * 1e-6, 1e-6))
    return torch.from_numpy(output)


def crop_uint8_roi_first(
    frame: np.ndarray, square: Sequence[float], *, size: int = RGB_SIZE
) -> np.ndarray:
    """Apply the original PIL bilinear resize after copying only cropped pixels."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("raw teacher frame must be uint8 HWC RGB")
    x0, y0, x1, y1 = _effective_square(frame, square)
    image = Image.fromarray(frame[y0:y1, x0:x1], mode="RGB")
    resized = image.resize((size, size), resample=Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1))
