"""Voxel grid event representation."""

from __future__ import annotations

import numpy as np

from e_jepa_ttc.data.types import EventBatch


def robust_normalize(voxel: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Robustly scale occupied voxels while preserving occupancy and sign.

    Centering sparse event counts by their nonzero median can erase every
    event when occupied bins have equal magnitude. A non-centred high-quantile
    magnitude scale keeps empty voxels exactly zero and isolated events
    distinguishable from empty space.
    """

    nonzero = voxel[np.abs(voxel) > eps]
    if nonzero.size == 0:
        return voxel.astype(np.float32)
    scale = float(np.quantile(np.abs(nonzero), 0.95))
    normalized = np.zeros_like(voxel, dtype=np.float32)
    occupied = np.abs(voxel) > eps
    normalized[occupied] = voxel[occupied] / max(scale, eps)
    return normalized


def encode_voxel_grid(
    events: EventBatch,
    *,
    bins: int = 5,
    separate_polarity: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Encode events into a linearly interpolated temporal voxel grid."""

    if bins <= 0:
        msg = "bins must be positive."
        raise ValueError(msg)
    channels = bins * 2 if separate_polarity else bins
    voxel = np.zeros((channels, events.height, events.width), dtype=np.float32)
    if events.num_events == 0 or events.duration_us <= 0:
        return voxel

    t_norm = (events.t_us.astype(np.float64) - float(events.t_start_us)) / float(events.duration_us)
    t_scaled = np.clip(t_norm * (bins - 1), 0.0, bins - 1)
    lower = np.floor(t_scaled).astype(np.int64)
    upper = np.ceil(t_scaled).astype(np.int64)
    upper_weight = t_scaled - lower
    lower_weight = 1.0 - upper_weight

    polarity_offset = np.where(events.polarity > 0, 0, bins) if separate_polarity else 0
    signed_value = np.ones(events.num_events, dtype=np.float32)
    if not separate_polarity:
        signed_value = events.polarity.astype(np.float32)

    lower_channels = lower + polarity_offset
    upper_channels = upper + polarity_offset
    np.add.at(voxel, (lower_channels, events.y, events.x), signed_value * lower_weight)
    np.add.at(voxel, (upper_channels, events.y, events.x), signed_value * upper_weight)

    if normalize:
        voxel = robust_normalize(voxel)
    return voxel.astype(np.float32)
