"""Label-free selection of real RGB frames and distinct observations."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from .contracts import LOOKBACK_US, MAX_OBSERVATIONS, RGBFrame


def sensor_timestamp_us(member_path: str) -> int:
    """Parse the public sensor timestamp suffix without mixing it with query time."""
    match = re.search(r"(\d+)_(\d+)\.[^.]+$", member_path)
    if match is None:
        raise ValueError(f"RGB member lacks a timestamp suffix: {member_path}")
    return int(match.group(1)) * 1_000_000 + int(match.group(2))


def select_frames(
    frames: Sequence[RGBFrame], *, cutoff_us: int, lookback_us: int = LOOKBACK_US
) -> list[RGBFrame]:
    """Select available frames causally, without interpolation or duplication."""
    if lookback_us <= 0:
        raise ValueError("lookback must be positive")
    if len({frame.frame_id for frame in frames}) != len(frames):
        raise ValueError("timeline contains duplicate frame identities")
    selected = [
        frame
        for frame in frames
        if cutoff_us - lookback_us <= frame.timestamp_us <= cutoff_us
        and frame.available_us <= cutoff_us
    ]
    selected.sort(key=lambda frame: (frame.timestamp_us, frame.frame_id))
    if any(a.timestamp_us >= b.timestamp_us for a, b in zip(selected, selected[1:], strict=False)):
        raise ValueError("RGB timeline requires strictly increasing frame timestamps")
    return selected


def observation_triplets(
    frames: Sequence[RGBFrame],
    *,
    cutoff_us: int,
    lookback_us: int = LOOKBACK_US,
    maximum: int = MAX_OBSERVATIONS,
) -> list[tuple[RGBFrame, RGBFrame, RGBFrame]]:
    """Return at most eight sliding triplets of distinct real frames."""
    if not 1 <= maximum <= MAX_OBSERVATIONS:
        raise ValueError("invalid RGB observation capacity")
    ordered = select_frames(frames, cutoff_us=cutoff_us, lookback_us=lookback_us)
    observations = [tuple(ordered[index - 2 : index + 1]) for index in range(2, len(ordered))]
    return observations[-maximum:]  # type: ignore[return-value]


def producer_frames(frames: Sequence[RGBFrame], *, cutoff_us: int) -> list[RGBFrame]:
    """Use the newest triplet, or a genuine T2 cold start, never a repeated image."""
    selected = select_frames(frames, cutoff_us=cutoff_us)
    if len(selected) < 2:
        return []
    return selected[-3:] if len(selected) >= 3 else selected[-2:]


def deltas_seconds(timestamps_us: Sequence[int]) -> np.ndarray:
    """Subtract integer clocks before conversion to FP32 seconds."""
    if len(timestamps_us) < 2 or not all(
        isinstance(value, (int, np.integer)) for value in timestamps_us
    ):
        raise TypeError("at least two integer timestamps are required")
    delta = np.asarray(
        [
            int(current) - int(previous)
            for previous, current in zip(timestamps_us, timestamps_us[1:], strict=False)
        ],
        dtype=np.int64,
    )
    if np.any(delta <= 0):
        raise ValueError("RGB endpoint deltas must be positive")
    return (delta.astype(np.float64) / 1_000_000.0).astype(np.float32)


__all__ = [
    "deltas_seconds",
    "observation_triplets",
    "producer_frames",
    "select_frames",
    "sensor_timestamp_us",
]
