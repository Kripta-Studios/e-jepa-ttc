"""Synthetic event data with analytically known TTC."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.data.validation import normalize_polarity, validate_event_batch


@dataclass(frozen=True)
class SyntheticSequence:
    """A continuous synthetic stream and aligned TTC targets."""

    events: EventBatch
    frame_id: np.ndarray
    timestamp_s: np.ndarray
    ttc_s: np.ndarray
    distance: np.ndarray
    relative_speed: np.ndarray


def _rectangle_perimeter(
    cx: float, cy: float, half_size: float, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    side = max(2, samples // 4)
    left = np.column_stack(
        [np.full(side, cx - half_size), np.linspace(cy - half_size, cy + half_size, side)]
    )
    right = np.column_stack(
        [np.full(side, cx + half_size), np.linspace(cy - half_size, cy + half_size, side)]
    )
    top = np.column_stack(
        [np.linspace(cx - half_size, cx + half_size, side), np.full(side, cy - half_size)]
    )
    bottom = np.column_stack(
        [np.linspace(cx - half_size, cx + half_size, side), np.full(side, cy + half_size)]
    )
    pts = np.vstack([left, right, top, bottom])
    return pts[:, 0], pts[:, 1]


def generate_synthetic_sequence(
    *,
    width: int = 64,
    height: int = 48,
    windows: int = 128,
    context_ms: int = 100,
    stride_ms: int = 20,
    horizons_ms: tuple[int, ...] = (50, 100),
    seed: int = 0,
    sequence_id: str = "synthetic-expanding-rect",
) -> SyntheticSequence:
    """Generate an expanding rectangle stream with known TTC.

    The object expands linearly until it reaches a configured apparent collision
    size. TTC is the remaining time until that apparent size would be reached.
    """

    if width <= 8 or height <= 8:
        msg = "Synthetic resolution must be larger than 8x8."
        raise ValueError(msg)
    if windows <= 0:
        msg = "Synthetic generation requires at least one window."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    max_horizon_ms = max(horizons_ms) if horizons_ms else 0
    total_ms = context_ms + stride_ms * (windows - 1) + max_horizon_ms + 50
    min_dim = min(width, height)
    start_half_size = min_dim * 0.08
    collision_half_size = min_dim * 0.45
    start_ttc_s = 6.0
    growth_px_per_s = (collision_half_size - start_half_size) / start_ttc_s
    center_x = width * 0.5
    center_y = height * 0.5

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    ps: list[np.ndarray] = []

    for ms in range(total_ms + 1):
        t_s = ms / 1000.0
        half_size = start_half_size + growth_px_per_s * t_s
        edge_x, edge_y = _rectangle_perimeter(center_x, center_y, half_size, samples=32)
        edge_x = edge_x + rng.normal(0.0, 0.25, size=edge_x.shape)
        edge_y = edge_y + rng.normal(0.0, 0.25, size=edge_y.shape)

        background = max(1, int(0.01 * width * height / 100))
        bg_x = rng.integers(0, width, size=background)
        bg_y = rng.integers(0, height, size=background)

        x = np.concatenate([edge_x, bg_x]).round().clip(0, width - 1).astype(np.int32)
        y = np.concatenate([edge_y, bg_y]).round().clip(0, height - 1).astype(np.int32)
        jitter = rng.integers(0, 1000, size=x.shape[0], endpoint=False)
        order = np.argsort(jitter, kind="stable")
        polarity = np.where(np.arange(x.shape[0]) % 2 == 0, 1, -1)

        xs.append(x[order])
        ys.append(y[order])
        ts.append((ms * 1000 + jitter[order]).astype(np.int64))
        ps.append(polarity[order].astype(np.int8))

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    t_all = np.concatenate(ts)
    p_all = normalize_polarity(np.concatenate(ps))
    events = EventBatch(
        x=x_all,
        y=y_all,
        t_us=t_all,
        polarity=p_all,
        width=width,
        height=height,
        sequence_id=sequence_id,
        t_start_us=0,
        t_end_us=int(total_ms * 1000 + 999),
    )
    validate_event_batch(events)

    ref_ms = context_ms + np.arange(windows, dtype=np.int64) * stride_ms
    timestamp_s = ref_ms.astype(np.float64) / 1000.0
    half_size = start_half_size + growth_px_per_s * timestamp_s
    ttc_s = np.maximum((collision_half_size - half_size) / growth_px_per_s, 0.05)
    distance = np.maximum(collision_half_size - half_size, 0.0)
    relative_speed = np.full_like(distance, growth_px_per_s)

    return SyntheticSequence(
        events=events,
        frame_id=np.arange(windows, dtype=np.int64),
        timestamp_s=timestamp_s,
        ttc_s=ttc_s.astype(np.float64),
        distance=distance.astype(np.float64),
        relative_speed=relative_speed.astype(np.float64),
    )


def write_synthetic_hdf5(path: str | Path, sequence: SyntheticSequence) -> None:
    """Write a synthetic sequence to a compact HDF5 fixture."""

    import h5py

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as h5:
        events_group = h5.create_group("events")
        events_group.attrs["width"] = sequence.events.width
        events_group.attrs["height"] = sequence.events.height
        events_group.attrs["sequence_id"] = sequence.events.sequence_id
        events_group.create_dataset("x", data=sequence.events.x, compression="gzip")
        events_group.create_dataset("y", data=sequence.events.y, compression="gzip")
        events_group.create_dataset("t", data=sequence.events.t_us, compression="gzip")
        events_group.create_dataset("p", data=sequence.events.polarity, compression="gzip")

        ttc_group = h5.create_group("ttc")
        ttc_group.create_dataset("frame_id", data=sequence.frame_id)
        ttc_group.create_dataset("timestamp_s", data=sequence.timestamp_s)
        ttc_group.create_dataset("distance", data=sequence.distance)
        ttc_group.create_dataset("relative_speed", data=sequence.relative_speed)
        ttc_group.create_dataset("ttc_s", data=sequence.ttc_s)


def read_synthetic_hdf5(path: str | Path) -> SyntheticSequence:
    """Read a synthetic HDF5 fixture."""

    import h5py

    with h5py.File(path, "r") as h5:
        events_group = h5["events"]
        events = EventBatch(
            x=events_group["x"][:].astype(np.int32),
            y=events_group["y"][:].astype(np.int32),
            t_us=events_group["t"][:].astype(np.int64),
            polarity=normalize_polarity(events_group["p"][:]),
            width=int(events_group.attrs["width"]),
            height=int(events_group.attrs["height"]),
            sequence_id=str(events_group.attrs["sequence_id"]),
            t_start_us=int(events_group["t"][0]),
            t_end_us=int(events_group["t"][-1]),
        )
        validate_event_batch(events)
        ttc_group = h5["ttc"]
        return SyntheticSequence(
            events=events,
            frame_id=ttc_group["frame_id"][:].astype(np.int64),
            timestamp_s=ttc_group["timestamp_s"][:].astype(np.float64),
            distance=ttc_group["distance"][:].astype(np.float64),
            relative_speed=ttc_group["relative_speed"][:].astype(np.float64),
            ttc_s=ttc_group["ttc_s"][:].astype(np.float64),
        )
