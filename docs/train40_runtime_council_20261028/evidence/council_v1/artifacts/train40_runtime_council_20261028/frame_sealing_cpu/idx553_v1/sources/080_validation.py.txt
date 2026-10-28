"""Validation utilities for event streams and TTC targets."""

from __future__ import annotations

import numpy as np

from e_jepa_ttc.data.types import EventBatch


def normalize_polarity(polarity: np.ndarray) -> np.ndarray:
    """Normalize polarity arrays to int8 values in {-1, +1}."""

    values = np.asarray(polarity)
    if values.dtype == np.bool_:
        return np.where(values, 1, -1).astype(np.int8)

    unique = set(np.unique(values).tolist())
    if unique.issubset({0, 1}):
        return np.where(values > 0, 1, -1).astype(np.int8)
    if unique.issubset({-1, 1}):
        return values.astype(np.int8)

    msg = f"Unsupported polarity values {sorted(unique)!r}; expected bool, {{0,1}}, or {{-1,+1}}."
    raise ValueError(msg)


def validate_event_batch(events: EventBatch, *, allow_empty: bool = False) -> None:
    """Validate event batch invariants."""

    arrays = {
        "x": np.asarray(events.x),
        "y": np.asarray(events.y),
        "t_us": np.asarray(events.t_us),
        "polarity": np.asarray(events.polarity),
    }
    lengths = {name: array.shape[0] for name, array in arrays.items()}
    if any(array.ndim != 1 for array in arrays.values()):
        msg = "Event arrays must be one-dimensional."
        raise ValueError(msg)
    if len(set(lengths.values())) != 1:
        msg = f"Event arrays must have aligned lengths, got {lengths}."
        raise ValueError(msg)
    if not allow_empty and lengths["x"] == 0:
        msg = "Event batch is empty."
        raise ValueError(msg)
    if events.width <= 0 or events.height <= 0:
        msg = f"Invalid resolution {events.width}x{events.height}."
        raise ValueError(msg)

    if lengths["x"] > 0:
        if np.any(arrays["x"] < 0) or np.any(arrays["x"] >= events.width):
            msg = "Event x coordinates are outside the declared width."
            raise ValueError(msg)
        if np.any(arrays["y"] < 0) or np.any(arrays["y"] >= events.height):
            msg = "Event y coordinates are outside the declared height."
            raise ValueError(msg)
        if np.any(np.diff(arrays["t_us"]) < 0):
            msg = "Event timestamps must be monotonic nondecreasing."
            raise ValueError(msg)
        if int(arrays["t_us"][0]) < events.t_start_us or int(arrays["t_us"][-1]) > events.t_end_us:
            msg = "Event timestamps fall outside the declared temporal window."
            raise ValueError(msg)

    normalized = normalize_polarity(arrays["polarity"])
    if not np.array_equal(normalized, arrays["polarity"].astype(np.int8)):
        msg = "Event polarity is not normalized to {-1,+1}."
        raise ValueError(msg)


def validate_ttc_table(timestamps_s: np.ndarray, ttc_s: np.ndarray) -> None:
    """Validate monotonic timestamps and finite positive TTC targets."""

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    targets = np.asarray(ttc_s, dtype=np.float64)
    if timestamps.ndim != 1 or targets.ndim != 1 or timestamps.shape[0] != targets.shape[0]:
        msg = "TTC timestamps and targets must be aligned one-dimensional arrays."
        raise ValueError(msg)
    if timestamps.shape[0] == 0:
        msg = "TTC table is empty."
        raise ValueError(msg)
    if not np.all(np.isfinite(timestamps)) or not np.all(np.isfinite(targets)):
        msg = "TTC table contains non-finite values."
        raise ValueError(msg)
    if np.any(np.diff(timestamps) < 0):
        msg = "TTC timestamps must be monotonic nondecreasing."
        raise ValueError(msg)
    if np.any(targets <= 0):
        msg = "TTC targets must be strictly positive for this MVP parser."
        raise ValueError(msg)
