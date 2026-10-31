"""Sensor-only bbox histories and independent joins to existing prediction caches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from operational.phase_robust.geometry import expansion_features


def histories(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Deduplicate explicitly timestamped RGB ROIs; no labels are read here."""
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        sequence = grouped.setdefault(str(row["sequence_id"]), {})
        for clock, box in zip(row["rgb_timestamps_us"], row["rgb_boxes"], strict=True):
            clock = int(clock)
            if clock in sequence and not np.array_equal(sequence[clock], box):
                raise ValueError("Conflicting boxes for one physical frame")
            sequence[clock] = box
    result = {}
    for name, observations in grouped.items():
        clocks = np.array(sorted(observations), dtype=np.int64)
        result[name] = np.array([observations[int(t)] for t in clocks]), clocks
    return result


def query_geometry(history: tuple[np.ndarray, np.ndarray], query_us: int) -> dict[str, Any]:
    """Bound the actual observations before invoking the fit."""
    boxes, clocks = history
    lo, hi = np.searchsorted(clocks, [query_us - 650000, query_us], side="left")
    hi = int(np.searchsorted(clocks, query_us, side="right"))
    return expansion_features(boxes[lo:hi], clocks[lo:hi], query_us)
