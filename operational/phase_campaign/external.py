"""Strict external label joining, independent of predictions and model errors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def join_external_targets(boxes: Sequence[Mapping[str, Any]], labels: np.ndarray) -> np.ndarray:
    """Join [frame, seconds, depth, speed, TTC] by identity with a 2 us clock check.

    Absent frames remain NaN. Duplicate identities, nonintegral indices and a
    different clock are errors rather than opportunities to search for alignment.
    """
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] != 5:
        raise ValueError("Expected five external label columns: frame,time,depth,speed,TTC")
    if not np.isfinite(labels[:, :2]).all() or np.any(labels[:, 0] != labels[:, 0].astype(int)):
        raise ValueError("Invalid external frame identity or clock")
    lookup = {int(row[0]): row for row in labels}
    if len(lookup) != len(labels):
        raise ValueError("Duplicate external frame identity")
    result = np.full(len(boxes), np.nan)
    for index, box in enumerate(boxes):
        row = lookup.get(int(box["frame_index"]))
        if row is None:
            continue
        if abs(float(row[1]) * 1e6 - int(box["anchor_us"])) > 2:
            raise ValueError("External target clock does not match sensor frame")
        result[index] = row[4]
    return result
