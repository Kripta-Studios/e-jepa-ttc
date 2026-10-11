"""Normalize only identity metadata before the original strict observation hash."""

# ruff: noqa: ANN401
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from operational.rgb_port.infer_experts import _observation_signature as original_signature

FIELDS = (
    "producer_times_us",
    "producer_sensor_times_us",
    "producer_available_us",
    "producer_shards",
    "producer_members",
    "producer_boxes_xyxy",
    "roi_xyxy",
)


def _native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        result = value.item()
        if isinstance(result, np.generic):
            raise TypeError(f"Unsupported NumPy identity scalar: {value.dtype}")
        return _native(result)
    if isinstance(value, Mapping):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


def observation_signature(row: Mapping[str, Any]) -> str:
    """Preserve native hashes, integer clocks, finite-value checks and mismatch rejection.

    Conversion allocates metadata containers only. Images, tensors, labels, clocks
    and boxes in the source row are never mutated or passed through floating point.
    """
    return original_signature({key: _native(row[key]) for key in FIELDS if key in row})
