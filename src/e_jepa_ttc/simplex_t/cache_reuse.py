"""Reuse frozen observation blocks only across identical content-key bindings."""

from __future__ import annotations

import numpy as np

from .compiled_context import validate_block


def rebind_block(
    arrays: dict[str, np.ndarray],
    *,
    source_keys: np.ndarray,
    destination_keys: np.ndarray,
    destination_ids: np.ndarray,
    anchors_us: np.ndarray,
    available_us: int,
) -> dict[str, np.ndarray]:
    """Change observation row numbers, never expert outputs or input dependencies.

    Callers first verify block, index, deduplication and producer manifest hashes.
    Content keys bind raw data, preprocessing, producer, crop and availability.
    Equal query names alone are insufficient. This operation grants no new role,
    temporal or replay authority; it does not load targets or execute experts.
    """
    original_ids = arrays["observation_ids"]
    for ids, keys in ((original_ids, source_keys), (destination_ids, destination_keys)):
        if (
            ids.ndim != 1
            or ids.dtype != np.int64
            or not len(ids)
            or len(np.unique(ids)) != len(ids)
            or keys.ndim != 1
            or (ids < 0).any()
            or (ids >= len(keys)).any()
        ):
            raise ValueError("invalid reuse observation index")
    if original_ids.shape != destination_ids.shape:
        raise ValueError("reused history support changed")
    original = source_keys[original_ids]
    target = destination_keys[destination_ids]
    if not np.array_equal(original, target):
        raise ValueError("reused content key changes producer or input dependencies")
    if any(
        not isinstance(key, str) or len(key) != 64 or set(key) - set("0123456789abcdef")
        for key in original.tolist()
    ):
        raise ValueError("reuse requires canonical SHA256 observation keys")
    validate_block(arrays, original_ids, anchors_us, available_us)
    result = dict(arrays)
    result["observation_ids"] = destination_ids.copy()
    validate_block(result, destination_ids, anchors_us, available_us)
    return result
