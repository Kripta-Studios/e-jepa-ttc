"""Retain the small DINO working set and index its immutable shard metadata.

The RGB pixel cache and the sampler are unchanged. Successful decoding and row
identity checks follow RGBProducerSource._dino_for_indices exactly. No Torch,
worker processes, disk copies or optimizer operations are needed by this layer.
"""

# ruff: noqa: ANN401 -- compatibility adapter for the frozen RGB source
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

MAX_SHARDS = 32
MAX_BYTES = 64 * 1024**2


class DinoCache:
    """Single-consumer, bounded LRU over unchanged generic-DINO NPZ payloads."""

    def __init__(self, source: Any, *, max_shards: int, max_bytes: int) -> None:
        if max_shards < 1 or max_bytes < 1:
            raise ValueError("DINO cache bounds must be positive")
        self.source = source
        self.max_shards, self.max_bytes = max_shards, max_bytes
        self.original = source._dino_for_indices
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.sizes: dict[str, int] = {}
        self.reads = self.hits = self.retained_bytes = self.peak_retained_bytes = 0
        self.largest_shard_bytes = 0
        self.lookup: dict[int, dict[str, Any] | None] = {}
        if source._dino_values is not None and source._dino_valid is not None:
            return
        shards = list(source._dino_shards)
        ordered = sorted(shards, key=lambda item: int(item["start"]))
        starts = [int(item["start"]) for item in ordered]
        disjoint = all(int(item["start"]) < int(item["stop"]) for item in ordered) and all(
            int(left["stop"]) <= int(right["start"])
            for left, right in zip(ordered, ordered[1:], strict=False)
        )
        for row in source.dataset.rows:
            ordinal = int(row["source_ordinal"])
            if ordinal in self.lookup:
                continue
            if disjoint:
                position = bisect_right(starts, ordinal) - 1
                candidate = ordered[position] if position >= 0 else None
                self.lookup[ordinal] = (
                    candidate
                    if candidate is not None and ordinal < int(candidate["stop"])
                    else None
                )
            else:
                # Preserve the original first-match behavior even for overlapping
                # metadata; this overlay must not silently choose another shard.
                self.lookup[ordinal] = next(
                    (item for item in shards if int(item["start"]) <= ordinal < int(item["stop"])),
                    None,
                )

    def _read(self, key: str) -> dict[str, np.ndarray]:
        stored = self.cache.get(key)
        if stored is not None:
            self.hits += 1
            self.cache.move_to_end(key)
            return stored
        with np.load(Path(key), allow_pickle=False) as payload:
            stored = {name: payload[name] for name in payload.files}
        self.reads += 1
        size = sum(array.nbytes for array in stored.values())
        self.largest_shard_bytes = max(self.largest_shard_bytes, size)
        if size <= self.max_bytes:
            while self.cache and (
                len(self.cache) >= self.max_shards or self.retained_bytes + size > self.max_bytes
            ):
                stale, _ = self.cache.popitem(last=False)
                self.retained_bytes -= self.sizes.pop(stale)
            self.cache[key] = stored
            self.sizes[key] = size
            self.retained_bytes += size
            self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
        # An oversized shard can still be served exactly, but is never retained.
        # Temporary decoded storage is separate from the retained-cache bound.
        return stored

    def get(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return the original arrays in request order, checking every row identity."""
        if self.source._dino_values is not None and self.source._dino_valid is not None:
            return self.original(indices)
        values: list[np.ndarray] = []
        valid: list[np.ndarray] = []
        for index in indices:
            row = self.source.dataset.rows[int(index)]
            ordinal = int(row["source_ordinal"])
            item = self.lookup[ordinal]
            if item is None:
                raise ValueError(f"no generic-DINO shard covers source ordinal {ordinal}")
            stored = self._read(str(item["path"]))
            local = ordinal - int(item["start"])
            if int(stored["ordinals"][local]) != ordinal or str(stored["tokens"][local]) != str(
                row["sample_token"]
            ):
                raise ValueError("generic-DINO shard row identity mismatch")
            values.append(stored["relation_targets"][local])
            valid.append(stored["relation_valid"][local])
        return np.stack(values), np.stack(valid)

    def snapshot(self) -> dict[str, int]:
        """Return auditable counters; retained bytes exclude a transient shard read."""
        return {
            "reads": self.reads,
            "hits": self.hits,
            "retained_shards": len(self.cache),
            "retained_bytes": self.retained_bytes,
            "peak_retained_bytes": self.peak_retained_bytes,
            "largest_shard_bytes": self.largest_shard_bytes,
            "max_shards": self.max_shards,
            "max_bytes": self.max_bytes,
            "indexed_ordinals": len(self.lookup),
        }

    def close(self) -> None:
        """Release only this overlay's private decoded arrays."""
        self.cache.clear()
        self.sizes.clear()
        self.retained_bytes = 0


@contextmanager
def installed_dino_cache(
    source: Any, *, max_shards: int = MAX_SHARDS, max_bytes: int = MAX_BYTES
) -> Iterator[DinoCache]:
    """Temporarily replace one source's reader while preserving source identity."""
    reader = DinoCache(source, max_shards=max_shards, max_bytes=max_bytes)
    had_attribute = "_dino_for_indices" in vars(source)
    previous_attribute = vars(source).get("_dino_for_indices")
    source._dino_for_indices = reader.get
    try:
        yield reader
    finally:
        if had_attribute:
            source._dino_for_indices = previous_attribute
        else:
            del source._dino_for_indices
        reader.close()
