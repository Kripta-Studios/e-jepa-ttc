"""Bounded row cache that follows the unchanged logical 256-row sampler."""

# ruff: noqa: ANN401 -- adapters bridge the retained dynamic source protocol

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4


class GroupRowCache:
    """Retain only unconsumed rows of the current P group, never entire shards.

    One shard is decoded at a time, bounding transient decompression memory.
    Repeated requests (including resume warmup) are safe: consumed rows reload.
    """

    def __init__(self, source: Any) -> None:
        self.source = source
        self.inputs = source._inputs
        self.rows: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.bytes = 0
        self.peak_bytes = 0
        self.group: tuple[int, ...] = ()
        self.consumed: set[int] = set()
        self.reads = self.hits = 0
        self.inputs.cache.clear()
        self.inputs.cache_bytes = 0

    def _decode(self, number: int) -> dict[str, np.ndarray]:
        name = f"shard_{number:05d}.npz"
        with np.load(self.inputs.directory / name, allow_pickle=False) as stored:
            value = {key: stored[key] for key in stored.files}
        with np.load(self.inputs.output / "teacher" / name, allow_pickle=False) as stored:
            for field in ("ordinals", "tokens"):
                if not np.array_equal(value[field], stored[field]):
                    raise ValueError(f"Teacher/event {field} differ")
            value.update({key: stored[key] for key in ("relation_targets", "relation_valid")})
        self.reads += 1
        self.inputs.reads += 1
        return value

    @staticmethod
    def _size(row: dict[str, np.ndarray]) -> int:
        return sum(value.nbytes for value in row.values())

    def _take(self, ordinal: int) -> dict[str, np.ndarray] | None:
        row = self.rows.pop(ordinal, None)
        if row is not None:
            self.bytes -= self._size(row)
            self.hits += 1
            self.inputs.hits += 1
        return row

    def batch(self, indices: list[int], modality: str) -> Any:
        import psutil

        if modality != "event" or not indices:
            raise ValueError("Nonempty event indices required")
        # A terminal batch can straddle two shuffled logical groups. Retain their
        # union under the same byte limit; order remains the caller's exact order.
        starts = sorted({index // 256 * 256 for index in indices})
        active = tuple(
            self.source._event_indices[index]
            for start in starts
            for index in range(start, min(start + 256, self.source.population_size))
        )
        if active != self.group:
            self.rows.clear()
            self.bytes = 0
            self.consumed.clear()
            self.group = active
        ordinals = [self.source._event_indices[index] for index in indices]
        selected = {ordinal: self._take(ordinal) for ordinal in set(ordinals)}
        missing = [ordinal for ordinal, row in selected.items() if row is None]
        wanted = set(active) - self.consumed
        wanted.update(missing)
        for number in sorted({ordinal // 32 for ordinal in missing}):
            shard = self._decode(number)
            for offset, raw_ordinal in enumerate(shard["ordinals"]):
                ordinal = int(raw_ordinal)
                if ordinal not in wanted or ordinal in self.rows:
                    continue
                if str(shard["tokens"][offset]) != self.inputs.metadata[ordinal]["sample_token"]:
                    raise ValueError("Event shard token drifted")
                row = {key: np.array(value[offset], copy=True) for key, value in shard.items()}
                if ordinal in selected:
                    if selected[ordinal] is None:
                        selected[ordinal] = row
                    continue
                size = self._size(row)
                while self.rows and (
                    self.bytes + size > self.inputs.limit
                    or psutil.virtual_memory().available < 4 * 1024**3
                ):
                    _, old = self.rows.popitem(last=False)
                    self.bytes -= self._size(old)
                if (
                    self.bytes + size <= self.inputs.limit
                    and psutil.virtual_memory().available >= 4 * 1024**3
                ):
                    self.rows[ordinal] = row
                    self.bytes += size
                    self.peak_bytes = max(self.peak_bytes, self.bytes)
            del shard
        self.consumed.update(ordinals)
        self.inputs.cache_bytes = self.bytes
        records = []
        fields = {
            "event_v4_common_roi": "events",
            "garl_delta_t_s": "delta",
            "observable_motion": "motion",
            "garl_visible_heights_px": "visible_heights",
            "ttc_s": "target_ttc",
            "event_v4_boxes_xyxy": "boxes",
            "event_v4_common_square_xyxy": "square",
            "dinov3_relation_targets": "relation_targets",
            "dinov3_relation_valid": "relation_valid",
        }
        for ordinal in ordinals:
            row = selected[ordinal]
            if row is None or int(row["ordinals"]) != ordinal:
                raise ValueError("Event shard ordinal is absent or changed")
            records.append(
                {
                    **self.inputs.metadata[ordinal],
                    **{field: row[key] for field, key in fields.items()},
                }
            )
        result = collate_object_event_v4(records)
        if result.sample_tokens != [self.source._tokens[index] for index in indices]:
            raise ValueError("Event sample order changed")
        return result
