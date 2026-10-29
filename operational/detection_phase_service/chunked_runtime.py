"""Additive ingest-once adapter for more than twenty TRAIN40 objects.

The sealed ``MultiObjectTrain40Service`` remains unchanged and processes one
bounded query chunk at a time.  This adapter owns packet ingestion, then uses
the service's existing event history, RGB preparation, frozen producers and
context runtime for every chunk.  Object IDs include tracker generations, so
no chunk can borrow another generation's context identity.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from torch import Tensor

from operational.train40_runtime.multiobject import (
    ObjectRegion,
    RawRGBHistory,
    RGBObservation,
    _validate_regions,
    _validate_rgb,
)


class _ChunkBackend(Protocol):
    """Inspected subset of ``MultiObjectTrain40Service`` used by the adapter."""

    max_objects: int
    history: Any
    rgb_feature_cache: Any
    parallel_branches: bool
    overlap_event_with_rgb_preparation: bool
    _event_overlap_inflight: bool
    _rgb_uint8: Any
    _closed: bool

    def _pack_rgb(self, objects: Sequence[ObjectRegion], history: RawRGBHistory) -> object: ...

    def infer_prepared(
        self,
        objects: Sequence[ObjectRegion],
        query_us: int,
        event: Tensor,
        rgb: Sequence[Sequence[RGBObservation]],
        unavailable_reason: str = "",
    ) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


class ChunkedTrain40Runtime:
    """Query an arbitrary admitted population through bounded sealed chunks.

    Version 1 intentionally supports the conservative replay configuration:
    no feature cache, uint8 lease, parallel branch, or overlap state.  Those
    modes have lifetimes owned by the historical monolithic ``update`` and are
    rejected rather than silently emulated.
    """

    VERSION = "train40_ingest_once_chunked_v1"
    supports_empty_updates = True

    def __init__(
        self,
        backend: _ChunkBackend,
        *,
        max_objects: int,
        chunk_size: int | None = None,
    ) -> None:
        if isinstance(max_objects, bool) or not isinstance(max_objects, int) or max_objects < 1:
            raise ValueError("Logical max_objects must be a positive integer")
        backend_max = backend.max_objects
        selected = backend_max if chunk_size is None else chunk_size
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or not 1 <= selected <= backend_max
        ):
            raise ValueError("Chunk size must fit the sealed backend capacity")
        if max_objects < selected:
            raise ValueError("Logical max_objects cannot be smaller than chunk size")
        if backend.rgb_feature_cache is not None:
            raise ValueError("Chunked v1 does not share the historical RGB feature cache")
        if backend._rgb_uint8 is not None:
            raise ValueError("Chunked v1 requires the original FP32 RGB preparation path")
        if backend.parallel_branches or backend.overlap_event_with_rgb_preparation:
            raise ValueError("Chunked v1 requires sequential sealed producer branches")
        self.backend = backend
        self.history = backend.history
        self.max_objects = max_objects
        self.chunk_size = selected
        self._last_end_us: int | None = None
        self._closed = False
        self._failed = False
        self._history_rebase_count = 0

    def rebase_history(
        self,
        raw_history: Mapping[str, np.ndarray],
        start_us: int,
        end_us: int,
    ) -> None:
        """Explicitly materialize one independent sealed 600 ms query history.

        RGB exposure endpoints are not aligned to the fixed 20 ms event clock.
        An evaluator that has already consumed each source interval exactly once
        may therefore re-materialize the bounded causal history before a new
        independent query.  This is a public reset boundary, never a retry of a
        failed or partially ingested packet.
        """
        if self._closed or self.backend._closed:
            raise RuntimeError("Chunked TRAIN40 runtime is closed")
        if self._failed:
            raise RuntimeError("Cannot rebase a failed chunked runtime")
        if (
            isinstance(start_us, bool)
            or isinstance(end_us, bool)
            or not isinstance(start_us, int)
            or not isinstance(end_us, int)
            or end_us - start_us != 600_000
        ):
            raise ValueError("History rebase requires exactly 600000 microseconds")
        try:
            self.history.initialize(raw_history, start_us, end_us)
        except BaseException:
            # Initialization may already have mutated the sealed ring. Never
            # permit a retry or another rebase after ambiguous ownership.
            self._failed = True
            raise
        self._last_end_us = end_us
        self._history_rebase_count += 1

    def close(self) -> None:
        """Close the sealed backend exactly once."""
        if not self._closed:
            self.backend.close()
            self._closed = True

    def update(
        self,
        raw_packet: Mapping[str, np.ndarray],
        start_us: int,
        end_us: int,
        objects: Sequence[ObjectRegion],
        rgb_history: RawRGBHistory,
    ) -> dict[str, Any]:
        """Ingest one packet, then evaluate all objects in stable chunks."""
        if self._closed or self.backend._closed:
            raise RuntimeError("Chunked TRAIN40 runtime is closed")
        if self._failed:
            raise RuntimeError("Chunked TRAIN40 runtime failed after packet ingestion")
        if isinstance(start_us, bool) or isinstance(end_us, bool):
            raise ValueError("Packet timestamps must be integer microseconds")
        if not isinstance(start_us, int) or not isinstance(end_us, int) or end_us <= start_us:
            raise ValueError("Packet timestamps must be increasing integer microseconds")
        if self._last_end_us is not None and start_us != self._last_end_us:
            raise ValueError("Packets must be contiguous and consumed exactly once")
        if len(objects) > self.max_objects:
            raise ValueError("Object count exceeds the configured logical capacity")
        if objects:
            _validate_regions(objects, self.max_objects)
        _validate_rgb(rgb_history, end_us)
        if self.backend._event_overlap_inflight:
            raise RuntimeError("Sealed backend has pending event overlap state")

        started_ns = time.perf_counter_ns()
        outputs: list[dict[str, Any]] = []
        chunks = 0
        ingested = False
        try:
            self.history.ingest(raw_packet, start_us, end_us)
            ingested = True
            for first in range(0, len(objects), self.chunk_size):
                current = objects[first : first + self.chunk_size]
                # The history tensor is borrowed until the next query, so each
                # chunk completes inference before another query can overwrite it.
                event = self.history.query_many([item.event_square for item in current])
                packed = self.backend._pack_rgb(current, rgb_history)
                if not isinstance(packed, list):
                    raise TypeError("Chunked v1 requires list-based FP32 RGB observations")
                rgb: list[list[RGBObservation]] = packed
                rows = self.backend.infer_prepared(
                    current,
                    end_us,
                    event,
                    rgb,
                    rgb_history.unavailable_reason,
                )
                if len(rows) != len(current):
                    raise ValueError("Sealed chunk output count differs from its object count")
                expected = [item.object_id for item in current]
                actual = [row.get("object_id") for row in rows]
                if actual != expected or len(set(actual)) != len(actual):
                    raise ValueError("Sealed chunk output identities differ from input order")
                outputs.extend(rows)
                chunks += 1
        except BaseException:
            if ingested:
                self._failed = True
            raise
        self._last_end_us = end_us
        completed_ns = time.perf_counter_ns()
        return {
            "schema": "train40_ingest_once_chunked_v1",
            "objects": outputs,
            "object_count": len(outputs),
            "chunk_count": chunks,
            "chunk_size": self.chunk_size,
            "packet_ingested_count": 1,
            "history_rebase_count": self._history_rebase_count,
            "started_ns": started_ns,
            "completed_ns": completed_ns,
            "complete_wall_ms": (completed_ns - started_ns) / 1e6,
            "targets_read": False,
            "optimizer_updates": 0,
        }


__all__ = ["ChunkedTrain40Runtime"]
