"""Parallel lossless shard decoding and deterministic two-batch lookahead.

Decoder threads share NumPy storage: no pickle or cross-process tensor copies.
Only the coordinator mutates the canonical row cache. All existing decoding,
teacher/token checks and full collate validation run unchanged.
"""

# ruff: noqa: ANN401
from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from types import SimpleNamespace
from typing import Any, cast

from operational.rgb_port_revision.cache import GroupRowCache


class ParallelGroupRowCache(GroupRowCache):
    """Decode at most ``workers`` shards ahead in canonical shard order."""

    def __init__(self, source: Any, *, workers: int = 2) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        super().__init__(source)
        self.workers = workers
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shard-decode")
        self._pending: dict[int, Future[Any]] = {}
        self._remaining: deque[int] = deque()
        self._counter_lock = Lock()
        self._closed = False
        self.outstanding_max = 0

    def _read(self, number: int) -> Any:
        # Preserve the original reader with private counters during concurrent
        # decompression, then atomically account for every completed read.
        reader = SimpleNamespace(
            inputs=SimpleNamespace(
                directory=self.inputs.directory, output=self.inputs.output, reads=0
            ),
            reads=0,
        )
        result = GroupRowCache._decode(cast(GroupRowCache, reader), number)
        with self._counter_lock:
            self.reads += 1
            self.inputs.reads += 1
        return result

    def _decode(self, number: int) -> Any:
        while self._remaining and len(self._pending) < self.workers:
            next_number = self._remaining.popleft()
            self._pending[next_number] = self.pool.submit(self._read, next_number)
        self.outstanding_max = max(self.outstanding_max, len(self._pending))
        return self._pending.pop(number).result()

    def batch(self, indices: list[int], modality: str) -> Any:
        if self._closed:
            raise RuntimeError("Parallel cache is closed")
        if modality != "event" or not indices:
            raise ValueError("Nonempty event indices required")
        starts = sorted({index // 256 * 256 for index in indices})
        active = tuple(
            self.source._event_indices[index]
            for start in starts
            for index in range(start, min(start + 256, self.source.population_size))
        )
        retained = self.rows if active == self.group else {}
        missing = {
            self.source._event_indices[index] // 32
            for index in indices
            if self.source._event_indices[index] not in retained
        }
        self._remaining = deque(sorted(missing))
        try:
            return super().batch(indices, modality)
        finally:
            # Join even on a failed shard before another request or close.
            for future in self._pending.values():
                if not future.cancel():
                    try:
                        future.result()
                    except Exception:
                        pass  # the requested canonical shard owns error reporting
            self._pending.clear()
            self._remaining.clear()

    def close(self) -> None:
        """Join decoder workers; the source still owns its original Inputs."""
        self._closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)


class DepthTwoEventPrefetch:
    """Look ahead within the restored order without advancing cursor or RNG.

    Restricted to homogeneous event batches with one microbatch per update. A
    single coordinator owns the row cache; ``depth`` bounds queued host batches.
    The consumer owns its returned tensor storage; buffers are never overwritten.
    """

    def __init__(self, source: Any, *, batch_size: int = 32, depth: int = 2) -> None:
        if depth < 1 or batch_size < 1 or set(source.frame_counts) != {3}:
            raise ValueError("Positive bounds and homogeneous T3 event source required")
        self._source = source
        self.batch_size, self.depth = batch_size, depth
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-prefetch")
        self._queue: deque[tuple[tuple[int, ...], Future[Any]]] = deque()
        self._cursor: Mapping[str, Any] | None = None
        self._order: Any = None
        self._position = 0
        self._closed = False
        self.submitted = self.consumed = self.outstanding_max = 0
        self.wait_seconds = self.prepare_seconds = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    @property
    def original(self) -> Any:
        """Expose synchronous source to unchanged CUDA graph warmup."""
        return self._source

    def bind(self, cursor: Mapping[str, Any]) -> None:
        """Bind only after restoration and graph warmup have completed."""
        if self._closed:
            raise RuntimeError("Prefetch is closed")
        if self._cursor is not None and self._cursor is not cursor:
            raise RuntimeError("Cannot replace canonical cursor")
        if self._cursor is None:
            self._cursor = cursor
            self._reset_order()
            self._fill()

    def _reset_order(self) -> None:
        assert self._cursor is not None
        self._order = self._cursor["order"]
        self._position = int(self._cursor["position"])
        if self._order.ndim != 1 or len(self._order) != self.population_size:
            raise ValueError("Invalid canonical order")

    def _load(self, ids: tuple[int, ...]) -> Any:
        start = time.perf_counter()
        try:
            return self._source.batch(list(ids), "event")
        finally:
            self.prepare_seconds += time.perf_counter() - start

    def _fill(self) -> None:
        while len(self._queue) < self.depth and self._position < self.population_size:
            stop = min(self._position + self.batch_size, self.population_size)
            ids = tuple(self._order[self._position : stop].tolist())
            self._queue.append((ids, self._pool.submit(self._load, ids)))
            self._position = stop
            self.submitted += 1
            self.outstanding_max = max(self.outstanding_max, len(self._queue))

    def batch(self, indices: Sequence[int], modality: str) -> Any:
        """Consume exactly the next ordered batch; defer I/O errors until use."""
        if self._closed or modality != "event":
            raise ValueError("Open event prefetch required")
        if self._cursor is None:
            return self._source.batch(indices, modality)
        if not self._queue:
            if self._cursor["order"] is self._order:
                raise RuntimeError("Exhausted epoch requires a new canonical order")
            self._reset_order()
            self._fill()
        expected, future = self._queue[0]
        if tuple(indices) != expected:
            raise RuntimeError("Prefetch request differs from canonical order")
        start = time.perf_counter()
        try:
            result = future.result()
        finally:
            self.wait_seconds += time.perf_counter() - start
        self._queue.popleft()
        self.consumed += 1
        self._fill()
        return result

    def close(self) -> None:
        """Cancel unused work and join before closing the underlying row cache."""
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)
        self._queue.clear()
