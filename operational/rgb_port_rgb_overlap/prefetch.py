"""Bounded CPU lookahead on the existing heterogeneous RGB EffectiveSource.

The source object, effective-batch reduction and temporal parts remain intact.
Only its instance batch method is intercepted, after canonical construction.
"""

# ruff: noqa: ANN401 -- adapt the retained dynamic effective-source protocol
from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock
from typing import Any
from unittest.mock import patch


class EffectiveRGBPrefetch:
    """Prepare at most two future effective batches with a single CPU owner.

    No RNG is used and no cursor is advanced. The worker invokes the complete
    original EffectiveSource.batch, including its real T2/T3 partition and the
    inherited bounded input retries. Future exceptions surface only on consume.
    """

    def __init__(self, source: Any, *, batch_size: int, depth: int = 2) -> None:
        counts = set(source.frame_counts)
        if batch_size < 1 or not 1 <= depth <= 2 or not counts or not counts <= {2, 3}:
            raise ValueError("RGB T2/T3 source, positive batch size and depth 1 or 2 required")
        self.source, self.batch_size, self.depth = source, batch_size, depth
        self.original_batch = source.batch
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rgb-effective-prefetch")
        self._queue: deque[tuple[int, tuple[int, ...], Future[Any]]] = deque()
        self._cursor: Mapping[str, Any] | None = None
        self._order: Any = None
        self._position = 0
        self._closed = False
        self._lock = Lock()
        self._submitted = self._consumed = self._prepared = self._outstanding_max = 0
        self._wait_seconds = self._prepare_seconds = 0.0

    @property
    def bound(self) -> bool:
        """Whether a canonical restored or initial cursor has been attached."""
        return self._cursor is not None

    def bind(self, cursor: Mapping[str, Any]) -> None:
        """Attach only after complete native restoration or the initial READY save."""
        if self._closed:
            raise RuntimeError("RGB prefetch is closed")
        if self._cursor is not None:
            if self._cursor is not cursor:
                raise RuntimeError("Cannot replace canonical cursor")
            return
        self._cursor = cursor
        self._reset_order()
        self._fill()

    def _reset_order(self) -> None:
        assert self._cursor is not None
        order = self._cursor["order"]
        position = int(self._cursor["position"])
        if (
            order.ndim != 1
            or len(order) != self.source.population_size
            or not 0 <= position <= self.source.population_size
        ):
            raise ValueError("Invalid canonical RGB order or position")
        self._order, self._position = order, position

    def _load(self, ids: tuple[int, ...]) -> Any:
        tick = time.perf_counter()
        try:
            return self.original_batch(list(ids), "rgb")
        finally:
            with self._lock:
                self._prepare_seconds += time.perf_counter() - tick
                self._prepared += 1

    def _fill(self) -> None:
        while len(self._queue) < self.depth and self._position < self.source.population_size:
            start = self._position
            stop = min(start + self.batch_size, self.source.population_size)
            ids = tuple(map(int, self._order[start:stop].tolist()))
            self._queue.append((start, ids, self._pool.submit(self._load, ids)))
            self._position = stop
            with self._lock:
                self._submitted += 1
                self._outstanding_max = max(self._outstanding_max, len(self._queue))

    def batch(self, indices: Sequence[int], modality: str) -> Any:
        """Consume exactly the next effective batch without taking ownership of its storage."""
        if self._closed or modality != "rgb":
            raise ValueError("Open RGB prefetch required")
        if self._cursor is None:
            return self.original_batch(indices, modality)
        if not self._queue:
            if self._cursor["order"] is self._order:
                raise RuntimeError("Exhausted epoch requires a new canonical order")
            self._reset_order()
            self._fill()
        if not self._queue:
            raise RuntimeError("Canonical order has no remaining effective batch")
        start, expected, future = self._queue[0]
        if (
            self._cursor["order"] is not self._order
            or int(self._cursor["position"]) != start
            or tuple(indices) != expected
        ):
            raise RuntimeError("RGB prefetch request differs from canonical order or cursor")
        tick = time.perf_counter()
        try:
            result = future.result()
        finally:
            with self._lock:
                self._wait_seconds += time.perf_counter() - tick
        self._queue.popleft()
        with self._lock:
            self._consumed += 1
        self._fill()
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return coherent preparation counters, including unused lookahead work."""
        with self._lock:
            return {
                "bound": self.bound,
                "prefetch_depth": self.depth,
                "worker_threads": 1,
                "submitted_batches": self._submitted,
                "consumed_batches": self._consumed,
                "prepared_batches": self._prepared,
                "queued_batches_max": self._outstanding_max,
                "source_wait_s": self._wait_seconds,
                "source_prepare_s": self._prepare_seconds,
            }

    def close(self) -> None:
        """Join the worker before the caller restores source/cache methods."""
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._queue.clear()


class EffectiveSourceCapture:
    """Install lookahead on exactly one freshly constructed effective source."""

    def __init__(self, expected_source: Any, *, batch_size: int, depth: int) -> None:
        self.expected_source = expected_source
        self.batch_size, self.depth = batch_size, depth
        self.loader: EffectiveRGBPrefetch | None = None
        self._had_override = False
        self._override: Any = None

    @property
    def captured(self) -> bool:
        """Whether the expected raw source has been wrapped canonically."""
        return self.loader is not None

    @property
    def bound(self) -> bool:
        """Whether the captured effective source has a canonical cursor."""
        return self.loader is not None and self.loader.bound

    def attach(self, effective: Any) -> None:
        """Preserve concrete type, identity and any prior instance override."""
        if self.loader is not None:
            raise RuntimeError("Expected raw RGB source was wrapped more than once")
        loader = EffectiveRGBPrefetch(effective, batch_size=self.batch_size, depth=self.depth)
        self._had_override = "batch" in vars(effective)
        self._override = vars(effective).get("batch")
        effective.batch = loader.batch
        self.loader = loader

    def bind(self, cursor: Mapping[str, Any]) -> None:
        """Bind the same cursor idempotently after native restore or READY save."""
        if self.loader is None:
            raise RuntimeError("Canonical EffectiveSource has not been captured")
        self.loader.bind(cursor)

    def snapshot(self) -> dict[str, Any]:
        """Expose whether installation and binding have happened."""
        return {"captured": self.captured, **(self.loader.snapshot() if self.loader else {})}

    def close(self) -> None:
        """Finish reads before restoring the captured instance's original method."""
        if self.loader is not None:
            self.loader.close()
            if self._had_override:
                self.loader.source.batch = self._override
            else:
                vars(self.loader.source).pop("batch", None)


@contextmanager
def capture_effective_prefetch(
    effective_class: type[Any],
    *,
    expected_source: Any,
    batch_size: int,
    depth: int = 2,
) -> Iterator[EffectiveSourceCapture]:
    """Capture the canonical constructor without replacing its class or recipe.

    Enter inside the admitted RGB input runtime so its retry/collate/cache
    adapters outlive this context. Bind after original checkpoint restore (or
    initial READY save); all calls before binding remain synchronous.
    """
    original_init = effective_class.__init__
    capture = EffectiveSourceCapture(expected_source, batch_size=batch_size, depth=depth)

    def initialize(instance: Any, source: Any, *args: Any, **kwargs: Any) -> None:
        original_init(instance, source, *args, **kwargs)
        if source is expected_source:
            capture.attach(instance)

    with patch.object(effective_class, "__init__", initialize):
        try:
            yield capture
        finally:
            capture.close()
