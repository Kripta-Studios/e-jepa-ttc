"""Deterministic depth-one host prefetch for RGB-PORT event producers."""

# ruff: noqa: ANN401

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from torch import Tensor


class DepthOneEventPrefetch:
    """Wrap a producer source without creating orders or advancing its cursor."""

    def __init__(self, source: Any, *, effective_batch_size: int, microbatch_size: int) -> None:
        self._source = source
        self.population_size: int = int(source.population_size)
        self.identity: Mapping[str, Any] = source.identity
        self.frame_counts: Sequence[int] = source.frame_counts
        self.input_span_us: Sequence[int] = source.input_span_us
        self._effective = int(effective_batch_size)
        self._micro = int(microbatch_size)
        if self._effective <= 0 or self._micro <= 0:
            raise ValueError("Prefetch batch sizes must be positive")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rgb-port-prefetch")
        self._cursor: Mapping[str, Any] | None = None
        self._future: Future[Any] | None = None
        self._future_ids: tuple[int, ...] | None = None
        self._plan: list[tuple[int, ...]] = []
        self._plan_index = 0
        self._plan_stop = 0
        self._closed = False
        self._effective_limit: int | None = None
        self._effective_consumed = 0
        self._cache_baseline = {"reads": 0, "hits": 0}
        self._lock = threading.Lock()
        self._timings = {
            "source_prepare_seconds": 0.0,
            "source_wait_seconds": 0.0,
            "source_batches": 0,
            "prefetch_hits": 0,
            "synchronous_boundaries": 0,
            "submitted": 0,
            "consumed": 0,
            "cancelled": 0,
            "errors": 0,
            "order_mismatches": 0,
            "outstanding_max": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    @property
    def original(self) -> Any:
        """Return the unwrapped source used by the V1 graph prewarm."""
        return self._source

    def timing_snapshot(self) -> dict[str, float | int]:
        """Return in-memory counters without touching the filesystem."""
        with self._lock:
            values = dict(self._timings)
            values["outstanding_current"] = max(
                0,
                int(values["submitted"])
                - int(values["consumed"])
                - int(values["cancelled"])
                - int(values["errors"]),
            )
        inputs = getattr(self._source, "_inputs", None)
        cache_lock = getattr(inputs, "lock", None)
        if inputs is not None and cache_lock is not None:
            with cache_lock:
                values.update(
                    cache_reads=int(inputs.reads) - self._cache_baseline["reads"],
                    cache_hits=int(inputs.hits) - self._cache_baseline["hits"],
                    cache_entries=len(inputs.cache),
                    cache_bytes=int(inputs.cache_bytes),
                )
        else:
            values.update(cache_reads=0, cache_hits=0, cache_entries=0, cache_bytes=0)
        return values

    def limit_effective_batches(self, count: int) -> None:
        """Stop submission exactly after a preregistered number of optimizer batches."""
        if count <= 0 or self._effective_limit is not None or self._future is not None:
            raise ValueError("Effective prefetch limit must be set once before bind")
        self._effective_limit = int(count)

    def bind(self, cursor: Mapping[str, Any]) -> None:
        """Bind the canonical restored cursor and prepare its first requested batch."""
        if self._closed:
            raise RuntimeError("Cannot bind a closed prefetch source")
        if self._cursor is not None:
            if self._cursor is not cursor:
                raise RuntimeError("Prefetch source cannot change cursor identity")
            return
        self._cursor = cursor
        if self._future is None:
            inputs = getattr(self._source, "_inputs", None)
            cache_lock = getattr(inputs, "lock", None)
            if inputs is not None and cache_lock is not None:
                with cache_lock:
                    self._cache_baseline = {
                        "reads": int(inputs.reads),
                        "hits": int(inputs.hits),
                    }
            self._plan_from_cursor()
            self._submit_current()

    def _microbatches(self, indices: Sequence[int]) -> list[tuple[int, ...]]:
        counts = getattr(self._source, "frame_counts", None)
        if counts is None:
            return [
                tuple(indices[start : start + self._micro])
                for start in range(0, len(indices), self._micro)
            ]
        groups: dict[int, list[int]] = {}
        for index in indices:
            groups.setdefault(int(counts[index]), []).append(int(index))
        return [
            tuple(group[start : start + self._micro])
            for _, group in sorted(groups.items())
            for start in range(0, len(group), self._micro)
        ]

    def _plan_from_cursor(self) -> None:
        if self._cursor is None:
            raise RuntimeError("Prefetch cursor is not bound")
        order = self._cursor.get("order")
        if not isinstance(order, Tensor) or order.ndim != 1:
            raise ValueError("Prefetch requires the canonical one-dimensional order tensor")
        start = int(self._cursor["position"])
        stop = min(start + self._effective, int(self.population_size))
        self._plan = self._microbatches(order[start:stop].tolist())
        self._plan_index = 0
        self._plan_stop = stop

    def _load(self, ids: tuple[int, ...]) -> Any:
        started = time.perf_counter()
        try:
            return self._source.batch(list(ids), "event")
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self._timings["source_prepare_seconds"] += elapsed
                self._timings["source_batches"] += 1

    def _submit_current(self) -> None:
        if self._closed or self._plan_index >= len(self._plan):
            return
        ids = self._plan[self._plan_index]
        self._future_ids = ids
        self._future = self._executor.submit(self._load, ids)
        with self._lock:
            self._timings["submitted"] += 1
            self._timings["outstanding_max"] = max(int(self._timings["outstanding_max"]), 1)

    def _advance(self) -> None:
        self._plan_index += 1
        if self._plan_index < len(self._plan):
            self._submit_current()
            return
        self._effective_consumed += 1
        if self._effective_limit is not None and self._effective_consumed >= self._effective_limit:
            self._plan = []
            self._plan_index = 0
            self._future = None
            self._future_ids = None
            return
        if self._plan_stop < int(self.population_size):
            if self._cursor is None:
                raise RuntimeError("Prefetch cursor disappeared")
            order = self._cursor["order"]
            start = self._plan_stop
            stop = min(start + self._effective, int(self.population_size))
            self._plan = self._microbatches(order[start:stop].tolist())
            self._plan_index = 0
            self._plan_stop = stop
            self._submit_current()
            return
        self._plan = []
        self._plan_index = 0
        self._future = None
        self._future_ids = None

    def batch(self, indices: Sequence[int], modality: str) -> Any:
        """Return exactly the requested canonical batch and prefetch its successor."""
        if modality != "event":
            raise ValueError("Pipeline V2 prefetch is restricted to event producers")
        if self._closed:
            raise RuntimeError("Prefetch source is closed")
        requested = tuple(map(int, indices))
        if self._future is None:
            self._plan_from_cursor()
            self._submit_current()
            with self._lock:
                self._timings["synchronous_boundaries"] += 1
        if requested != self._future_ids or self._future is None:
            with self._lock:
                self._timings["order_mismatches"] += 1
            raise RuntimeError(
                f"Prefetch request differs from canonical cursor: {requested} != {self._future_ids}"
            )
        future = self._future
        started = time.perf_counter()
        try:
            value = future.result()
        except BaseException:
            with self._lock:
                self._timings["errors"] += 1
            raise
        finally:
            waited = time.perf_counter() - started
            with self._lock:
                self._timings["source_wait_seconds"] += waited
                self._timings["prefetch_hits"] += 1
        with self._lock:
            self._timings["consumed"] += 1
        self._future = None
        self._future_ids = None
        self._advance()
        return value

    def close(self) -> None:
        """Cancel the sole queued future and join the coordinator thread."""
        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            cancelled = self._future.cancel()
            if cancelled:
                with self._lock:
                    self._timings["cancelled"] += 1
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._future = None
        self._future_ids = None

    def __enter__(self) -> DepthOneEventPrefetch:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["DepthOneEventPrefetch"]
