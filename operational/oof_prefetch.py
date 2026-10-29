"""Bounded, deterministic CPU prefetch for frozen OOF fold producers."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable


class PopulationSource(Protocol):
    """Minimal source contract required by the prefetcher."""

    @property
    def population_size(self) -> int: ...


class EffectiveBatchRecipe(Protocol):
    """Recipe field needed to preserve the frozen effective-batch boundary."""

    @property
    def effective_batch_size(self) -> int: ...


@runtime_checkable
class _HasToList(Protocol):
    def tolist(self) -> object: ...


BatchT = TypeVar("BatchT")
SourceT = TypeVar("SourceT", bound=PopulationSource)
RecipeT = TypeVar("RecipeT", bound=EffectiveBatchRecipe)
BuildBatch = Callable[[SourceT, tuple[int, ...], RecipeT], BatchT]


def _indices(value: object) -> tuple[int, ...]:
    """Convert a saved tensor/array/sequence order without consuming RNG state."""

    if isinstance(value, _HasToList):
        converted = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        converted = list(value)
    else:
        raise ValueError("prefetch order must be an index sequence")
    if not isinstance(converted, list) or any(isinstance(item, list) for item in converted):
        raise ValueError("prefetch order must be one-dimensional")
    try:
        return tuple(int(item) for item in converted)
    except (TypeError, ValueError) as error:
        raise ValueError("prefetch order must contain integer indices") from error


def _effective_batch_size(recipe: EffectiveBatchRecipe) -> int:
    value = recipe.effective_batch_size
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("recipe must define a positive integer effective_batch_size")
    return value


class BoundedFoldPrefetch(Generic[SourceT, RecipeT, BatchT]):
    """Prepare at most two canonical effective batches on one CPU worker.

    The training cursor remains the sole owner of epoch, order and position. This
    helper never advances it and never samples indices. Replacing ``cursor['order']``
    starts a new epoch queue; mutating an order in place fails closed.
    """

    def __init__(
        self,
        source: SourceT,
        recipe: RecipeT,
        cursor: Mapping[str, object],
        build_batch: BuildBatch[SourceT, RecipeT, BatchT],
        *,
        depth: int = 2,
    ) -> None:
        if depth not in {1, 2}:
            raise ValueError("OOF prefetch depth must be one or two")
        if not callable(build_batch):
            raise TypeError("build_batch must be callable")
        population = int(source.population_size)
        if population <= 0:
            raise ValueError("prefetch source population must be positive")
        self.source = source
        self.recipe = recipe
        self.cursor = cursor
        self.build_batch = build_batch
        self.depth = depth
        self.population = population
        self.batch_size = _effective_batch_size(recipe)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oof-fold-prefetch")
        self.queue: deque[tuple[int, tuple[int, ...], Future[BatchT]]] = deque()
        self.order_object: object | None = None
        self.order: tuple[int, ...] = ()
        self.next_position = 0
        self.closed = False
        self.failure: BaseException | None = None
        self._lock = threading.Lock()
        self.submitted = 0
        self.consumed = 0
        self.prepared = 0
        self.cancelled = 0
        self.drained = 0
        self.errors = 0
        self.epoch_resets = 0
        self.wait_seconds = 0.0
        self.prepare_seconds = 0.0
        self.max_retained = 0
        try:
            self._reset(initial=True)
        except BaseException:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.closed = True
            raise

    def _cursor_position(self) -> int:
        raw = self.cursor.get("position")
        if not isinstance(raw, int) or isinstance(raw, bool) or not 0 <= raw <= self.population:
            raise ValueError("saved prefetch cursor position is invalid")
        return raw

    def _validated_order(self) -> tuple[object, tuple[int, ...]]:
        if "order" not in self.cursor:
            raise ValueError("saved prefetch cursor lacks canonical order")
        order_object = self.cursor["order"]
        order = _indices(order_object)
        if len(order) != self.population or set(order) != set(range(self.population)):
            raise ValueError("saved prefetch order is not a population permutation")
        return order_object, order

    def _prepare(self, ids: tuple[int, ...]) -> BatchT:
        started = time.perf_counter()
        try:
            return self.build_batch(self.source, ids, self.recipe)
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self.prepare_seconds += elapsed
                self.prepared += 1

    def _fill(self) -> None:
        while len(self.queue) < self.depth and self.next_position < self.population:
            start = self.next_position
            stop = min(start + self.batch_size, self.population)
            ids = self.order[start:stop]
            if not ids:
                raise RuntimeError("prefetch attempted to submit an empty effective batch")
            self.queue.append((start, ids, self.pool.submit(self._prepare, ids)))
            self.next_position = stop
            self.submitted += 1
            self.max_retained = max(self.max_retained, len(self.queue))

    def _discard_queue(self) -> None:
        while self.queue:
            _, _, future = self.queue.popleft()
            if future.cancel():
                self.cancelled += 1
                continue
            try:
                future.result()
            except BaseException:
                # A stale epoch's result is deliberately discarded. The active
                # epoch will still propagate its own builder failures from get().
                pass
            self.drained += 1

    def _reset(self, *, initial: bool) -> None:
        order_object, order = self._validated_order()
        position = self._cursor_position()
        if not initial:
            self._discard_queue()
            self.epoch_resets += 1
        self.order_object = order_object
        self.order = order
        self.next_position = position
        self._fill()

    def _synchronize_epoch(self) -> None:
        current = self.cursor.get("order")
        if current is not self.order_object:
            self._reset(initial=False)
            return
        if _indices(current) != self.order:
            raise ValueError("saved prefetch order was mutated in place")

    def _abort(self, error: BaseException) -> None:
        self.failure = error
        self._discard_queue()

    def get(self, indices: Sequence[int]) -> BatchT:
        """Return the exact next saved-order batch or fail without advancing state."""

        if self.closed:
            raise RuntimeError("OOF prefetch is closed")
        if self.failure is not None:
            raise RuntimeError("OOF prefetch previously failed") from self.failure
        try:
            self._synchronize_epoch()
            if not self.queue:
                raise StopIteration("saved OOF epoch has no remaining effective batch")
            start, expected, future = self.queue[0]
            requested = _indices(indices)
            position = self._cursor_position()
            if position != start or requested != expected:
                raise ValueError(
                    "requested OOF batch differs from saved cursor position/order: "
                    f"position={position}, expected_start={start}"
                )
            self.queue.popleft()
            started = time.perf_counter()
            try:
                result = future.result()
            finally:
                self.wait_seconds += time.perf_counter() - started
            self.consumed += 1
            self._fill()
            return result
        except BaseException as error:
            self.errors += 1
            self._abort(error)
            raise

    def snapshot(self) -> dict[str, Any]:
        """Return runtime counters without touching futures or cursor state."""

        return {
            "depth": self.depth,
            "worker_threads": 1,
            "population": self.population,
            "effective_batch_size": self.batch_size,
            "submitted": self.submitted,
            "consumed": self.consumed,
            "prepared": self.prepared,
            "outstanding": len(self.queue),
            "max_retained_effective_batches": self.max_retained,
            "cancelled": self.cancelled,
            "drained": self.drained,
            "epoch_resets": self.epoch_resets,
            "errors": self.errors,
            "source_wait_seconds": self.wait_seconds,
            "source_prepare_seconds": self.prepare_seconds,
            "closed": self.closed,
            "failed": self.failure is not None,
        }

    def close(self) -> None:
        """Cancel queued work, drain the running call, and stop the sole worker."""

        if self.closed:
            return
        self._discard_queue()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.closed = True

    def __enter__(self) -> BoundedFoldPrefetch[SourceT, RecipeT, BatchT]:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "BoundedFoldPrefetch",
    "BuildBatch",
    "EffectiveBatchRecipe",
    "PopulationSource",
]
