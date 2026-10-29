"""Bounded, ordered CPU prefetch without sampler or CUDA ownership.

The caller supplies an immutable, explicit index plan.  Prefetch may materialize
future items, but only items returned by :func:`next` advance the committed
cursor.  A checkpoint can therefore persist ``committed_cursor`` and resume
against the same plan without treating speculative work as consumed.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

IndexT = TypeVar("IndexT")
ItemT = TypeVar("ItemT")
LoaderIndexT = TypeVar("LoaderIndexT", contravariant=True)
LoaderItemT = TypeVar("LoaderItemT", covariant=True)


class CloseableIndexedLoader(Protocol[LoaderIndexT, LoaderItemT]):
    """Per-worker loader contract used by :class:`BoundedOrderedPrefetch`."""

    def __getitem__(self, index: LoaderIndexT, /) -> LoaderItemT: ...

    def close(self) -> None: ...


class PrefetchWorkerError(RuntimeError):
    """Raised in the consumer when the worker fails to load an item."""


@dataclass(frozen=True)
class _Result(Generic[IndexT, ItemT]):
    cursor: int
    index: IndexT
    item: ItemT | None = None
    error: BaseException | None = None
    cleanup_error: BaseException | None = None


@dataclass(frozen=True)
class _Terminal:
    error: BaseException | None = None
    error_context: str | None = None
    cleanup_error: BaseException | None = None


class BoundedOrderedPrefetch(Generic[IndexT, ItemT]):
    """Prefetch an explicit index plan with one CPU worker and strict ordering.

    ``capacity`` bounds all speculative loads, including a load currently in
    progress.  The worker owns the loader returned by ``loader_factory`` and
    closes it before exiting.  The class never imports CUDA, pins memory, moves
    tensors, mutates returned items, or advances a sampler/RNG.

    Args:
        indices: Fixed index sequence. It is snapshotted as a tuple; its items
            should themselves be immutable values (for example batch-index
            tuples). Generate and checkpoint this plan outside the prefetcher
            when a randomized sampler is required.
        loader_factory: Creates the loader used exclusively by the worker.
        start_cursor: First unconsumed position in ``indices``.
        capacity: Maximum number of loaded or in-progress speculative items.
        thread_name: Diagnostic name for the worker thread.
    """

    def __init__(
        self,
        indices: Sequence[IndexT],
        loader_factory: Callable[[], CloseableIndexedLoader[IndexT, ItemT]],
        *,
        start_cursor: int = 0,
        capacity: int = 2,
        thread_name: str = "bounded-prefetch",
    ) -> None:
        if type(capacity) is not int:
            raise TypeError("capacity must be an int (bool is not accepted)")
        if capacity not in (1, 2):
            raise ValueError("capacity must be 1 or 2")
        if type(start_cursor) is not int:
            raise TypeError("start_cursor must be an int (bool is not accepted)")
        if not 0 <= start_cursor <= len(indices):
            raise ValueError("start_cursor must be within the explicit index plan")

        self._indices = tuple(indices)
        self._loader_factory = loader_factory
        self._capacity = capacity
        self._committed_cursor = start_cursor
        self._results: queue.Queue[_Result[IndexT, ItemT] | _Terminal] = queue.Queue(
            maxsize=capacity
        )
        self._permits = threading.BoundedSemaphore(capacity)
        self._cancelled = threading.Event()
        self._close_requested = False
        self._closed = False
        self._cleanup_error: BaseException | None = None
        self._state_lock = threading.Lock()
        self._outstanding = 0
        self._maximum_outstanding = 0
        self._thread = threading.Thread(
            target=self._worker,
            args=(start_cursor,),
            name=thread_name,
            daemon=True,
        )
        self._thread.start()

    @property
    def committed_cursor(self) -> int:
        """Position after the last item actually returned to the consumer."""

        return self._committed_cursor

    @property
    def maximum_outstanding(self) -> int:
        """Maximum observed speculative loads, including an in-progress load."""

        with self._state_lock:
            return self._maximum_outstanding

    @property
    def worker_alive(self) -> bool:
        """Whether the private loader worker is still running."""

        return self._thread.is_alive()

    def state_dict(self) -> dict[str, int]:
        """Return the only cursor that is safe to store in a checkpoint."""

        return {"committed_cursor": self._committed_cursor}

    def __iter__(self) -> BoundedOrderedPrefetch[IndexT, ItemT]:
        return self

    def __next__(self) -> ItemT:
        if self._closed:
            raise StopIteration
        if self._close_requested:
            raise RuntimeError("prefetch close is pending; retry close()")

        result = self._results.get()
        if isinstance(result, _Terminal):
            if result.error is not None:
                self._close_internal(raise_cleanup=False)
                failure = PrefetchWorkerError(f"prefetch {result.error_context or 'worker'} failed")
                if result.cleanup_error is not None:
                    failure.add_note(
                        "loader cleanup also failed: "
                        f"{type(result.cleanup_error).__name__}: {result.cleanup_error}"
                    )
                raise failure from result.error
            self._close_internal(raise_cleanup=True)
            raise StopIteration

        self._release_outstanding()
        if result.cursor != self._committed_cursor:
            self.close()
            raise RuntimeError(
                "prefetch ordering invariant failed: "
                f"expected cursor {self._committed_cursor}, got {result.cursor}"
            )
        if result.error is not None:
            self._close_internal(raise_cleanup=False)
            failure = PrefetchWorkerError(
                f"failed to load plan cursor {result.cursor} (index={result.index!r})"
            )
            if result.cleanup_error is not None:
                failure.add_note(
                    "loader cleanup also failed: "
                    f"{type(result.cleanup_error).__name__}: {result.cleanup_error}"
                )
            raise failure from result.error
        self._committed_cursor += 1
        return cast(ItemT, result.item)

    def close(self, *, timeout_s: float = 5.0) -> None:
        """Cancel work and synchronously release the worker-owned loader.

        Python cannot interrupt an arbitrary blocking loader call. If shutdown
        exceeds ``timeout_s``, this method raises but remains retryable: a later
        call joins the same worker and completes cleanup.
        """

        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        self._close_internal(raise_cleanup=True, timeout_s=timeout_s)

    def _close_internal(self, *, raise_cleanup: bool, timeout_s: float = 5.0) -> None:
        """Shared shutdown path with optional cleanup-error composition."""

        if self._closed:
            return
        self._close_requested = True
        self._cancelled.set()
        # Do not drain/release permits until the worker has observed
        # cancellation. Releasing them first could admit a new speculative load
        # during shutdown and violate the configured lookahead bound.
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise RuntimeError(
                f"prefetch worker did not stop within {timeout_s:g} seconds; "
                "close() may be retried after blocking I/O returns"
            )
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            if isinstance(result, _Result):
                self._release_outstanding()
        self._closed = True
        if raise_cleanup and self._cleanup_error is not None:
            raise PrefetchWorkerError("prefetch loader cleanup failed") from self._cleanup_error

    def __enter__(self) -> BoundedOrderedPrefetch[IndexT, ItemT]:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _worker(self, start_cursor: int) -> None:
        loader: CloseableIndexedLoader[IndexT, ItemT] | None = None
        load_failure: tuple[int, IndexT, BaseException] | None = None
        factory_error: BaseException | None = None
        unexpected_error: BaseException | None = None
        try:
            if start_cursor == len(self._indices):
                return
            try:
                loader = self._loader_factory()
            except BaseException as error:
                factory_error = error
                return
            for cursor in range(start_cursor, len(self._indices)):
                if not self._acquire_permit():
                    return
                index = self._indices[cursor]
                try:
                    item = loader[index]
                    result = _Result(cursor=cursor, index=index, item=item)
                except BaseException as error:
                    load_failure = (cursor, index, error)
                    break
                if not self._put(result):
                    self._release_outstanding()
                    return
        except BaseException as error:
            unexpected_error = error
        finally:
            if loader is not None:
                try:
                    loader.close()
                except BaseException as error:
                    self._cleanup_error = error

            # Completion is visible only after loader cleanup. For an item
            # failure the existing permit remains attached to the error result.
            if load_failure is not None:
                cursor, index, error = load_failure
                result = _Result[IndexT, ItemT](
                    cursor=cursor,
                    index=index,
                    error=error,
                    cleanup_error=self._cleanup_error,
                )
                if not self._put(result):
                    self._release_outstanding()
            elif factory_error is not None:
                self._put(
                    _Terminal(
                        error=factory_error,
                        error_context="loader factory",
                        cleanup_error=self._cleanup_error,
                    )
                )
            elif unexpected_error is not None:
                self._put(
                    _Terminal(
                        error=unexpected_error,
                        error_context="worker",
                        cleanup_error=self._cleanup_error,
                    )
                )
            else:
                self._put(_Terminal())

    def _acquire_permit(self) -> bool:
        while not self._cancelled.is_set():
            if self._permits.acquire(timeout=0.05):
                with self._state_lock:
                    self._outstanding += 1
                    self._maximum_outstanding = max(self._maximum_outstanding, self._outstanding)
                return True
        return False

    def _release_outstanding(self) -> None:
        with self._state_lock:
            if self._outstanding <= 0:
                return
            self._outstanding -= 1
        self._permits.release()

    def _put(self, result: _Result[IndexT, ItemT] | _Terminal) -> bool:
        while not self._cancelled.is_set():
            try:
                self._results.put(result, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False
