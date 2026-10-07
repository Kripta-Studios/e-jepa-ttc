"""Spawn-safe fixed shared-memory arenas for H8 raw preparation.

The public surface intentionally mirrors the small subset of
``ProcessPoolExecutor`` used by ``history_cache``: context management and
``submit(function, job).result()``. Each result is an ephemeral NumPy view.
Calling ``result`` releases its arena lease, so the caller must synchronously
consume/copy the view before the next ``submit``. The canonical H8 caller does
exactly that with ``torch.from_numpy(...).to("cuda")``.
"""

from __future__ import annotations

import atexit
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from threading import Condition, Lock
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

FULL_H8_SHAPE = (16, 3, 12, 128, 128)
FULL_H8_DTYPE = np.dtype(np.float32)
MAX_ARENAS = 8

T = TypeVar("T")
Prepare = Callable[[dict[str, Any]], NDArray[np.generic]]
Initializer = Callable[[], None]

_CHILD_HANDLES: dict[str, SharedMemory] = {}
_CHILD_HANDLES_LOCK = Lock()
_CHILD_CLEANUP_REGISTERED = False


def _close_child_handles() -> None:
    """Close child-only handles; the parent remains the sole unlink owner."""
    with _CHILD_HANDLES_LOCK:
        handles = list(_CHILD_HANDLES.values())
        _CHILD_HANDLES.clear()
    for handle in handles:
        handle.close()


def _worker_init(initializer: Initializer | None) -> None:
    """Run the original raw-worker initializer and install local cleanup."""
    global _CHILD_CLEANUP_REGISTERED
    if initializer is not None:
        initializer()
    if not _CHILD_CLEANUP_REGISTERED:
        atexit.register(_close_child_handles)
        _CHILD_CLEANUP_REGISTERED = True


def _child_handle(name: str) -> SharedMemory:
    with _CHILD_HANDLES_LOCK:
        handle = _CHILD_HANDLES.get(name)
        if handle is None:
            handle = SharedMemory(name=name, create=False)
            _CHILD_HANDLES[name] = handle
        return handle


def _prepare_into_arena(
    function: Prepare,
    job: dict[str, Any],
    shared_name: str,
    slot: int,
    shape: tuple[int, ...],
    dtype_string: str,
    row_token: str,
    profile: bool,
) -> dict[str, Any]:
    """Prepare one row and return metadata only, never the large array."""
    prepare_started = time.perf_counter()
    value = np.asarray(function(job))
    prepared = time.perf_counter()
    dtype = np.dtype(dtype_string)
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(
            f"Shared H8 result contract differs: {value.shape}/{value.dtype} "
            f"!= {shape}/{dtype}"
        )
    handle = _child_handle(shared_name)
    destination = np.ndarray(shape, dtype=dtype, buffer=handle.buf)
    np.copyto(destination, value, casting="no")
    copied = time.perf_counter()
    metadata: dict[str, Any] = {
        "row_token": row_token,
        "slot": slot,
        "shape": shape,
        "dtype": dtype.str,
    }
    if profile:
        metadata["profile"] = {
            "worker_prepare_seconds": prepared - prepare_started,
            "arena_copy_seconds": copied - prepared,
        }
    return metadata


class SharedArrayFuture:
    """One ordered, single-consumption future backed by a leased arena."""

    def __init__(
        self,
        pool: SharedArrayProcessPool,
        future: Future[dict[str, Any]],
        submission_id: int,
        row_token: str,
        slot: int,
    ) -> None:
        self._pool = pool
        self._future = future
        self._submission_id = submission_id
        self._row_token = row_token
        self._slot = slot
        self._consumed = False
        self.profile: dict[str, float] | None = None

    @property
    def row_token(self) -> str:
        return self._row_token

    @property
    def slot(self) -> int:
        return self._slot

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> NDArray[np.generic]:
        """Return one ephemeral view and release its slot for the next submit."""
        if self._consumed:
            raise RuntimeError("A shared-array future may be consumed only once")
        self._pool._require_next(self._submission_id)
        try:
            metadata = self._future.result(timeout=timeout)
        except FutureTimeoutError:
            raise
        except BaseException:
            self._consumed = True
            self._pool._finish_result(self._submission_id, self._slot)
            raise
        try:
            self._pool._validate_metadata(metadata, self._row_token, self._slot)
            profile = metadata.get("profile")
            if profile is not None:
                self.profile = {
                    "worker_prepare_seconds": float(profile["worker_prepare_seconds"]),
                    "arena_copy_seconds": float(profile["arena_copy_seconds"]),
                }
                self._pool._record_profile(self.profile)
            return self._pool._view(self._slot)
        finally:
            self._consumed = True
            self._pool._finish_result(self._submission_id, self._slot)


class SharedArrayProcessPool(Generic[T]):
    """A bounded process pool whose workers publish arrays into fixed arenas."""

    def __init__(
        self,
        max_workers: int = MAX_ARENAS,
        initializer: Initializer | None = None,
        *,
        shape: tuple[int, ...] = FULL_H8_SHAPE,
        dtype: np.dtype[Any] | type[np.generic] = FULL_H8_DTYPE,
        profile: bool = False,
    ) -> None:
        if not 1 <= max_workers <= MAX_ARENAS:
            raise ValueError(f"max_workers must be between 1 and {MAX_ARENAS}")
        self.max_workers = max_workers
        self.shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        if not self.shape or any(value <= 0 for value in self.shape):
            raise ValueError("Shared arena shape must be nonempty and positive")
        self.profile = profile
        self._condition = Condition()
        self._free_slots = deque(range(max_workers))
        self._submission_id = 0
        self._next_result_id = 0
        self._submitted = 0
        self._consumed = 0
        self._active_leases = 0
        self._inflight_max = 0
        self._worker_prepare_seconds = 0.0
        self._arena_copy_seconds = 0.0
        self._closed = False
        self._handles: list[SharedMemory] = []
        size = int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize
        self._arena_bytes = size * max_workers
        try:
            self._handles = [SharedMemory(create=True, size=size) for _ in range(max_workers)]
            self._executor = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=get_context("spawn"),
                initializer=_worker_init,
                initargs=(initializer,),
            )
        except BaseException:
            self._close_and_unlink()
            raise

    def __enter__(self) -> SharedArrayProcessPool[T]:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown(wait=True)

    def submit(self, function: Prepare, job: dict[str, Any]) -> SharedArrayFuture:
        """Lease one arena, applying backpressure once all arenas are pending."""
        with self._condition:
            while not self._free_slots and not self._closed:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("Cannot submit to a closed shared-array pool")
            slot = self._free_slots.popleft()
            submission_id = self._submission_id
            self._submission_id += 1
        row_token = str(job.get("row_token", submission_id))
        try:
            future = self._executor.submit(
                _prepare_into_arena,
                function,
                job,
                self._handles[slot].name,
                slot,
                self.shape,
                self.dtype.str,
                row_token,
                self.profile,
            )
        except BaseException:
            with self._condition:
                self._free_slots.appendleft(slot)
                self._condition.notify()
            raise
        with self._condition:
            self._submitted += 1
            self._active_leases += 1
            self._inflight_max = max(self._inflight_max, self._active_leases)
        return SharedArrayFuture(self, future, submission_id, row_token, slot)

    def _require_next(self, submission_id: int) -> None:
        with self._condition:
            if submission_id != self._next_result_id:
                raise RuntimeError(
                    f"Shared H8 futures must be consumed in submission order: "
                    f"expected {self._next_result_id}, received {submission_id}"
                )

    def _finish_result(self, submission_id: int, slot: int) -> None:
        with self._condition:
            if submission_id == self._next_result_id:
                self._next_result_id += 1
            self._consumed += 1
            self._active_leases -= 1
            self._free_slots.append(slot)
            self._condition.notify()

    def _record_profile(self, profile: dict[str, float]) -> None:
        with self._condition:
            self._worker_prepare_seconds += profile["worker_prepare_seconds"]
            self._arena_copy_seconds += profile["arena_copy_seconds"]

    def snapshot(self) -> dict[str, int | float | bool]:
        """Return bounded metadata-only throughput and ownership accounting."""
        with self._condition:
            return {
                "submitted": self._submitted,
                "consumed": self._consumed,
                "inflight_current": self._active_leases,
                "inflight_max": self._inflight_max,
                "arena_bytes": self._arena_bytes,
                "metadata_payload_only": True,
                "pool_closed": self._closed,
                "worker_prepare_seconds": self._worker_prepare_seconds,
                "arena_copy_seconds": self._arena_copy_seconds,
            }

    def _validate_metadata(
        self, metadata: dict[str, Any], row_token: str, slot: int
    ) -> None:
        expected = {
            "row_token": row_token,
            "slot": slot,
            "shape": self.shape,
            "dtype": self.dtype.str,
        }
        actual = {key: metadata.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"Shared H8 worker metadata differs: {actual} != {expected}")

    def _view(self, slot: int) -> NDArray[np.generic]:
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self._handles[slot].buf)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """Drain workers before closing and unlinking every parent-owned arena."""
        if not wait:
            raise ValueError("Shared H8 shutdown must drain workers before unlinking arenas")
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        try:
            self._executor.shutdown(wait=True, cancel_futures=cancel_futures)
        finally:
            self._close_and_unlink()

    def _close_and_unlink(self) -> None:
        errors: list[BaseException] = []
        handles, self._handles = self._handles, []
        for handle in handles:
            try:
                handle.close()
            except BaseException as error:
                errors.append(error)
            try:
                handle.unlink()
            except FileNotFoundError:
                pass
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError("Failed to release one or more shared H8 arenas") from errors[0]


def h8_shared_pool(
    max_workers: int = MAX_ARENAS,
    initializer: Initializer | None = None,
    *,
    shape: tuple[int, ...] = FULL_H8_SHAPE,
    dtype: np.dtype[Any] | type[np.generic] = FULL_H8_DTYPE,
    profile: bool = False,
) -> SharedArrayProcessPool[Any]:
    """Construct the scoped ProcessPoolExecutor replacement for H8 history only."""
    return SharedArrayProcessPool(
        max_workers, initializer, shape=shape, dtype=dtype, profile=profile
    )


__all__ = [
    "FULL_H8_DTYPE",
    "FULL_H8_SHAPE",
    "MAX_ARENAS",
    "SharedArrayFuture",
    "SharedArrayProcessPool",
    "h8_shared_pool",
]
