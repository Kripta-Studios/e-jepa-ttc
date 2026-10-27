"""Byte-bounded extension of the frozen deterministic OOF prefetch helper."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Generic

from operational.oof_prefetch import (
    BatchT,
    BoundedFoldPrefetch,
    BuildBatch,
    RecipeT,
    SourceT,
    _indices,
)


class SharedByteBudget:
    """Reservation pool shared by all input-side helpers in one worker."""

    def __init__(self, limit_bytes: int) -> None:
        if not isinstance(limit_bytes, int) or isinstance(limit_bytes, bool) or limit_bytes <= 0:
            raise ValueError("shared byte limit must be a positive integer")
        self.limit_bytes = limit_bytes
        self.reserved_bytes = 0
        self.peak_reserved_bytes = 0
        self._lock = threading.Lock()

    def reserve(self, value: int) -> bool:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("reservation must be a positive integer")
        with self._lock:
            if self.reserved_bytes + value > self.limit_bytes:
                return False
            self.reserved_bytes += value
            self.peak_reserved_bytes = max(self.peak_reserved_bytes, self.reserved_bytes)
            return True

    def release(self, value: int) -> None:
        with self._lock:
            if value <= 0 or value > self.reserved_bytes:
                raise RuntimeError("byte reservation accounting underflow")
            self.reserved_bytes -= value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "limit_bytes": self.limit_bytes,
                "reserved_bytes": self.reserved_bytes,
                "peak_reserved_bytes": self.peak_reserved_bytes,
            }


def retained_nbytes(value: object, seen: set[int] | None = None) -> int:
    """Count array/tensor storage reachable from one prepared batch."""
    visited = set() if seen is None else seen
    if id(value) in visited:
        return 0
    visited.add(id(value))
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return int(value.nbytes)
    except ImportError:
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return int(value.numel() * value.element_size())
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return sum(retained_nbytes(item, visited) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(retained_nbytes(item, visited) for item in value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return retained_nbytes(attributes, visited)
    return len(value) if isinstance(value, (bytes, bytearray)) else 0


class ByteBoundedFoldPrefetch(
    BoundedFoldPrefetch[SourceT, RecipeT, BatchT], Generic[SourceT, RecipeT, BatchT]
):
    """Preserve frozen queue semantics while reserving memory before submission."""

    def __init__(
        self,
        source: SourceT,
        recipe: RecipeT,
        cursor: Mapping[str, object],
        build_batch: BuildBatch[SourceT, RecipeT, BatchT],
        *,
        depth: int,
        byte_budget: SharedByteBudget,
        reservation_bytes_per_batch: int,
    ) -> None:
        if (
            reservation_bytes_per_batch <= 0
            or reservation_bytes_per_batch > byte_budget.limit_bytes
        ):
            raise ValueError("invalid per-batch byte reservation")
        self.byte_budget = byte_budget
        self.reservation_bytes_per_batch = reservation_bytes_per_batch
        super().__init__(source, recipe, cursor, build_batch, depth=depth)

    def _prepare(self, ids: tuple[int, ...]) -> BatchT:
        result = super()._prepare(ids)
        if retained_nbytes(result) > self.reservation_bytes_per_batch:
            raise MemoryError("prepared batch exceeds its admitted byte reservation")
        return result

    def _fill(self) -> None:
        while len(self.queue) < self.depth and self.next_position < self.population:
            start = self.next_position
            stop = min(start + self.batch_size, self.population)
            ids = self.order[start:stop]
            if not ids:
                raise RuntimeError("prefetch attempted to submit an empty effective batch")
            reserve = self.reservation_bytes_per_batch
            if not self.byte_budget.reserve(reserve):
                break
            try:
                future = self.pool.submit(self._prepare, ids)
            except BaseException:
                self.byte_budget.release(reserve)
                raise
            self.queue.append((start, ids, future, reserve))  # type: ignore[arg-type]
            self.next_position = stop
            self.submitted += 1
            self.max_retained = max(self.max_retained, len(self.queue))

    def _discard_queue(self) -> None:
        while self.queue:
            _, _, future, reserve = self.queue.popleft()  # type: ignore[misc]
            if future.cancel():
                self.cancelled += 1
                self.byte_budget.release(reserve)
                continue
            try:
                future.result()
            except BaseException:
                pass
            finally:
                self.byte_budget.release(reserve)
            self.drained += 1

    def get(self, indices: Sequence[int]) -> BatchT:
        if self.closed:
            raise RuntimeError("OOF prefetch is closed")
        if self.failure is not None:
            raise RuntimeError("OOF prefetch previously failed") from self.failure
        try:
            self._synchronize_epoch()
            if not self.queue:
                raise MemoryError("shared byte budget cannot admit the next OOF batch")
            start, expected, future, reserve = self.queue[0]  # type: ignore[misc]
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
                self.byte_budget.release(reserve)
            self.consumed += 1
            self._fill()
            return result
        except BaseException as error:
            self.errors += 1
            self._abort(error)
            raise

    def snapshot(self) -> dict[str, Any]:
        return {
            **super().snapshot(),
            "reservation_bytes_per_batch": self.reservation_bytes_per_batch,
            "shared_byte_budget": self.byte_budget.snapshot(),
        }


__all__ = ["ByteBoundedFoldPrefetch", "SharedByteBudget", "retained_nbytes"]
