"""Bounded parallel prepared-RGB reads with the complete canonical validation.

Only the coordinator mutates the admitted LRU. Workers use private reader shells
to execute the original SHA256, NPZ/CRC, token and row decoding path unchanged.
Pending reads are bounded by count and a conservative uncompressed-byte reserve.
"""

# ruff: noqa: ANN401 -- adapter for the frozen prepared-cache/source interfaces
from __future__ import annotations

import time
import zipfile
from collections import OrderedDict, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock
from types import MethodType
from typing import Any


class ParallelPreparedCache:
    """Prefetch only requested cold shards; preserve original access/LRU order."""

    def __init__(self, source: Any, *, workers: int, max_pending_bytes: int) -> None:
        if workers < 1 or max_pending_bytes < 1:
            raise ValueError("Positive decoder worker and pending-byte limits required")
        self.source = source
        self.original = source.dataset.prepared_cache
        if self.original is None:
            raise ValueError("Parallel decoding requires the admitted prepared RGB cache")
        self.workers, self.max_pending_bytes = workers, max_pending_bytes
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rgb-decode")
        self.pending: dict[str, tuple[Future[Any], int]] = {}
        self.remaining: deque[tuple[str, int, str, int]] = deque()
        self.reservations: dict[str, int] = {}
        self.lock = Lock()
        self.closed = False
        self.pending_bytes = self.peak_pending_bytes = self.peak_pending_reads = 0
        self.reads = self.fallback_reads = self.hits = 0
        self.decode_seconds = self.wait_seconds = 0.0
        self.largest_decoded_bytes = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)

    def _reservation(self, key: str) -> int:
        existing = self.reservations.get(key)
        if existing is not None:
            return existing
        with zipfile.ZipFile(key) as archive:
            raw_bytes = sum(item.file_size for item in archive.infolist())
        # Arrays, member decoding buffers, one returned RGB float row, hashing
        # buffers and small metadata are covered conservatively for this schema.
        reserve = 4 * raw_bytes + 2 * 1024**2
        self.reservations[key] = reserve
        return reserve

    def _read(self, index: int, token: str, key: str) -> Any:
        started = time.perf_counter()
        shell = object.__new__(type(self.original))
        shell.__dict__.update(vars(self.original))
        shell._loaded = OrderedDict()
        shell._loaded_bytes = 0
        try:
            # Use the exact canonical implementation instead of reproducing its
            # validation or replacing it with trusted-file shortcuts.
            shell.get(index, token)
            stored = shell._loaded[key]
            size = sum(array.nbytes for array in stored.values())
            with self.lock:
                self.reads += 1
                self.largest_decoded_bytes = max(self.largest_decoded_bytes, size)
            return stored
        finally:
            with self.lock:
                self.decode_seconds += time.perf_counter() - started

    def _fill(self) -> None:
        while self.remaining and len(self.pending) < self.workers:
            key, index, token, reserve = self.remaining[0]
            if self.pending_bytes + reserve > self.max_pending_bytes:
                break
            self.remaining.popleft()
            self.pending[key] = (self.pool.submit(self._read, index, token, key), reserve)
            self.pending_bytes += reserve
            self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
            self.peak_pending_reads = max(self.peak_pending_reads, len(self.pending))

    def plan(self, indices: Sequence[int]) -> None:
        """Schedule missing unique shards without changing sample or batch order."""
        if self.closed or self.pending or self.remaining:
            raise RuntimeError("Decoder is closed or its previous plan was not drained")
        seen: set[str] = set()
        for index in indices:
            item = self.original._lookup.get(int(index))
            if item is None:
                continue  # canonical get reports the original missing-row error
            key = str(item[0]["path"])
            if key in seen or key in self.original._loaded:
                continue
            seen.add(key)
            try:
                reserve = self._reservation(key)
            except (OSError, zipfile.BadZipFile):
                # Let get perform the canonical read inside the outer admitted
                # retry boundary, rather than failing outside that boundary.
                continue
            if reserve > self.max_pending_bytes:
                continue  # oversized shards remain on the canonical sync route
            token = str(self.source.dataset.rows[int(index)]["sample_token"])
            self.remaining.append((key, int(index), token, reserve))
        self._fill()

    def get(self, index: int, token: str) -> Any:
        """Adopt validated arrays in the original LRU order, then use original get."""
        if self.closed:
            raise RuntimeError("Decoder is closed")
        item = self.original._lookup.get(index)
        if item is None:
            return self.original.get(index, token)
        key = str(item[0]["path"])
        if key in self.original._loaded:
            self.hits += 1
            return self.original.get(index, token)
        future_and_reserve = self.pending.pop(key, None)
        if future_and_reserve is None:
            self.fallback_reads += 1
            return self.original.get(index, token)
        future, reserve = future_and_reserve
        started = time.perf_counter()
        try:
            stored = future.result()
        finally:
            self.wait_seconds += time.perf_counter() - started
            self.pending_bytes -= reserve
            self._fill()
        self.original._loaded[key] = stored
        self.original._loaded_bytes += sum(array.nbytes for array in stored.values())
        while (
            self.original._loaded_bytes > self.original._max_loaded_bytes
            and len(self.original._loaded) > 1
        ):
            _, stale = self.original._loaded.popitem(last=False)
            self.original._loaded_bytes -= sum(array.nbytes for array in stale.values())
        return self.original.get(index, token)

    def drain(self) -> None:
        """Join abandoned work and drop errors not encountered by canonical get."""
        for future, _ in self.pending.values():
            if not future.cancel():
                try:
                    future.result()
                except Exception:
                    pass  # a later canonical access must retry its actual file
        self.pending.clear()
        self.remaining.clear()
        self.pending_bytes = 0

    def snapshot(self) -> dict[str, Any]:
        """Expose counters and reserved-byte bounds, not a process RSS claim."""
        return {
            "workers": self.workers,
            "max_pending_bytes": self.max_pending_bytes,
            "peak_pending_bytes": self.peak_pending_bytes,
            "peak_pending_reads": self.peak_pending_reads,
            "parallel_reads": self.reads,
            "fallback_reads": self.fallback_reads,
            "hits": self.hits,
            "decode_worker_seconds": self.decode_seconds,
            "consumer_wait_seconds": self.wait_seconds,
            "largest_decoded_shard_bytes": self.largest_decoded_bytes,
            "rgb_cache_bytes": self.original._loaded_bytes,
        }

    def close(self) -> None:
        """Join decoder workers while leaving original validated cached arrays live."""
        self.drain()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.closed = True


@contextmanager
def installed_decoder(
    source: Any, *, workers: int = 2, max_pending_bytes: int = 128 * 1024**2
) -> Iterator[ParallelPreparedCache]:
    """Install inside the existing RGB loader and restore both instance methods."""
    cache = ParallelPreparedCache(source, workers=workers, max_pending_bytes=max_pending_bytes)
    original_cache = source.dataset.prepared_cache
    original_batch = source.batch
    had_override = "batch" in vars(source)
    previous = vars(source).get("batch")

    def batch(_self: Any, indices: Sequence[int], modality: str) -> Any:
        cache.plan(indices)
        try:
            return original_batch(indices, modality)
        finally:
            cache.drain()

    source.dataset.prepared_cache = cache
    source.batch = MethodType(batch, source)
    try:
        yield cache
    finally:
        cache.close()
        source.dataset.prepared_cache = original_cache
        if had_override:
            source.batch = previous
        else:
            del source.batch
