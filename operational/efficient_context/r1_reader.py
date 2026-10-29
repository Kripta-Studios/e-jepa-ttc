"""Bounded causal sensor replay; no ROI, targets, or model outputs are cached."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from typing import Protocol

import numpy as np


class ChunkSource(Protocol):
    """Read exact half-open intervals in original sensor order."""

    def iter_window_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int = 250_000
    ) -> Iterator[dict[str, np.ndarray]]: ...


class ResidentReplay:
    """Retain a chronological suffix, falling back to causal reads on overflow.

    Each advance ingests every event since the preceding cutoff, including gaps.
    The initial replay segment begins at its first requested history start, not
    the beginning of the recording. Memory capacity counts owned sensor arrays;
    a source chunk and the caller's representation are additional allocations.
    """

    def __init__(self, source: ChunkSource, capacity_bytes: int = 64 * 1024**2) -> None:
        if capacity_bytes < 1:
            raise ValueError("positive capacity required")
        self.source = source
        self.capacity_bytes = capacity_bytes
        self.chunks: deque[dict[str, np.ndarray]] = deque()
        self.cutoff: int | None = None
        self.coverage_start: int | None = None
        self.retained_bytes = 0
        self.stats: dict[str, float] = dict(
            ingestion_ms=0,
            ingested_events=0,
            ingested_bytes=0,
            evicted_bytes=0,
            capacity_fallback_reads=0,
            resident_reads=0,
            stream_seconds=0,
            peak_retained_bytes=0,
        )

    def _evict(self, floor: int) -> None:
        while self.chunks and int(self.chunks[0]["t"][-1]) < floor:
            removed = self.chunks.popleft()
            size = sum(v.nbytes for v in removed.values())
            self.retained_bytes -= size
            self.stats["evicted_bytes"] += size
        if self.chunks and int(self.chunks[0]["t"][0]) < floor:
            old = self.chunks.popleft()
            lo = int(np.searchsorted(old["t"], floor, side="left"))
            trimmed = {k: v[lo:].copy() for k, v in old.items()}
            reduction = sum(v.nbytes for v in old.values()) - sum(
                v.nbytes for v in trimmed.values()
            )
            self.retained_bytes -= reduction
            self.stats["evicted_bytes"] += reduction
            self.chunks.appendleft(trimmed)

    def advance(self, start_us: int, cutoff_us: int) -> None:
        """Ingest up to, never including, cutoff; reject chronological rollback."""
        if start_us < 0 or cutoff_us <= start_us:
            raise ValueError("positive half-open replay interval required")
        if self.cutoff is not None and cutoff_us < self.cutoff:
            raise ValueError("query cutoff rollback")
        begin = time.perf_counter()
        first = start_us if self.cutoff is None else self.cutoff
        self._evict(start_us)
        if self.coverage_start is None:
            self.coverage_start = start_us
        self.coverage_start = max(self.coverage_start, start_us)
        previous: int | None = None
        if self.chunks:
            previous = int(self.chunks[-1]["t"][-1])
        if first < cutoff_us:
            for raw in self.source.iter_window_chunks(first, cutoff_us):
                t = raw["t"]
                if any(v.ndim != 1 or len(v) != len(t) for v in raw.values()):
                    raise ValueError("unaligned sensor columns")
                if len(t) == 0:
                    continue
                if (
                    (np.diff(t) < 0).any()
                    or int(t[0]) < first
                    or int(t[-1]) >= cutoff_us
                    or (previous is not None and int(t[0]) < previous)
                ):
                    raise ValueError("timestamp rollback or noncausal source")
                previous = int(t[-1])
                self.stats["ingested_events"] += len(t)
                self.stats["ingested_bytes"] += sum(v.nbytes for v in raw.values())
                lo = int(np.searchsorted(t, start_us, side="left"))
                if lo == len(t):
                    continue
                chunk = {k: v[lo:].copy() for k, v in raw.items()}
                size = sum(v.nbytes for v in chunk.values())
                if size > self.capacity_bytes:
                    # Do not retain an oversized chunk or pretend the interval is cached.
                    self.stats["evicted_bytes"] += self.retained_bytes + size
                    self.chunks.clear()
                    self.retained_bytes = 0
                    self.coverage_start = int(t[-1]) + 1
                    continue
                while self.chunks and self.retained_bytes + size > self.capacity_bytes:
                    removed = self.chunks.popleft()
                    nbytes = sum(v.nbytes for v in removed.values())
                    self.retained_bytes -= nbytes
                    self.stats["evicted_bytes"] += nbytes
                    self.coverage_start = max(self.coverage_start, int(removed["t"][-1]) + 1)
                self.chunks.append(chunk)
                self.retained_bytes += size
                self.stats["peak_retained_bytes"] = max(
                    self.stats["peak_retained_bytes"], self.retained_bytes
                )
            self.stats["stream_seconds"] += (cutoff_us - first) / 1e6
        self.cutoff = cutoff_us
        self.stats["ingestion_ms"] += (time.perf_counter() - begin) * 1000

    def iter_window_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int = 250_000
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield sensor views; fall back without looking past the admitted cutoff."""
        if start_us < 0 or end_us <= start_us or chunk_events < 1:
            raise ValueError("invalid half-open request")
        if self.cutoff is None or end_us > self.cutoff:
            raise ValueError("request exceeds ingested causal cutoff")
        if self.coverage_start is None or start_us < self.coverage_start:
            self.stats["capacity_fallback_reads"] += 1
            yield from self.source.iter_window_chunks(start_us, end_us, chunk_events=chunk_events)
            return
        self.stats["resident_reads"] += 1
        for chunk in self.chunks:
            lo, hi = np.searchsorted(chunk["t"], [start_us, end_us], side="left")
            for cursor in range(int(lo), int(hi), chunk_events):
                stop = min(int(hi), cursor + chunk_events)
                yield {k: v[cursor:stop] for k, v in chunk.items()}
