"""Bounded lossless raw sensor cache; all requested intervals remain half-open."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np


class SensorCache:
    """Cache 100ms decoded chunks and 16 frames, without altering preprocessing."""

    def __init__(self, source: Any, *, max_event_bytes: int = 64 * 1024**2) -> None:  # noqa: ANN401 -- transparent sensor facade
        self.source = source
        self.max_event_bytes = max_event_bytes
        self.events: OrderedDict[int, np.ndarray] = OrderedDict()
        self.frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self.event_bytes = 0
        self.hits = 0
        self.misses = 0

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- delegate sensor metadata
        return getattr(self.source, name)

    def read_events(self, start_us: int, end_us: int) -> np.ndarray:
        """Return exactly the requested interval, regardless of prefetched bytes."""
        if end_us < start_us:
            raise ValueError("End precedes start")
        if end_us == start_us:
            return np.empty((0, 4), dtype=np.int64)
        pieces = []
        for block in range(start_us // 100000, (end_us - 1) // 100000 + 1):
            if block in self.events:
                self.hits += 1
                value = self.events.pop(block)
                self.events[block] = value
            else:
                self.misses += 1
                value = self.source.read_events(block * 100000, (block + 1) * 100000)
                if value.nbytes <= self.max_event_bytes:
                    while self.events and self.event_bytes + value.nbytes > self.max_event_bytes:
                        _, old = self.events.popitem(last=False)
                        self.event_bytes -= old.nbytes
                    self.events[block] = value
                    self.event_bytes += value.nbytes
            lo, hi = np.searchsorted(value[:, 2], [start_us, end_us])
            pieces.append(value[lo:hi])
        return np.concatenate(pieces)

    def read_rgb(self, index: int) -> np.ndarray:
        """Keep detached frame semantics even when serving a cache hit."""
        if index in self.frames:
            value = self.frames.pop(index)
        else:
            value = self.source.read_rgb(index)
        self.frames[index] = value
        while len(self.frames) > 16:
            self.frames.popitem(last=False)
        return value.copy()
