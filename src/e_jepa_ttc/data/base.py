"""Protocols shared by event readers and temporal-window datasets."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from e_jepa_ttc.data.types import EventBatch, TTCWindowSample


class EventReader(Protocol):
    """Minimal reader contract used by representations and streaming code."""

    def read_window(self, start_us: int, end_us: int) -> EventBatch:
        """Read the half-open event interval ``[start_us, end_us)``."""

        ...


class WindowDataset(Protocol):
    """Dataset contract for causal TTC windows."""

    def __len__(self) -> int:
        """Return the number of indexed windows."""

        ...

    def __getitem__(self, index: int) -> TTCWindowSample:
        """Return one window without crossing sequence boundaries."""

        ...


def assert_nonempty_events(events: Iterable[EventBatch]) -> None:
    """Fail explicitly when a batch unexpectedly contains no event windows."""

    if not any(batch.num_events > 0 for batch in events):
        raise ValueError("The event collection contains no non-empty window.")


__all__ = ["EventReader", "WindowDataset", "assert_nonempty_events"]
