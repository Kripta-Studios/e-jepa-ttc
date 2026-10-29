"""Bounded persistent HDF5 handles; identical historical event slicing."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from e_jepa_ttc.data.eap import EAPEventReader, _require_hdf5plugin


class CachedEventReader(EAPEventReader):
    """Keep dataset objects alive so HDF5 can reuse decompressed chunks.

    No event sampling, reordering, timestamp rounding or dtype changes. The
    8 MiB per-dataset cache bounds four event columns plus the time index at
    40 MiB nominal; h5py metadata and output arrays are additional allocations.
    """

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self.datasets: dict[str, h5py.Dataset] = {}

    def open(self) -> None:
        """Open read-only and retain exactly the five historical datasets."""
        if self._handle is not None:
            return
        _require_hdf5plugin()
        self._handle = h5py.File(self.path, "r", rdcc_nbytes=8 * 1024**2)
        required = {"events/x", "events/y", "events/t", "events/p", "ms_to_idx"}
        missing = sorted(name for name in required if name not in self._handle)
        if missing:
            self.close()
            raise ValueError(f"{self.path} is missing eAP event datasets: {missing}.")
        self.datasets = {name: cast(h5py.Dataset, self._handle[name]) for name in required}

    def close(self) -> None:
        """Release dataset references before closing their owner."""
        self.datasets.clear()
        super().close()

    def iter_window_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int = 250_000
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield the same half-open chunks and dtypes as EAPEventReader."""
        if start_us < 0 or end_us <= start_us:
            raise ValueError("Event window requires 0 <= start_us < end_us.")
        if chunk_events <= 0:
            raise ValueError("chunk_events must be positive.")
        self.open()
        index = self.datasets["ms_to_idx"]
        maximum_ms = int(index.shape[0] - 1)
        start_ms = min(maximum_ms, max(0, start_us // 1000))
        end_ms = min(maximum_ms, max(0, math.ceil(end_us / 1000)))
        first, last = int(index[start_ms]), int(index[end_ms])
        x, y, t, p = (self.datasets[f"events/{key}"] for key in ("x", "y", "t", "p"))
        for cursor in range(first, last, chunk_events):
            stop = min(last, cursor + chunk_events)
            timestamps = np.asarray(t[cursor:stop], dtype=np.int64)
            exact = (timestamps >= start_us) & (timestamps < end_us)
            yield {
                "x": np.asarray(x[cursor:stop], dtype=np.int32)[exact],
                "y": np.asarray(y[cursor:stop], dtype=np.int32)[exact],
                "t": timestamps[exact],
                "p": np.asarray(p[cursor:stop], dtype=np.int8)[exact],
            }


class ReaderPool:
    """One open sequence only, owned by one replay process and thread."""

    def __init__(self) -> None:
        self.reader: CachedEventReader | None = None

    def get(self, path: str | Path) -> CachedEventReader:
        """Reuse the current sequence; close it before opening another."""
        path = Path(path).resolve(strict=True)
        if self.reader is None or self.reader.path != path:
            self.close()
            self.reader = CachedEventReader(path)
            self.reader.open()
        return self.reader

    def close(self) -> None:
        """Release the sole reader, including on exceptions."""
        if self.reader is not None:
            self.reader.close()
            self.reader = None
