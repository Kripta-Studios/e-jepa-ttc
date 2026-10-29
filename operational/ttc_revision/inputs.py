"""Bounded persistent EvTTC preparation with exact per-window representations.

Raw event cache keys are file identity and half-open times, never labels. Crops
and normalization are recomputed for each query; only identical windows within
that query share a voxel. No prepared-tensor cache hides preprocessing cost.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import torch

from e_jepa_ttc.data.garl_official_preprocessing import official_square_box
from e_jepa_ttc.efficient_context.garl_input import native_feature
from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from e_jepa_ttc.simplex_t.context_raw_union import _read_crop
from operational.evttc_transfer.inputs import H8_LAGS_US, EvTTCEventReader

ArrayMap = dict[str, np.ndarray]


class PersistentReader(EvTTCEventReader):
    """Keep dataset handles open so HDF5 chunk caches survive successive reads."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.open()
        handle = self._require_handle()
        self.datasets: dict[str, h5py.Dataset] = {}
        for key in ("x", "y", "t", "p"):
            dataset = handle[getattr(self.layout, key)]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"missing event dataset: {key}")
            self.datasets[key] = dataset

    def iter_window_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int = 250_000
    ) -> Iterator[ArrayMap]:
        """Read exact half-open slices using the original ms-index bounds."""
        if not 0 < chunk_events <= self.chunk_events_max:
            raise ValueError("invalid chunk_events")
        first, last = self._bounds(start_us, end_us)
        dtypes = {"x": np.int32, "y": np.int32, "t": np.int64, "p": np.int8}
        for cursor in range(first, last, chunk_events):
            stop = min(last, cursor + chunk_events)
            t = np.asarray(self.datasets["t"][cursor:stop], dtype=np.int64)
            keep = (t >= start_us) & (t < end_us)
            yield {
                key: (
                    t if key == "t" else np.asarray(self.datasets[key][cursor:stop], dtype=dtype)
                )[keep]
                for key, dtype in dtypes.items()
            }


def _slice(raw: ArrayMap, start: int, end: int) -> ArrayMap:
    lo, hi = np.searchsorted(raw["t"], (start, end), side="left")
    return {key: values[lo:hi] for key, values in raw.items()}


class EventPreparer:
    """Single-sequence event buffer with bounded retention and explicit close."""

    def __init__(self, *, cache_bytes: int = 128 * 1024**2) -> None:
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self.cache_bytes = cache_bytes
        self.reader: PersistentReader | None = None
        self.identity: tuple[str, int, int] | None = None
        self.raw: ArrayMap | None = None
        self.span: tuple[int, int] | None = None

    @property
    def retained_bytes(self) -> int:
        """Count retained raw column bytes, excluding bounded HDF5 internals."""
        return sum(x.nbytes for x in self.raw.values()) if self.raw is not None else 0

    def clear_raw_cache(self) -> None:
        """Discard raw retention; useful for independent-query benchmarking."""
        self.raw, self.span = None, None

    def close(self) -> None:
        """Release retained raw arrays and file/dataset handles."""
        self.clear_raw_cache()
        if self.reader is not None:
            self.reader.close()
        self.reader, self.identity = None, None

    def __enter__(self) -> EventPreparer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_reader(self, row: Mapping[str, Any]) -> PersistentReader:
        path = Path(row["raw_path"]).resolve()
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns)
        expected = row["raw_stat"]
        if (stat.st_size, stat.st_mtime_ns) != (
            int(expected["size_bytes"]),
            int(expected["mtime_ns"]),
        ):
            self.close()
            raise ValueError("raw file stat changed after query freeze")
        if self.identity != identity:
            self.close()
            self.reader = PersistentReader(path)
            self.identity = identity
        assert self.reader is not None
        return self.reader

    def _read(self, row: Mapping[str, Any], start: int, end: int) -> tuple[ArrayMap, bool]:
        self._ensure_reader(row)
        assert self.reader is not None
        hit = False
        if self.raw is not None and self.span is not None:
            first, last = self.span
            if first <= start and end <= last:
                return _slice(self.raw, start, end), True
            if first <= start < last < end:
                tail = self.reader.read_window(last, end)
                prefix = _slice(self.raw, start, last)
                raw = {key: np.concatenate((prefix[key], tail[key])) for key in prefix}
                hit = True
            else:
                raw = self.reader.read_window(start, end)
        else:
            raw = self.reader.read_window(start, end)
        if sum(x.nbytes for x in raw.values()) <= self.cache_bytes:
            self.raw, self.span = raw, (start, end)
        else:
            self.clear_raw_cache()
        return raw, hit

    def prepare(
        self, row: Mapping[str, Any], *, system: Literal["h8", "garl", "both"] = "h8"
    ) -> dict[str, Any]:
        """Construct only the selected system's tensors and same-pass CPU timings."""
        if system not in {"h8", "garl", "both"}:
            raise ValueError("unknown system")
        begun = time.perf_counter()
        windows = np.asarray(row["windows_us"], dtype=np.int64)
        anchor = int(row["anchor_us"])
        expected = np.asarray(
            [
                [anchor - 300000, anchor - 200000],
                [anchor - 200000, anchor - 100000],
                [anchor - 100000, anchor],
            ]
        )
        if windows.shape != (3, 2) or not np.array_equal(windows, expected):
            raise ValueError("windows must be three contiguous exact 100 ms endpoints")
        intervals = windows[None] - H8_LAGS_US[:, None, None]
        start = int(intervals.min()) if system != "garl" else int(windows[1, 0])
        if system == "h8" and self.cache_bytes == 0:
            reader = self._ensure_reader(row)
            square = cast(tuple[float, float, float, float], tuple(row["square_xyxy"]))
            cropped = _read_crop(reader, start, anchor, square, 0.0, 256 * 1024**2)  # type: ignore[arg-type]
            if cropped is None:
                raise MemoryError("ROI union exceeds 256 MiB; reduce event acquisition rate")
            raw, hit = cropped, False
        else:
            raw, hit = self._read(row, start, anchor)
        read_end = time.perf_counter()
        result: dict[str, Any] = {}
        unique: dict[tuple[int, int], torch.Tensor] = {}
        if system != "garl":
            # Pre-filter integer coordinates before allocating float64 projection arrays.
            x0, y0, x1, y1 = row["square_xyxy"]
            keep = (
                (raw["x"] >= max(0, x0))
                & (raw["x"] < min(1280, x1))
                & (raw["y"] >= max(0, y0))
                & (raw["y"] < min(720, y1))
            )
            cropped = {key: value[keep] for key, value in raw.items()}
            mapped = map_roi(cropped, tuple(row["square_xyxy"]), 128, 0.0)
            own = torch.empty((8, 3, 12, 128, 128), dtype=torch.float32)
            for slot, group in enumerate(intervals):
                for index, (first, last) in enumerate(group):
                    key = (int(first), int(last))
                    if key not in unique:
                        unique[key] = encode_mapped(mapped, *key, 128, str(row["sequence_id"]))
                    own[slot, index] = unique[key]
            result.update(
                own_events=own.numpy(), valid=np.ones(8, dtype=bool), delta_t_s=np.float32(0.1)
            )
        voxel_end = time.perf_counter()
        if system != "h8":
            boxes = [tuple(float(v) for v in b) for b in row["boxes_xyxy3"][-2:]]
            if len(boxes) != 2 or any(len(box) != 4 for box in boxes):
                raise ValueError("two Garl boxes required")
            boxes = cast(list[tuple[float, float, float, float]], boxes)
            parts = []
            for index, (first, last) in enumerate(windows[-2:]):
                endpoint = _slice(raw, int(first), int(last))
                if not len(endpoint["t"]):
                    break
                parts.append(native_feature(endpoint, official_square_box(boxes, index), 128))
            result["garl_events"] = torch.cat(parts).numpy() if len(parts) == 2 else None
            result["garl_unavailable_reason"] = None if len(parts) == 2 else "empty_sensor_endpoint"
        finished = time.perf_counter()
        result["diagnostics"] = {
            "requested_span_us": anchor - start,
            "raw_cache_hit": hit,
            "retained_bytes": self.retained_bytes,
            "unique_voxel_windows": len(unique),
            "read_ms": (read_end - begun) * 1000,
            "h8_voxel_ms": (voxel_end - read_end) * 1000,
            "garl_repr_ms": (finished - voxel_end) * 1000,
            "cpu_total_ms": (finished - begun) * 1000,
        }
        return result
