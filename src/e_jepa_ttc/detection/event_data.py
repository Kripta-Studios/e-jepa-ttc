"""Bounded, causal event representation for RGB detector fusion.

This module is intentionally independent of the historical detector and SSL
sampler.  It ends every event window at RGB exposure end, streams every event
without a count cap, and retains at most two HDF5 readers.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.representations.voxel_grid import robust_normalize

SOURCE_SIZE = (1280, 720)


class EventReader(Protocol):
    """Minimal interface needed from an indexed event reader."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def iter_window_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int
    ) -> Iterable[dict[str, np.ndarray]]: ...


@dataclass(frozen=True)
class CausalEventConfig:
    """Fixed S4 representation and resource contract."""

    width: int = 640
    height: int = 384
    window_ms: int = 50
    bins_per_polarity: int = 5
    chunk_events: int = 250_000
    max_open_handles: int = 2
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.window_ms not in (50, 100):
            raise ValueError("S4 event windows must be 50 or 100 ms")
        if self.bins_per_polarity != 5:
            raise ValueError("S4 requires exactly five bins per polarity")
        if min(self.width, self.height, self.chunk_events, self.max_open_handles) <= 0:
            raise ValueError("Event representation dimensions and bounds must be positive")
        if self.max_open_handles > 2:
            raise ValueError("S4 reader cache is bounded to at most two HDF5 handles")


@dataclass(frozen=True)
class EventVoxelResult:
    """One dense event tensor and its auditable bounded-read receipt."""

    voxel: np.ndarray
    receipt: dict[str, Any]


def resize_event_coordinates_pixel_center(
    x: np.ndarray,
    y: np.ndarray,
    *,
    source_size: tuple[int, int] = SOURCE_SIZE,
    output_size: tuple[int, int] = (640, 384),
) -> tuple[np.ndarray, np.ndarray]:
    """Map source pixel centers to the nearest resized RGB pixel deterministically.

    Pillow's inverse resize convention is ``src=(dst+0.5)/scale-0.5``.
    The corresponding source-to-destination center is rounded to the nearest
    pixel using exact integer arithmetic.
    """
    source_width, source_height = source_size
    output_width, output_height = output_size
    if min(source_width, source_height, output_width, output_height) <= 0:
        raise ValueError("Source and output dimensions must be positive")
    x_array = np.asarray(x)
    y_array = np.asarray(y)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.shape != y_array.shape:
        raise ValueError("Event coordinates must be aligned one-dimensional arrays")
    if not np.issubdtype(x_array.dtype, np.integer) or not np.issubdtype(y_array.dtype, np.integer):
        raise ValueError("Event coordinates must be integer-valued")
    x_int, y_int = x_array.astype(np.int64), y_array.astype(np.int64)
    if (
        np.any(x_int < 0)
        or np.any(x_int >= source_width)
        or np.any(y_int < 0)
        or np.any(y_int >= source_height)
    ):
        raise ValueError("Event coordinate is outside the pinned 1280x720 plane")
    resized_x = ((2 * x_int + 1) * output_width // (2 * source_width)).astype(np.int32)
    resized_y = ((2 * y_int + 1) * output_height // (2 * source_height)).astype(np.int32)
    return resized_x, resized_y


def _float32_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def encode_causal_event_chunks(
    chunks: Iterable[Mapping[str, np.ndarray]],
    *,
    sequence_id: str,
    cutoff_us: int,
    config: CausalEventConfig,
) -> EventVoxelResult:
    """Accumulate all events in ``[cutoff-window, cutoff)`` into ten channels."""
    if not sequence_id:
        raise ValueError("sequence_id must be nonempty")
    if cutoff_us < config.window_ms * 1000:
        raise ValueError("cutoff_us cannot provide the configured causal history")
    start_us = cutoff_us - config.window_ms * 1000
    channels = config.bins_per_polarity * 2
    # Accumulate exact integer numerators and divide once after every chunk.
    # The linear temporal weights have the common integer denominator
    # ``window_us``.  Integer accumulation makes the result independent of how
    # the reader partitions an otherwise identical ordered event stream.
    window_us = config.window_ms * 1000
    raw_numerator = np.zeros((channels, config.height, config.width), dtype=np.int64)
    event_count = 0
    chunk_count = 0
    maximum_chunk_events = 0
    previous_timestamp: int | None = None
    for chunk in chunks:
        if set(chunk) != {"x", "y", "t", "p"}:
            raise ValueError("Event chunks must contain exactly x/y/t/p")
        x = np.asarray(chunk["x"])
        y = np.asarray(chunk["y"])
        timestamps = np.asarray(chunk["t"])
        polarity = np.asarray(chunk["p"])
        if not (x.ndim == y.ndim == timestamps.ndim == polarity.ndim == 1):
            raise ValueError("Event chunk arrays must be one-dimensional")
        if not (x.shape == y.shape == timestamps.shape == polarity.shape):
            raise ValueError("Event chunk arrays must be aligned")
        if len(x) > config.chunk_events:
            raise ValueError("Reader exceeded the configured chunk event bound")
        if len(x) == 0:
            continue
        if not np.issubdtype(timestamps.dtype, np.integer):
            raise ValueError("Event timestamps must be integer microseconds")
        timestamps = timestamps.astype(np.int64)
        if np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("Event timestamps rolled back within a chunk")
        if previous_timestamp is not None and int(timestamps[0]) < previous_timestamp:
            raise ValueError("Event timestamps rolled back across chunks")
        if int(timestamps[0]) < start_us or int(timestamps[-1]) >= cutoff_us:
            raise ValueError("Reader returned an event outside the causal half-open window")
        if not np.issubdtype(polarity.dtype, np.integer) or not np.all(np.isin(polarity, (0, 1))):
            raise ValueError("Pinned eAP polarity must contain only raw values 0 and 1")
        resized_x, resized_y = resize_event_coordinates_pixel_center(
            x,
            y,
            source_size=SOURCE_SIZE,
            output_size=(config.width, config.height),
        )
        temporal_numerator = (timestamps - start_us) * (config.bins_per_polarity - 1)
        lower = temporal_numerator // window_us
        upper_remainder = temporal_numerator % window_us
        upper = np.minimum(lower + 1, config.bins_per_polarity - 1)
        polarity_offset = np.where(polarity > 0, 0, config.bins_per_polarity)
        spatial = resized_y.astype(np.int64) * config.width + resized_x.astype(np.int64)
        pixels = config.height * config.width
        lower_flat = (lower + polarity_offset) * pixels + spatial
        upper_flat = (upper + polarity_offset) * pixels + spatial
        # Each per-chunk weighted sum is an integer below 2**53 at the pinned
        # bounds, so NumPy's float64 bincount is exact before the int64 cast.
        # Combining those integer histograms is partition independent.
        raw_flat = raw_numerator.reshape(-1)
        flat_size = raw_flat.size
        lower_counts = np.bincount(
            lower_flat,
            weights=window_us - upper_remainder,
            minlength=flat_size,
        )
        raw_flat += lower_counts.astype(np.int64)
        upper_counts = np.bincount(
            upper_flat,
            weights=upper_remainder,
            minlength=flat_size,
        )
        raw_flat += upper_counts.astype(np.int64)
        event_count += len(x)
        chunk_count += 1
        maximum_chunk_events = max(maximum_chunk_events, len(x))
        previous_timestamp = int(timestamps[-1])
    raw = raw_numerator.astype(np.float64) / window_us
    voxel = robust_normalize(raw) if config.normalize else raw.astype(np.float32)
    receipt: dict[str, Any] = {
        "schema": "eap_causal_event_voxel_v2",
        "sequence_id": sequence_id,
        "window_start_us_inclusive": start_us,
        "window_end_us_exclusive": cutoff_us,
        "event_count": event_count,
        "nonempty_chunk_count": chunk_count,
        "maximum_chunk_events_observed": maximum_chunk_events,
        "voxel_shape": list(voxel.shape),
        "voxel_dtype": str(voxel.dtype),
        "voxel_sha256_float32_le": _float32_sha256(voxel),
        "config": asdict(config),
        "source_size_wh": list(SOURCE_SIZE),
        "coordinate_mapping": "floor((2*coord+1)*dst/(2*src)); Pillow pixel-center nearest",
        "temporal_mapping": "linear interpolation over five bins per polarity",
        "accumulation": "exact int64 common-denominator numerators; one final float cast",
        "raw_polarity_mapping": "0 -> negative channels; 1 -> positive channels",
        "events_discarded_by_cap": 0,
        "cutoff_source": "caller_supplied_timestamp_us",
        "uses_future_events": False,
    }
    return EventVoxelResult(voxel=voxel, receipt=receipt)


class CausalEventVoxelizer:
    """Resolve train HDF5 paths and retain an LRU of at most two readers."""

    def __init__(
        self,
        root: str | Path,
        config: CausalEventConfig,
        *,
        reader_factory: Callable[[Path], EventReader] = EAPEventReader,
    ) -> None:
        self.root = Path(root).resolve()
        self.config = config
        self.reader_factory = reader_factory
        self._readers: OrderedDict[Path, EventReader] = OrderedDict()

    @property
    def open_reader_paths(self) -> tuple[Path, ...]:
        """Expose cache identity for resource assertions, not event content."""
        return tuple(self._readers)

    def _reader(self, relative_path: str | Path) -> EventReader:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Event path must be relative and remain under the dataset root")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Event path is absent or escapes the dataset root")
        reader = self._readers.pop(path, None)
        if reader is None:
            # Evict before opening so the live handle count never transiently
            # exceeds the configured maximum.
            while len(self._readers) >= self.config.max_open_handles:
                _, oldest = self._readers.popitem(last=False)
                oldest.close()
            reader = self.reader_factory(path)
            reader.open()
        self._readers[path] = reader
        return reader

    def encode(
        self,
        relative_path: str | Path,
        *,
        sequence_id: str,
        rgb_exposure_end_us: int,
    ) -> EventVoxelResult:
        """Read and encode exactly one causal event window ending at RGB exposure end."""
        duration_us = self.config.window_ms * 1000
        reader = self._reader(relative_path)
        chunks = reader.iter_window_chunks(
            rgb_exposure_end_us - duration_us,
            rgb_exposure_end_us,
            chunk_events=self.config.chunk_events,
        )
        result = encode_causal_event_chunks(
            chunks,
            sequence_id=sequence_id,
            cutoff_us=rgb_exposure_end_us,
            config=self.config,
        )
        result.receipt["cutoff_source"] = "rgb_exposure_end_timestamp_us"
        result.receipt["events_path"] = Path(relative_path).as_posix()
        result.receipt["open_reader_count_after_read"] = len(self._readers)
        return result

    def close(self) -> None:
        """Close all retained HDF5 handles."""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> CausalEventVoxelizer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
