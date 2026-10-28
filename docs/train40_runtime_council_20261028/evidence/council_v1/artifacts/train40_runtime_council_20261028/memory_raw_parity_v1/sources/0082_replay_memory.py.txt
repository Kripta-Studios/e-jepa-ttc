"""Bounded offline RAW feeder for paced arrivals at the CPU input boundary.

Disk reads are startup work in this explicitly separate replay scope. Future RAW
bytes remain private to the feeder. Selection, event conversion/copy, new-frame
copy/checksum, causal annotation parsing and payload construction run on each
scheduled acquisition. This is neither a storage benchmark nor a physical sensor.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np
from numpy.typing import DTypeLike

from operational.evttc_transfer.inputs import (
    EvTTCEventReader,
    _calibration,
    _calibration_sha256,
    _canonical_sha256,
    _single_object_bbox,
)
from operational.rgb_port_revision.transfer_inputs import sensor_triplets
from operational.train40_runtime.rgb_immutable_history import FrozenRGBHistory
from operational.train40_runtime.rgb_raw_ring import (
    FrozenRGBFrame,
    RGBSourceIdentity,
    freeze_rgb_frame,
)

SCOPE = "PACED_PRELOADED_RAW_CPU_ARRIVALS_NOT_PHYSICAL_SENSOR"
STEP_US = 20_000
EVENT_DTYPES = {"x": np.int32, "y": np.int32, "t": np.int64, "p": np.int8}
_MIB = 1024**2


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _readonly_copy(value: np.ndarray, dtype: DTypeLike | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MemoryRawUpdate:
    """Only causal owned events, sealed frames and currently available metadata."""

    index: int
    row_json: bytes
    events: Mapping[str, np.ndarray]
    rgb: FrozenRGBHistory
    boxes: tuple[tuple[float, float, float, float], ...]
    calibration: Mapping[str, np.ndarray]
    byte_count: int
    acquisition_thread: int
    source_diagnostics: Mapping[str, Any]


class PreloadedRawSource:
    """Private bounded replay archive with a single acquiring thread and causal release.

    Call preload() and initial_history() before starting the paced clock. Invoke
    warmup/acquire_raw/finalize on one producer thread. Warmup may seal only frames
    available by first_query minus 20 ms. Every later frame is copied and sealed
    when its original timestamp first becomes available to acquire_raw.

    The declared source memory envelope includes RAW archive, a 16-frame token
    ring, conservative queue/consumer/producer payloads, initial history and a
    frame-copy scratch. HDF5 internals, Python object overhead, model/GPU storage
    and consumer result receipts are excluded and explicitly reported separately.
    """

    FRAME_CAPACITY = 16

    def __init__(
        self,
        schedule: Sequence[Mapping[str, Any]],
        max_payload_bytes: int = 128 * _MIB,
        *,
        max_total_bytes: int = 2 * 1024**3,
        max_frame_bytes: int = 128 * _MIB,
        max_initial_bytes: int = 512 * _MIB,
        queue_capacity: int = 2,
    ) -> None:
        self.max_payload_bytes = _positive_integer(max_payload_bytes, "max_payload_bytes")
        self.max_total_bytes = _positive_integer(max_total_bytes, "max_total_bytes")
        self.max_frame_bytes = _positive_integer(max_frame_bytes, "max_frame_bytes")
        self.max_initial_bytes = _positive_integer(max_initial_bytes, "max_initial_bytes")
        self.queue_capacity = _positive_integer(queue_capacity, "queue_capacity")
        if not schedule or "reason" in schedule[0]:
            raise ValueError("RAW memory replay needs a valid first scheduled slot")
        # Metadata only: no RAW data, tensors, future annotations or tokens here.
        self.schedule = json.loads(json.dumps(list(schedule), allow_nan=False))
        first = self.schedule[0]
        self._path = Path(first["raw_path"]).resolve()
        self._stat = dict(first["raw_stat"])
        self._first_query = int(first["anchor_us"])
        self._last_query = int(self.schedule[-1]["anchor_us"])
        self._start_us = self._first_query - STEP_US - 600_000
        if self._start_us < 0:
            raise ValueError("RAW memory replay requires the preceding 600 ms history")
        for index, row in enumerate(self.schedule):
            if int(row["anchor_us"]) != self._first_query + index * STEP_US:
                raise ValueError("RAW memory replay preserves contiguous original 20 ms slots")
            if "reason" not in row and (
                Path(row["raw_path"]).resolve() != self._path
                or row["raw_stat"] != self._stat
                or row["calibration_sha256"] != first["calibration_sha256"]
            ):
                raise ValueError("RAW memory replay cannot mix physical source revisions")
        self._origin = RGBSourceIdentity(
            f"{self._path}::blackflys/left/data",
            _canonical_sha256(
                {
                    "path": str(self._path),
                    "raw_stat": self._stat,
                    "calibration_sha256": first["calibration_sha256"],
                    "dataset": "blackflys/left/data",
                }
            ),
        )
        self._events: dict[str, np.ndarray] = {}
        self._frames: np.ndarray | None = None
        self._clock: np.ndarray | None = None
        self._frame_start = 0
        self._calibration: dict[str, np.ndarray] = {}
        self._annotation_bytes: dict[Path, bytes] = {}
        self._boxes: OrderedDict[str, tuple[float, float, float, float]] = OrderedDict()
        self._tokens: dict[int, FrozenRGBFrame] = {}
        self._released_through = -1
        self._last_release_query: int | None = None
        self._next_index = 0
        self.owner_thread: int | None = None
        self.closed = False
        self.preloaded = False
        self._preload_attempted = False
        self.preload_receipt: dict[str, Any] = {}
        self.stats: dict[str, Any] = {
            "scope": SCOPE,
            "acquired_updates": 0,
            "unique_frames_frozen": 0,
            "frame_freeze_bytes": 0,
            "frame_freeze_seconds": 0.0,
            "event_packet_copy_bytes": 0,
            "payload_frame_copy_bytes": 0,
            "peak_frame_token_count": 0,
            "peak_frame_token_bytes": 0,
            "bbox_validations": 0,
            "bbox_cache_hits": 0,
            "physical_sensor_claim": False,
            "targets_read": False,
        }

    def _check_stat(self) -> None:
        value = self._path.stat()
        if (value.st_size, value.st_mtime_ns) != (self._stat["size_bytes"], self._stat["mtime_ns"]):
            raise ValueError("RAW source changed since the replay manifest")

    def preload(self) -> dict[str, Any]:
        """Preflight the complete source envelope, then read RAW without future processing."""
        if self._preload_attempted or self.closed or self.owner_thread is not None:
            raise RuntimeError("RAW memory can be preloaded only once before acquisition")
        self._preload_attempted = True
        started = time.perf_counter()
        self._check_stat()
        annotation_paths = {
            Path(item["path"]).resolve()
            for row in self.schedule
            for item in row.get("bbox_sources", ())
        }
        annotation_sizes = {path: path.stat().st_size for path in annotation_paths}
        if any(size > _MIB for size in annotation_sizes.values()):
            raise ValueError("RAW_ANNOTATION_BYTE_LIMIT")
        metadata_bytes = len(json.dumps(self.schedule, allow_nan=False).encode())
        # Keep clock metadata itself bounded before reading it to plan RAW storage.
        with EvTTCEventReader(self._path) as reader:
            handle = reader._require_handle()
            names = {key: getattr(reader.layout, key) for key in EVENT_DTYPES}
            datasets: dict[str, h5py.Dataset] = {}
            for key, name in names.items():
                value = handle[name]
                if not isinstance(value, h5py.Dataset) or value.ndim != 1:
                    raise ValueError("RAW event fields must be one-dimensional datasets")
                datasets[key] = value
            stamps = datasets["t"]
            if (
                not len(stamps)
                or int(stamps[0]) > self._start_us
                or int(stamps[-1]) < self._last_query
            ):
                raise ValueError("RAW memory replay interval exceeds physical event support")
            ms_map = handle[reader.layout.ms_map_idx]
            if not isinstance(ms_map, h5py.Dataset) or not len(ms_map):
                raise ValueError("RAW memory replay requires the original millisecond index")
            start_ms = min(self._start_us // 1000, len(ms_map) - 1)
            end_ms = min(math.ceil(self._last_query / 1000) + 1, len(ms_map) - 1)
            event_start = int(ms_map[start_ms])
            event_end = int(ms_map[end_ms]) if end_ms < len(ms_map) - 1 else len(stamps)
            if not 0 <= event_start <= event_end <= len(stamps) or any(
                len(value) != len(stamps) for value in datasets.values()
            ):
                raise ValueError("RAW event source lengths/index are inconsistent")
            event_bytes = (event_end - event_start) * sum(
                value.dtype.itemsize for value in datasets.values()
            )
            data, clock = handle.get("blackflys/left/data"), handle.get("blackflys/left/ts")
            rgb_bytes = frame_bytes = clock_bytes = 0
            frame_end = 0
            if isinstance(data, h5py.Dataset) and isinstance(clock, h5py.Dataset):
                if clock.ndim != 1 or len(clock) * clock.dtype.itemsize > min(
                    _MIB, self.max_total_bytes
                ):
                    raise ValueError("RAW_CLOCK_BYTE_LIMIT")
                if (
                    data.dtype != np.uint8
                    or data.ndim != 4
                    or data.shape[-1] != 3
                    or len(data) != len(clock)
                ):
                    raise ValueError("RAW RGB must contain aligned uint8 HWC frames")
                self._clock = np.asarray(clock)
                if len(self._clock) > 1 and np.any(self._clock[1:] <= self._clock[:-1]):
                    raise ValueError("RAW RGB timestamps must be strictly chronological")
                clock_bytes = self._clock.nbytes
                self._frame_start = int(
                    np.searchsorted(self._clock, self._first_query - STEP_US - 650_000, side="left")
                )
                frame_end = int(np.searchsorted(self._clock, self._last_query, side="right"))
                frame_bytes = int(np.prod(data.shape[1:]))
                rgb_bytes = (frame_end - self._frame_start) * frame_bytes
            calibration = _calibration(handle)
            calibration_bytes = sum(value.nbytes for value in calibration.values())
            if calibration_bytes > _MIB:
                raise ValueError("RAW_CALIBRATION_BYTE_LIMIT")
            if _calibration_sha256(calibration) != self.schedule[0]["calibration_sha256"]:
                raise ValueError("RAW calibration differs from replay selection")
            archive_bytes = (
                event_bytes
                + rgb_bytes
                + clock_bytes
                + calibration_bytes
                + sum(annotation_sizes.values())
                + metadata_bytes
            )
            ring_bound = min(self.max_frame_bytes, self.FRAME_CAPACITY * frame_bytes)
            payload_bound = (self.queue_capacity + 3) * self.max_payload_bytes
            scratch_bound = max(
                self.max_initial_bytes,
                self.max_payload_bytes,
                frame_bytes,
                event_end - event_start,  # Boolean scratch for timestamp monotonicity validation.
            )
            total_bound = archive_bytes + ring_bound + payload_bound + scratch_bound
            if total_bound > self.max_total_bytes:
                raise ValueError(
                    f"RAW_PRELOAD_TOTAL_BYTE_LIMIT: {total_bound} > {self.max_total_bytes}"
                )
            # Native dtypes are preserved. Conversion to the service ABI occurs on acquisition.
            self._events = {
                key: np.asarray(value[event_start:event_end]) for key, value in datasets.items()
            }
            if len(self._events["t"]) > 1 and np.any(
                self._events["t"][1:] < self._events["t"][:-1]
            ):
                raise ValueError("RAW event timestamps must be monotonic")
            if self._clock is not None:
                assert isinstance(data, h5py.Dataset)
                self._frames = np.asarray(data[self._frame_start : frame_end])
                self._clock.setflags(write=False)
            self._calibration = calibration
            for value in (*self._events.values(), *calibration.values()):
                value.setflags(write=False)
            if self._frames is not None:
                self._frames.setflags(write=False)
            self._annotation_bytes = {path: path.read_bytes() for path in annotation_paths}
            if any(
                len(value) != annotation_sizes[path]
                for path, value in self._annotation_bytes.items()
            ):
                raise ValueError("Annotation source size changed during RAW preload")
        self._check_stat()
        self.preloaded = True
        self.preload_receipt = {
            "scope": SCOPE,
            "preload_seconds": time.perf_counter() - started,
            "raw_event_bytes": event_bytes,
            "raw_rgb_bytes": rgb_bytes,
            "archive_bytes": archive_bytes,
            "token_ring_upper_bound_bytes": ring_bound,
            "live_payload_upper_bound_bytes": payload_bound,
            "scratch_upper_bound_bytes": scratch_bound,
            "source_total_upper_bound_bytes": total_bound,
            "max_total_bytes": self.max_total_bytes,
            "memory_exclusions": (
                "Python object overhead, HDF5 internals, model/GPU storage and result receipts"
            ),
            "event_archive_rows": event_end - event_start,
            "rgb_archive_frames": frame_end - self._frame_start,
            "future_raw_bytes_preloaded_private_to_feeder": True,
            "future_tokens_or_hashes_prepared": False,
            "representations_precomputed": False,
            "physical_sensor_claim": False,
        }
        return dict(self.preload_receipt)

    def _require_owner(self) -> None:
        if not self.preloaded or self.closed or threading.get_ident() != self.owner_thread:
            raise RuntimeError("RAW memory acquisition requires its live producer owner thread")

    def _event_packet(self, start_us: int, end_us: int, max_bytes: int) -> Mapping[str, np.ndarray]:
        if not self.preloaded or self.closed:
            raise RuntimeError("RAW memory must be preloaded before copying event packets")
        if not self._start_us <= start_us < end_us <= self._last_query:
            raise ValueError("Event packet is outside the declared RAW replay interval")
        begin, end = np.searchsorted(self._events["t"], (start_us, end_us), side="left")
        if (int(end) - int(begin)) * 17 > max_bytes:
            raise ValueError("RAW_EVENT_BYTE_LIMIT: packet cannot be truncated")
        return MappingProxyType(
            {
                key: _readonly_copy(self._events[key][begin:end], dtype)
                for key, dtype in EVENT_DTYPES.items()
            }
        )

    def initial_history(self, max_bytes: int) -> Mapping[str, np.ndarray]:
        """Copy only the preceding 600 ms, with no future frame or representation work."""
        if self.owner_thread is not None:
            raise RuntimeError("Initial history belongs before the paced producer starts")
        return self._event_packet(
            self._start_us, self._first_query - STEP_US, min(max_bytes, self.max_initial_bytes)
        )

    def _release_frames(self, query_us: int) -> None:
        self._require_owner()
        if self._last_release_query is not None and query_us < self._last_release_query:
            raise ValueError("RAW memory release clock cannot roll back")
        if self._clock is None:
            self._last_release_query = query_us
            return
        assert self._frames is not None
        first = int(np.searchsorted(self._clock, query_us - 650_000, side="left"))
        end = int(np.searchsorted(self._clock, query_us, side="right"))
        frame_bytes = int(np.prod(self._frames.shape[1:]))
        if end - first > self.FRAME_CAPACITY or (end - first) * frame_bytes > self.max_frame_bytes:
            raise ValueError("RAW_FRAME_CAPACITY_LIMIT: complete causal history must fit")
        for index in list(self._tokens):
            if index < first:
                del self._tokens[index]
        for index in range(first, end):
            if index in self._tokens:
                continue
            if index <= self._released_through:
                raise ValueError("RAW memory cannot rerelease or recycle an evicted frame ID")
            token = freeze_rgb_frame(
                self._frames[index - self._frame_start],
                index,
                int(self._clock[index]),
                self._origin,
                max_bytes=self.max_frame_bytes,
            )
            self._tokens[index] = token
            self._released_through = index
            self.stats["unique_frames_frozen"] += 1
            self.stats["frame_freeze_bytes"] += token.hashed_bytes
            self.stats["frame_freeze_seconds"] += token.freeze_seconds
        self._last_release_query = query_us
        self.stats["peak_frame_token_count"] = max(
            self.stats["peak_frame_token_count"], len(self._tokens)
        )
        self.stats["peak_frame_token_bytes"] = max(
            self.stats["peak_frame_token_bytes"], len(self._tokens) * frame_bytes
        )

    def warmup(self) -> dict[str, Any]:
        """Seal historical frames only; future archive bytes stay private and unprocessed."""
        if not self.preloaded or self.closed or self.owner_thread is not None:
            raise RuntimeError("Preloaded RAW source can warm only once before acquisition")
        self.owner_thread = threading.get_ident()
        cutoff = self._first_query - STEP_US
        self._release_frames(cutoff)
        return {
            **self.stats,
            "warmup_cutoff_us": cutoff,
            "warmup_frames": len(self._tokens),
            "preload": dict(self.preload_receipt),
        }

    def _box(self, source: Mapping[str, Any]) -> tuple[float, float, float, float]:
        key = _canonical_sha256(dict(source))
        if key in self._boxes:
            self._boxes.move_to_end(key)
            self.stats["bbox_cache_hits"] += 1
            return self._boxes[key]
        path = Path(source["path"]).resolve()
        raw = self._annotation_bytes[path]
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("Frozen RGB bbox source SHA-256 mismatch")
        objects = json.loads(raw).get("objects", [])
        matches = [
            item
            for item in objects
            if isinstance(item, dict)
            and str(item.get("category")) == str(source["category"])
            and item.get("group") == source.get("group")
        ]
        if len(matches) != 1:
            raise ValueError("Frozen RGB bbox identity is no longer unique")
        box = _single_object_bbox(matches[0])
        if box is None:
            raise ValueError("Frozen RGB bbox became invalid")
        self._boxes[key] = box
        self.stats["bbox_validations"] += 1
        while len(self._boxes) > 128:
            self._boxes.popitem(last=False)
        return box

    def acquire_raw(self, index: int) -> MemoryRawUpdate:
        """Release only this original clock slot, including every new-copy/seal cost."""
        self._require_owner()
        if index != self._next_index or not 0 <= index < len(self.schedule):
            raise ValueError("RAW memory acquisition indices must be contiguous")
        self._next_index += 1
        row = self.schedule[index]
        if "reason" in row:
            raise ValueError(f"MANIFEST_EXCLUSION: {row['reason']}")
        query = int(row["anchor_us"])
        sources = row["bbox_sources"]
        if len(sources) != 3 or any(
            not 0 <= int(item["timestamp_us"]) <= int(item["available_us"]) <= query
            for item in sources
        ):
            raise ValueError("Future or invalid bbox metadata is unavailable at this query")
        row_json = json.dumps(dict(row), sort_keys=True, allow_nan=False).encode()
        if len(row_json) > min(_MIB, self.max_payload_bytes):
            raise ValueError("RAW_METADATA_BYTE_LIMIT")
        boxes = tuple(self._box(item) for item in sources)
        calibration = {key: _readonly_copy(value) for key, value in self._calibration.items()}
        used = len(row_json) + sum(value.nbytes for value in calibration.values())
        events = self._event_packet(query - STEP_US, query, self.max_payload_bytes - used)
        event_bytes = sum(value.nbytes for value in events.values())
        used += event_bytes
        before = self.stats["unique_frames_frozen"]
        before_seconds = self.stats["frame_freeze_seconds"]
        self._release_frames(query)
        groups = (
            tuple(
                tuple(int(value) for value in group)
                for group in sensor_triplets(self._clock, query)
            )
            if self._clock is not None
            else ()
        )
        ids = {item for group in groups for item in group}
        used += sum(self._tokens[frame_id].hashed_bytes for frame_id in ids)
        if used > self.max_payload_bytes:
            raise ValueError("RAW_PAYLOAD_BYTE_LIMIT: RGB frames cannot be truncated")
        reason = (
            ""
            if groups
            else "RGB_DATASET_MISSING"
            if self._clock is None
            else "INSUFFICIENT_CAUSAL_RGB_HISTORY"
        )
        rgb = FrozenRGBHistory(
            query, {frame_id: self._tokens[frame_id] for frame_id in ids}, groups, reason
        )
        self.stats["acquired_updates"] += 1
        self.stats["event_packet_copy_bytes"] += event_bytes
        return MemoryRawUpdate(
            index,
            row_json,
            events,
            rgb,
            boxes,
            MappingProxyType(calibration),
            used,
            threading.get_ident(),
            MappingProxyType(
                {
                    "source_mode": "memory",
                    "scope": SCOPE,
                    "new_frames_frozen": self.stats["unique_frames_frozen"] - before,
                    "new_frame_freeze_seconds": self.stats["frame_freeze_seconds"] - before_seconds,
                    "event_packet_copy_bytes": event_bytes,
                    "payload_frame_copy_bytes": 0,
                    "frame_token_count": len(self._tokens),
                    "disk_io_inside_acquisition": False,
                }
            ),
        )

    def finalize(self) -> dict[str, Any]:
        """Release private archive/ring on the owner; delivered token payloads remain valid."""
        if self.owner_thread is not None and threading.get_ident() != self.owner_thread:
            raise RuntimeError("RAW memory finalization requires the producer owner thread")
        self._events.clear()
        self._frames = self._clock = None
        self._tokens.clear()
        self._boxes.clear()
        self._annotation_bytes.clear()
        self._calibration.clear()
        self.closed = True
        self.stats["closed_on_owner_thread"] = threading.get_ident() == self.owner_thread
        return dict(self.stats)
