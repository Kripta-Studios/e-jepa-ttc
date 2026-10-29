"""Exact TRAIN40 inputs with bounded, persistent caches of raw sensor history.

Only physical events, original RGB frames and verified bbox metadata are cached.
Each query's ROI is applied anew, because the frozen transfer adapter uses a
retrospective shared query ROI. Prepared tensors or predictions are never
memoized here. Reads of new sensor history remain inside ``prepare_event``/
``raw_rgb_observations``.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict, deque
from collections.abc import Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from e_jepa_ttc.data.event_v4_geometry import common_square_from_boxes
from operational.evttc_rgb_transfer.inputs import _rgb_box
from operational.evttc_transfer.inputs import EvTTCEventReader
from operational.rgb_port_revision.transfer_inputs import EVENT_LAGS_US, sensor_triplets

from .encoding import crop_uint8_roi_first, encode_mapped_lut, map_roi_lut

BBoxKey = tuple[str, str, int, int, int, int, int]


def _normalized_crop(frame: np.ndarray, square: tuple[float, float, float, float]) -> np.ndarray:
    """Perform the unchanged uint8 crop, then normalize each unique frame once."""
    return np.ascontiguousarray(crop_uint8_roi_first(frame, square).astype(np.float32) / 255)


class PersistentSensorInputs:
    """Reuse raw history without assuming that different query ROIs are equal.

    The cache owns one source at a time and evicts oldest raw event chunks and
    least-recently-used RGB frames by bytes. A cache miss falls back to the same
    bounded HDF5 reader; it never drops a requested event or truncates history.
    The optional bounded crop pool receives only already-read immutable frames.
    The instance is sequential and must not be shared between caller threads.
    """

    def __init__(
        self,
        *,
        max_raw_bytes: int = 256 * 1024**2,
        max_frame_bytes: int = 128 * 1024**2,
        retained_bytes_max: int = 1024 * 1024**2,
        rgb_workers: int = 1,
        max_bbox_entries: int = 128,
    ) -> None:
        if min(max_raw_bytes, max_frame_bytes) < 0 or retained_bytes_max <= 0:
            raise ValueError("Cache budgets must be nonnegative and retained budget positive")
        if (
            not isinstance(rgb_workers, int)
            or isinstance(rgb_workers, bool)
            or not 1 <= rgb_workers <= 8
        ):
            raise ValueError("RGB crop workers must be an integer from one to eight")
        if not isinstance(max_bbox_entries, int) or max_bbox_entries < 0:
            raise ValueError("BBox cache capacity must be a nonnegative integer")
        self.max_raw_bytes = int(max_raw_bytes)
        self.max_frame_bytes = int(max_frame_bytes)
        self.retained_bytes_max = int(retained_bytes_max)
        self.rgb_crop_workers = rgb_workers
        self.max_bbox_entries = max_bbox_entries
        self._crop_pool: ThreadPoolExecutor | None = None
        self._boxes: OrderedDict[BBoxKey, tuple[float, float, float, float]] = OrderedDict()
        self._reader: EvTTCEventReader | None = None
        self._path: Path | None = None
        self._source_identity: tuple[Path, int, int] | None = None
        self._raw: deque[dict[str, np.ndarray]] = deque()
        self._raw_bytes = 0
        self._history_start = 0
        self._history_end = 0
        self._frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self._frame_bytes = 0
        self._clock: np.ndarray | None = None
        self.last_profile: dict[str, float] = {}
        self.stats: dict[str, int | float] = {
            "source_opens": 0,
            "event_read_bytes": 0,
            "event_cache_hit_bytes": 0,
            "frame_read_bytes": 0,
            "frame_cache_hits": 0,
            "event_unique_windows_encoded": 0,
            "event_prepare_seconds": 0.0,
            "rgb_prepare_seconds": 0.0,
            "rgb_crop_tasks": 0,
            "bbox_cache_hits": 0,
            "bbox_validations": 0,
        }

    def __enter__(self) -> PersistentSensorInputs:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Join crop workers, close HDF5 handles and release cache-owned arrays."""
        if self._crop_pool is not None:
            self._crop_pool.shutdown(wait=True, cancel_futures=True)
            self._crop_pool = None
        self._boxes.clear()
        self._close_source()

    def _close_source(self) -> None:
        """Release source-local state while preserving the bounded worker pool."""
        if self._reader is not None:
            self._reader.close()
        self._reader = None
        self._path = None
        self._source_identity = None
        self._raw.clear()
        self._raw_bytes = 0
        self._history_start = self._history_end = 0
        self._frames.clear()
        self._frame_bytes = 0
        self._clock = None

    @property
    def cache_bytes(self) -> dict[str, int]:
        """Current retained sensor bytes, excluding transient preprocessing arrays."""
        return {"events": self._raw_bytes, "frames": self._frame_bytes}

    def _source(self, row: Mapping[str, Any]) -> EvTTCEventReader:
        path = Path(row["raw_path"]).resolve(strict=True)
        stat = path.stat()
        expected = row["raw_stat"]
        if (
            int(expected["size_bytes"]) != stat.st_size
            or int(expected["mtime_ns"]) != stat.st_mtime_ns
        ):
            raise ValueError("Raw EvTTC source changed after query selection")
        identity = (path, stat.st_size, stat.st_mtime_ns)
        if identity != self._source_identity:
            self._close_source()
            self._reader = EvTTCEventReader(path).open()
            self._path = path
            self._source_identity = identity
            self.stats["source_opens"] += 1
        assert self._reader is not None
        return self._reader

    @staticmethod
    def _bytes(chunk: Mapping[str, np.ndarray]) -> int:
        return sum(value.nbytes for value in chunk.values())

    def _append_raw(self, chunk: dict[str, np.ndarray]) -> None:
        if not len(chunk["t"]):
            return
        self._raw.append(chunk)
        self._raw_bytes += self._bytes(chunk)
        while self._raw_bytes > self.max_raw_bytes:
            stale = self._raw.popleft()
            self._raw_bytes -= self._bytes(stale)
            # A chunk boundary may split equal timestamps. Claim completeness
            # only after every timestamp present in a discarded chunk.
            self._history_start = max(self._history_start, int(stale["t"][-1]) + 1)

    def iter_window_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int = 250_000
    ) -> Generator[dict[str, np.ndarray], None, None]:
        """Yield exact half-open events, incrementally reading only missing history."""
        complete = False
        try:
            yield from self._cached_chunks(start_us, end_us, chunk_events=chunk_events)
            complete = True
        finally:
            if not complete:
                # Capacity fallback may stop consuming a read midway. Its
                # partial suffix must never be advertised as complete history.
                self._raw.clear()
                self._raw_bytes = 0
                self._history_start = self._history_end = 0

    def _cached_chunks(
        self, start_us: int, end_us: int, *, chunk_events: int
    ) -> Iterator[dict[str, np.ndarray]]:
        if self._reader is None:
            raise RuntimeError("Select a validated source before requesting events")
        if start_us < 0 or end_us <= start_us or not 0 < chunk_events <= 250_000:
            raise ValueError("Invalid half-open event window or chunk capacity")
        overlaps = self._history_start <= start_us < self._history_end
        read_start = max(start_us, self._history_end) if overlaps else start_us
        if not overlaps:
            self._raw.clear()
            self._raw_bytes = 0
            self._history_start = self._history_end = start_us
        else:
            # Drop full stale arrays. Partial views are yielded only transiently;
            # cache byte accounting always includes their complete backing array.
            while self._raw and int(self._raw[0]["t"][-1]) < start_us:
                self._raw_bytes -= self._bytes(self._raw.popleft())
            self._history_start = start_us
            for chunk in self._raw:
                lo, hi = np.searchsorted(chunk["t"], (start_us, end_us), side="left")
                for cursor in range(int(lo), int(hi), chunk_events):
                    selected = {
                        key: value[cursor : min(cursor + chunk_events, hi)]
                        for key, value in chunk.items()
                    }
                    self.stats["event_cache_hit_bytes"] += self._bytes(selected)
                    yield selected
        if read_start < end_us:
            for chunk in self._reader.iter_window_chunks(
                read_start, end_us, chunk_events=chunk_events
            ):
                self.stats["event_read_bytes"] += self._bytes(chunk)
                self._append_raw(chunk)
                yield chunk
            self._history_end = end_us

    def _mapped_window(
        self, start: int, end: int, roi: tuple[float, float, float, float]
    ) -> dict[str, np.ndarray] | None:
        """Map each raw chunk once, retaining the original reduction order."""
        pieces: dict[str, list[np.ndarray]] = {key: [] for key in ("x", "y", "t", "p")}
        retained = 0
        last_timestamp: int | None = None
        chunks = self.iter_window_chunks(start, end)
        try:
            while True:
                begin = time.perf_counter()
                try:
                    chunk = next(chunks)
                except StopIteration:
                    self.last_profile["event_read_seconds"] += time.perf_counter() - begin
                    break
                self.last_profile["event_read_seconds"] += time.perf_counter() - begin
                begin = time.perf_counter()
                mapped = map_roi_lut(chunk, roi, 128, 0.0)
                if len(mapped["t"]):
                    if last_timestamp is not None and int(mapped["t"][0]) < last_timestamp:
                        raise ValueError("nonchronological raw union")
                    last_timestamp = int(mapped["t"][-1])
                self.last_profile["event_mapping_seconds"] += time.perf_counter() - begin
                retained += self._bytes(mapped)
                if retained > self.retained_bytes_max:
                    return None
                for key, values in mapped.items():
                    pieces[key].append(values)
        finally:
            chunks.close()
        begin = time.perf_counter()
        result = {
            key: np.concatenate(values) if values else np.empty(0, dtype=dtype)
            for (key, values), dtype in zip(
                pieces.items(), (np.int32, np.int32, np.int64, np.int8), strict=True
            )
        }
        self.last_profile["event_concat_seconds"] += time.perf_counter() - begin
        return result

    def prepare_event(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Return canonical H4 event tensors, encoding six unique windows once."""
        begin = time.perf_counter()
        self.last_profile = dict.fromkeys(
            (
                "event_source_seconds",
                "event_read_seconds",
                "event_mapping_seconds",
                "event_concat_seconds",
                "event_encoding_seconds",
                "event_assembly_seconds",
            ),
            0.0,
        )
        self._source(row)
        self.last_profile["event_source_seconds"] = time.perf_counter() - begin
        windows = np.asarray(row["windows_us"], dtype=np.int64)
        if windows.shape != (3, 2) or not np.array_equal(
            windows[:, 1] - windows[:, 0], [100_000] * 3
        ):
            raise ValueError("Three 100ms event windows are required")
        if not np.array_equal(np.diff(windows[:, 1]), [100_000] * 2):
            raise ValueError("Event endpoints must follow native 100ms cadence")
        if windows[-1, 1] != int(row["anchor_us"]):
            raise ValueError("Event reference must be the last context endpoint")
        intervals = windows[None] - EVENT_LAGS_US[:, None, None]
        values = row["square_xyxy"]
        if len(values) != 4:
            raise ValueError("Event ROI must contain four coordinates")
        roi = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
        mapped = self._mapped_window(int(intervals.min()), int(intervals.max()), roi)
        # Unlike the historical storage ABI, no twelve unused H16 slots are
        # allocated. Canonical normalization and float64 histogram order remain.
        own = np.empty((4, 3, 12, 128, 128), dtype=np.float32)
        encoded: dict[tuple[int, int], np.ndarray] = {}
        for slot, triplet in enumerate(intervals):
            for index, bounds in enumerate(triplet):
                start, end = int(bounds[0]), int(bounds[1])
                key = (start, end)
                if key not in encoded:
                    current = mapped
                    if current is None:
                        current = self._mapped_window(start, end, roi)
                        if current is None:
                            raise RuntimeError("RESOURCE_PAUSE:single window retained byte cap")
                    encoding_start = time.perf_counter()
                    encoded[key] = encode_mapped_lut(
                        current, start, end, 128, str(row["sequence_id"])
                    ).numpy()
                    self.last_profile["event_encoding_seconds"] += (
                        time.perf_counter() - encoding_start
                    )
                    self.stats["event_unique_windows_encoded"] += 1
                assembly_start = time.perf_counter()
                own[slot, index] = encoded[key]
                self.last_profile["event_assembly_seconds"] += time.perf_counter() - assembly_start
        self.stats["event_prepare_seconds"] += time.perf_counter() - begin
        return {"own_events": own}

    def _frame(self, data: h5py.Dataset, index: int) -> np.ndarray:
        if index in self._frames:
            self.stats["frame_cache_hits"] += 1
            self._frames.move_to_end(index)
            return self._frames[index]
        result = np.asarray(data[index], np.uint8)
        result.setflags(write=False)
        self.stats["frame_read_bytes"] += result.nbytes
        if result.nbytes <= self.max_frame_bytes:
            self._frames[index] = result
            self._frame_bytes += result.nbytes
            while self._frame_bytes > self.max_frame_bytes:
                _, stale = self._frames.popitem(last=False)
                self._frame_bytes -= stale.nbytes
        return result

    @staticmethod
    def _bbox_key(source: Mapping[str, Any]) -> BBoxKey:
        path = Path(str(source["path"])).resolve(strict=True)
        stat = path.stat()
        identity = json.dumps(dict(source), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return (
            identity,
            str(path),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def _box(self, source: Mapping[str, Any]) -> tuple[float, float, float, float]:
        """Reuse a verified bbox only while full manifest identity and stat match.

        Changed metadata or bytes force the frozen parser's SHA validation. This
        stat-guarded cache does not claim protection against an adversarial file
        mutation that also restores every recorded filesystem attribute.
        """
        # Alternate in-memory test adapters can supply a parser without a file.
        if self.max_bbox_entries == 0 or "path" not in source:
            self.stats["bbox_validations"] += 1
            return _rgb_box(source)
        key = self._bbox_key(source)
        if key in self._boxes:
            self._boxes.move_to_end(key)
            self.stats["bbox_cache_hits"] += 1
            return self._boxes[key]
        self.stats["bbox_validations"] += 1
        box = _rgb_box(source)
        if key != self._bbox_key(source):
            raise ValueError("Frozen RGB bbox source changed during validation")
        self._boxes[key] = box
        while len(self._boxes) > self.max_bbox_entries:
            self._boxes.popitem(last=False)
        return box

    def _crop_frames(
        self,
        frames: Mapping[int, np.ndarray],
        square: tuple[float, float, float, float],
    ) -> dict[int, np.ndarray]:
        """Crop a bounded unique-frame set; all HDF5 reads remain on the owner."""
        self.stats["rgb_crop_tasks"] += len(frames)
        if self.rgb_crop_workers == 1:
            return {index: _normalized_crop(frame, square) for index, frame in frames.items()}
        if self._crop_pool is None:
            self._crop_pool = ThreadPoolExecutor(
                max_workers=self.rgb_crop_workers, thread_name_prefix="train40-rgb-crop"
            )
        # sensor_triplets bounds the history to ten unique native frames, so the
        # queue cannot grow with the number of queries or length of a sequence.
        if len(frames) > 10:
            raise ValueError("RGB native history exceeds the ten-frame task bound")
        pending = [
            (index, self._crop_pool.submit(_normalized_crop, frame, square))
            for index, frame in frames.items()
        ]
        try:
            return {index: task.result() for index, task in pending}
        except BaseException:
            for _, task in pending:
                task.cancel()
            # Do not return while a worker still owns a frame from this query.
            wait([task for _, task in pending])
            raise

    def raw_rgb_observations(
        self, row: Mapping[str, Any]
    ) -> tuple[list[tuple[np.ndarray, dict[str, Any]]], str]:
        """Crop cached original RGB frames in the current query's exact ROI."""
        begin = time.perf_counter()
        reader = self._source(row)
        self.last_profile["rgb_source_seconds"] = time.perf_counter() - begin
        roi_start = time.perf_counter()
        query = int(row["anchor_us"])
        boxes = [self._box(source) for source in list(row["bbox_sources"])[-2:]]
        square = common_square_from_boxes(boxes, (0, 1), margin_fraction=0.25)
        handle = reader._require_handle()
        data, timestamps = handle.get("blackflys/left/data"), handle.get("blackflys/left/ts")
        if not isinstance(data, h5py.Dataset) or not isinstance(timestamps, h5py.Dataset):
            return [], "RGB_DATASET_MISSING"
        if self._clock is None:
            self._clock = np.asarray(timestamps, dtype=np.int64)
        clock = self._clock
        groups = sensor_triplets(clock, query)
        if not groups:
            return [], "INSUFFICIENT_CAUSAL_RGB_HISTORY"
        self.last_profile["rgb_roi_seconds"] = time.perf_counter() - roi_start
        self.last_profile["rgb_frame_read_seconds"] = 0.0
        self.last_profile["rgb_crop_seconds"] = 0.0
        raw_frames = {}
        for index in np.unique(np.concatenate(groups)):
            frame_start = time.perf_counter()
            raw_frames[int(index)] = self._frame(data, int(index))
            self.last_profile["rgb_frame_read_seconds"] += time.perf_counter() - frame_start
        crop_start = time.perf_counter()
        frames = self._crop_frames(raw_frames, square)
        self.last_profile["rgb_crop_seconds"] = time.perf_counter() - crop_start
        stack_start = time.perf_counter()
        result: list[tuple[np.ndarray, dict[str, Any]]] = []
        for ids in groups:
            sensor = clock[ids]
            rgb = np.ascontiguousarray(np.stack([frames[int(index)] for index in ids]))
            result.append(
                (
                    rgb,
                    {
                        "first_sensor_ts_us": int(sensor[0]),
                        "anchor_us": int(sensor[-1]),
                        "available_us": query,
                        "timestamps_us": sensor.tolist(),
                        "delta_t_s": (np.diff(sensor) / 1e6).tolist(),
                        "square_xyxy": list(square),
                    },
                )
            )
        self.last_profile["rgb_stack_seconds"] = time.perf_counter() - stack_start
        self.stats["rgb_prepare_seconds"] += time.perf_counter() - begin
        return result, ""

    def preload_raw_rgb(self, row: Mapping[str, Any]) -> None:
        """Load only causal original frames, without ROI work or normalization.

        This models raw sensor observations becoming available in memory for a
        last-packet replay. Disk-inclusive callers should use the normal method.
        """
        reader = self._source(row)
        handle = reader._require_handle()
        data, timestamps = handle.get("blackflys/left/data"), handle.get("blackflys/left/ts")
        if not isinstance(data, h5py.Dataset) or not isinstance(timestamps, h5py.Dataset):
            return
        if self._clock is None:
            self._clock = np.asarray(timestamps, dtype=np.int64)
        groups = sensor_triplets(self._clock, int(row["anchor_us"]))
        if groups:
            for index in np.unique(np.concatenate(groups)):
                self._frame(data, int(index))
