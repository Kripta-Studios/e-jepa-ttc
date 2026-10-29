"""Parity and cache invariants for the persistent frozen TRAIN40 input adapter."""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from e_jepa_ttc.efficient_context.mapped_union import encode_union
from operational.rgb_port_revision.transfer_inputs import raw_rgb_observations
from operational.train40_runtime import inputs


class FakeReader:
    """Physical ordered events whose I/O requests can be audited by the tests."""

    def __init__(self, path, raw):
        self.path, self.raw = path, raw
        self.requests = []
        self.handle = None

    def open(self):
        return self

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _require_handle(self):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        return self.handle

    def iter_window_chunks(self, start, end, *, chunk_events=250_000):
        self.requests.append((start, end))
        lo, hi = np.searchsorted(self.raw["t"], (start, end), side="left")
        step = min(chunk_events, 117)
        for cursor in range(int(lo), int(hi), step):
            yield {
                key: value[cursor : min(cursor + step, hi)].copy()
                for key, value in self.raw.items()
            }


@pytest.fixture
def sensor(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    path = tmp_path / "sensor.h5"
    with h5py.File(path, "w") as handle:
        handle["blackflys/left/ts"] = np.arange(10, dtype=np.int64) * 100_000
        handle["blackflys/left/data"] = rng.integers(0, 256, (10, 24, 24, 3), np.uint8)
    raw = {
        "x": rng.integers(0, 24, 2100, dtype=np.int32),
        "y": rng.integers(0, 24, 2100, dtype=np.int32),
        "t": np.sort(rng.integers(0, 1_000_000, 2100, dtype=np.int64)),
        "p": rng.integers(0, 2, 2100, dtype=np.int8),
    }
    readers = []

    def factory(source):
        reader = FakeReader(source, raw)
        readers.append(reader)
        return reader

    monkeypatch.setattr(inputs, "EvTTCEventReader", factory)
    stat = path.stat()
    row = {
        "raw_path": str(path),
        "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "sequence_id": "fixture",
        "anchor_us": 600_000,
        "windows_us": [[300_000, 400_000], [400_000, 500_000], [500_000, 600_000]],
        "square_xyxy": [0.3, 0.2, 23.2, 23.1],
        "bbox_sources": [{"box": [2, 2, 16, 16]}, {"box": [3, 3, 19, 19]}],
    }
    return row, raw, readers


def reference_events(row, raw):
    reader = FakeReader(Path(row["raw_path"]), raw)
    return encode_union(
        reader,
        np.asarray(row["windows_us"], np.int64),
        np.arange(15, -1, -1, dtype=np.int64) * 100_000,
        np.arange(16) >= 12,
        tuple(row["square_xyxy"]),
        sequence_id=row["sequence_id"],
        roi_size=128,
        event_pixel_diff=0.0,
    )[-4:].numpy()


def test_event_windows_are_bitwise_equal_and_incremental(sensor):
    row, raw, readers = sensor
    with inputs.PersistentSensorInputs() as adapter:
        actual = adapter.prepare_event(row)["own_events"]
        np.testing.assert_array_equal(actual, reference_events(row, raw))
        assert adapter.stats["event_unique_windows_encoded"] == 6
        assert readers[0].requests == [(0, 600_000)]
        row = dict(
            row, anchor_us=620_000, windows_us=(np.asarray(row["windows_us"]) + 20_000).tolist()
        )
        actual = adapter.prepare_event(row)["own_events"]
        np.testing.assert_array_equal(actual, reference_events(row, raw))
        assert readers[0].requests[-1] == (600_000, 620_000)
        assert adapter.stats["event_cache_hit_bytes"] > 0
        assert adapter.cache_bytes["events"] <= adapter.max_raw_bytes


def test_event_roi_is_reapplied_to_raw_cache(sensor):
    row, raw, readers = sensor
    with inputs.PersistentSensorInputs() as adapter:
        first = adapter.prepare_event(row)["own_events"]
        different = dict(row, square_xyxy=[4.1, 3.2, 17.9, 17.0])
        second = adapter.prepare_event(different)["own_events"]
        np.testing.assert_array_equal(second, reference_events(different, raw))
        assert not np.array_equal(first, second)
        assert len(readers[0].requests) == 1


def test_small_cache_preserves_events_and_rewind(sensor):
    row, raw, _ = sensor
    with inputs.PersistentSensorInputs(max_raw_bytes=1) as adapter:
        for _ in range(2):
            actual = adapter.prepare_event(row)["own_events"]
            np.testing.assert_array_equal(actual, reference_events(row, raw))
            assert adapter.cache_bytes["events"] <= 1


def test_partial_read_never_becomes_complete_history(sensor):
    row, raw, readers = sensor
    with inputs.PersistentSensorInputs() as adapter:
        adapter._source(row)
        stream = adapter.iter_window_chunks(0, 600_000)
        next(stream)
        stream.close()
        assert adapter.cache_bytes["events"] == 0
        actual = adapter.prepare_event(row)["own_events"]
        np.testing.assert_array_equal(actual, reference_events(row, raw))
        assert readers[0].requests == [(0, 600_000), (0, 600_000)]


def test_union_capacity_fallback_preserves_every_window(sensor):
    row, raw, _ = sensor
    with inputs.PersistentSensorInputs(retained_bytes_max=8_000) as adapter:
        actual = adapter.prepare_event(row)["own_events"]
        np.testing.assert_array_equal(actual, reference_events(row, raw))
        assert adapter.stats["event_unique_windows_encoded"] == 6


def test_changed_source_fails_before_using_cache(sensor):
    row, _, _ = sensor
    with inputs.PersistentSensorInputs() as adapter:
        adapter.prepare_event(row)
        bad = dict(row, raw_stat={"size_bytes": 0, "mtime_ns": 0})
        with pytest.raises(ValueError, match="source changed"):
            adapter.prepare_event(bad)


def test_new_source_identity_invalidates_same_path_cache(sensor):
    row, _, readers = sensor
    with inputs.PersistentSensorInputs() as adapter:
        adapter.prepare_event(row)
        path = Path(row["raw_path"])
        with path.open("ab") as stream:
            stream.write(b"changed")
        stat = path.stat()
        new = dict(row, raw_stat={"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        adapter.prepare_event(new)
        assert len(readers) == 2
        assert readers[1].requests == [(0, 600_000)]


@pytest.mark.parametrize("workers", [1, 4])
def test_rgb_frames_cached_before_roi_not_after(sensor, monkeypatch, workers):
    row, _, _ = sensor
    monkeypatch.setattr(inputs, "_rgb_box", lambda source: source["box"])
    import operational.rgb_port_revision.transfer_inputs as reference

    monkeypatch.setattr(reference, "_rgb_box", lambda source: source["box"])
    with inputs.PersistentSensorInputs(
        max_frame_bytes=10 * 24 * 24 * 3, rgb_workers=workers
    ) as adapter:
        actual, reason = adapter.raw_rgb_observations(row)
        expected, expected_reason = raw_rgb_observations(row)
        assert reason == expected_reason == ""
        for (pixels, metadata), (reference_pixels, reference_metadata) in zip(
            actual, expected, strict=True
        ):
            np.testing.assert_array_equal(pixels, reference_pixels)
            assert metadata == reference_metadata
        read_bytes = adapter.stats["frame_read_bytes"]
        different = dict(row, bbox_sources=[{"box": [5, 5, 10, 10]}, {"box": [6, 6, 13, 13]}])
        changed, _ = adapter.raw_rgb_observations(different)
        expected_changed, _ = raw_rgb_observations(different)
        for (pixels, metadata), (reference_pixels, reference_metadata) in zip(
            changed, expected_changed, strict=True
        ):
            np.testing.assert_array_equal(pixels, reference_pixels)
            assert metadata == reference_metadata
        assert not np.array_equal(actual[0][0], changed[0][0])
        assert adapter.stats["frame_read_bytes"] == read_bytes
        assert adapter.stats["frame_cache_hits"] == 7
        assert adapter.stats["rgb_crop_tasks"] == 14
        assert adapter.cache_bytes["frames"] <= adapter.max_frame_bytes


def test_rgb_box_source_still_checks_frozen_hash(tmp_path):
    source = tmp_path / "box.json"
    source.write_text(json.dumps({"objects": []}), encoding="utf-8")
    wrong = {"path": str(source), "sha256": hashlib.sha256(b"different").hexdigest()}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inputs._rgb_box(wrong)


def bbox_source(tmp_path):
    path = tmp_path / "roi.json"
    path.write_text(
        json.dumps({"objects": [{"category": "car", "group": 1, "bbox": [2, 3, 20, 21]}]}),
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "category": "car",
        "group": 1,
        "object_index": 0,
        "frame_index": 7,
    }


def test_bbox_cache_binds_entire_source_identity_and_is_bounded(tmp_path):
    source = bbox_source(tmp_path)
    with inputs.PersistentSensorInputs(max_bbox_entries=2) as adapter:
        assert adapter._box(source) == (2.0, 3.0, 20.0, 21.0)
        assert adapter._box(dict(source)) == adapter._box(source)
        assert adapter.stats["bbox_validations"] == 1
        assert adapter.stats["bbox_cache_hits"] == 2
        adapter._box(dict(source, frame_index=8))
        adapter._box(dict(source, object_index=1))
        assert adapter.stats["bbox_validations"] == 3
        assert len(adapter._boxes) == 2
        with pytest.raises(ValueError, match="no longer unique"):
            adapter._box(dict(source, category="pedestrian"))


def test_bbox_cache_rehashes_changed_stat_and_rejects_changed_bytes(tmp_path):
    source = bbox_source(tmp_path)
    path = Path(source["path"])
    with inputs.PersistentSensorInputs() as adapter:
        box = adapter._box(source)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        assert adapter._box(source) == box
        assert adapter.stats["bbox_validations"] == 2
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            adapter._box(source)


def test_rgb_pool_keeps_hdf5_reads_on_owner_and_joins_workers(sensor, monkeypatch):
    row, _, _ = sensor
    monkeypatch.setattr(inputs, "_rgb_box", lambda source: source["box"])
    adapter = inputs.PersistentSensorInputs(rgb_workers=3)
    owner = threading.get_ident()
    original_frame = adapter._frame
    reads = []

    def frame(data, index):
        reads.append(threading.get_ident())
        return original_frame(data, index)

    monkeypatch.setattr(adapter, "_frame", frame)
    adapter.raw_rgb_observations(row)
    pool = adapter._crop_pool
    assert pool is not None
    assert set(reads) == {owner}
    assert 1 <= len(pool._threads) <= 3
    adapter.raw_rgb_observations(row)
    assert adapter._crop_pool is pool
    adapter.close()
    assert not any(thread.is_alive() for thread in pool._threads)
    assert adapter._crop_pool is None
    assert adapter.cache_bytes == {"events": 0, "frames": 0}
    adapter.close()


def test_rgb_pool_drains_failed_query_and_remains_usable(sensor, monkeypatch):
    row, _, _ = sensor
    monkeypatch.setattr(inputs, "_rgb_box", lambda source: source["box"])
    original_crop = inputs._normalized_crop
    lock = threading.Lock()
    running, calls = 0, 0

    def failing_crop(frame, square):
        nonlocal running, calls
        with lock:
            calls += 1
            first = calls == 1
            running += 1
        try:
            if first:
                raise RuntimeError("crop fixture failure")
            time.sleep(0.01)
            return original_crop(frame, square)
        finally:
            with lock:
                running -= 1

    with inputs.PersistentSensorInputs(rgb_workers=3) as adapter:
        monkeypatch.setattr(inputs, "_normalized_crop", failing_crop)
        with pytest.raises(RuntimeError, match="crop fixture failure"):
            adapter.raw_rgb_observations(row)
        assert running == 0
        monkeypatch.setattr(inputs, "_normalized_crop", original_crop)
        actual, reason = adapter.raw_rgb_observations(row)
        assert len(actual) == 5 and reason == ""


@pytest.mark.parametrize("workers", [0, -1, 9, 1.5, True])
def test_rgb_pool_capacity_is_explicitly_bounded(workers):
    with pytest.raises(ValueError, match="one to eight"):
        inputs.PersistentSensorInputs(rgb_workers=workers)
