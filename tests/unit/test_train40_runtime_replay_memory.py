"""Private RAW preload, causal release, memory envelopes and owner lifecycle."""

from __future__ import annotations

import hashlib
import json
import threading
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from operational.train40_runtime import replay_memory as module
from operational.train40_runtime.paced import run_paced_replay


@pytest.fixture
def sensor(tmp_path, monkeypatch):
    path = tmp_path / "sensor.h5"
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps({"objects": [{"category": "car", "group": 1, "bbox": [0, 0, 3, 3]}]})
    )
    calibration = {"test": np.eye(2)}
    event_t = np.arange(0, 2_000_001, 1000, dtype=np.int64)
    clock = np.arange(300_000, 1_300_001, 50_000, dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle["events/t"] = event_t
        handle["events/x"] = np.arange(len(event_t), dtype=np.uint16) % 4
        handle["events/y"] = np.arange(len(event_t), dtype=np.uint16) % 3
        handle["events/p"] = np.arange(len(event_t), dtype=np.uint8) % 2
        handle["events/ms"] = np.searchsorted(event_t, np.arange(2001) * 1000)
        handle["blackflys/left/data"] = np.stack(
            [np.full((4, 4, 3), index, np.uint8) for index in range(len(clock))]
        )
        handle["blackflys/left/ts"] = clock
    stat = path.stat()
    rows = [
        {
            "query_id": f"query-{index}",
            "anchor_us": 1_000_000 + index * 20_000,
            "raw_path": str(path),
            "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "calibration_sha256": module._calibration_sha256(calibration),
            "bbox_sources": [
                {
                    "path": str(annotation),
                    "sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
                    "category": "car",
                    "group": 1,
                    "timestamp_us": stamp,
                    "available_us": stamp,
                }
                for stamp in (800_000, 900_000, 1_000_000)
            ],
        }
        for index in range(10)
    ]
    reads = []

    class Reader(module.EvTTCEventReader):
        def __init__(self, path):
            self.path = path
            self.chunk_events_max = 250_000
            self.layout = SimpleNamespace(
                x="events/x", y="events/y", t="events/t", p="events/p", ms_map_idx="events/ms"
            )

        def __enter__(self):
            reads.append(threading.get_ident())
            self.handle = h5py.File(self.path, "r")
            return self

        def __exit__(self, *args):
            self.handle.close()

        def _require_handle(self):
            return self.handle

    monkeypatch.setattr(module, "EvTTCEventReader", Reader)
    monkeypatch.setattr(
        module,
        "_calibration",
        lambda handle: {key: value.copy() for key, value in calibration.items()},
    )
    return rows, reads


def source(rows, **kwargs):
    return module.PreloadedRawSource(
        rows,
        max_payload_bytes=10_000,
        max_initial_bytes=20_000,
        max_total_bytes=200_000,
        max_frame_bytes=16 * 48,
        queue_capacity=2,
        **kwargs,
    )


def test_preload_keeps_native_bytes_private_and_defers_copy_hash_until_arrival(sensor, monkeypatch):
    rows, reads = sensor
    instance = source(rows)
    sealed = []
    original = module.freeze_rgb_frame

    def record(frame, index, timestamp, origin, **kwargs):
        sealed.append((index, timestamp, threading.get_ident()))
        return original(frame, index, timestamp, origin, **kwargs)

    monkeypatch.setattr(module, "freeze_rgb_frame", record)
    receipt = instance.preload()
    assert not sealed and not instance._tokens and instance.stats["bbox_validations"] == 0
    assert receipt["scope"] == module.SCOPE and not receipt["physical_sensor_claim"]
    assert receipt["future_raw_bytes_preloaded_private_to_feeder"]
    assert (
        not receipt["future_tokens_or_hashes_prepared"]
        and not receipt["representations_precomputed"]
    )
    assert instance._events["x"].dtype == np.uint16
    assert receipt["source_total_upper_bound_bytes"] <= receipt["max_total_bytes"]
    assert receipt["source_total_upper_bound_bytes"] == sum(
        receipt[key]
        for key in (
            "archive_bytes",
            "token_ring_upper_bound_bytes",
            "live_payload_upper_bound_bytes",
            "scratch_upper_bound_bytes",
        )
    )
    initial = instance.initial_history(20_000)
    np.testing.assert_array_equal(initial["t"], np.arange(380_000, 980_000, 1000))
    assert not sealed
    warm = instance.warmup()
    assert warm["warmup_cutoff_us"] == 980_000 and warm["warmup_frames"] == 13
    assert max(stamp for _, stamp, _ in sealed) == 950_000
    packets = [instance.acquire_raw(index) for index in range(10)]
    assert len(reads) == 1
    assert len(sealed) == instance.stats["unique_frames_frozen"] == 17
    assert packets[0].source_diagnostics["new_frames_frozen"] == 1
    assert packets[1].source_diagnostics["new_frames_frozen"] == 0
    assert packets[3].source_diagnostics["new_frames_frozen"] == 1
    assert packets[0].rgb.tokens[2] is packets[1].rgb.tokens[2]
    assert instance.stats["peak_frame_token_count"] <= 16
    for index, packet in enumerate(packets):
        query = 1_000_000 + index * 20_000
        np.testing.assert_array_equal(packet.events["t"], np.arange(query - 20_000, query, 1000))
        assert packet.events["x"].dtype == np.int32 and not packet.events["x"].flags.writeable
        assert not np.shares_memory(packet.events["x"], instance._events["x"])
        assert all(token.timestamp_us <= query for token in packet.rgb.tokens.values())
        assert packet.boxes == ((0.0, 0.0, 3.0, 3.0),) * 3
        assert not hasattr(packet, "reader") and not hasattr(packet, "_frames")
        for frame_id, frame in packet.rgb.frames.items():
            np.testing.assert_array_equal(frame, np.full((4, 4, 3), frame_id, np.uint8))
            with pytest.raises(ValueError):
                frame.setflags(write=True)
    instance.finalize()
    assert instance.closed and not instance._tokens and not instance._events
    np.testing.assert_array_equal(packets[0].rgb.frames[2], np.full((4, 4, 3), 2, np.uint8))


def test_preload_budget_rejects_before_loading_event_or_frame_arrays(sensor):
    rows, _ = sensor
    instance = source(rows)
    instance.max_total_bytes = 5000
    try:
        with pytest.raises(ValueError, match="TOTAL_BYTE_LIMIT"):
            instance.preload()
        assert instance._events == {} and instance._frames is None and not instance._tokens
        assert not instance.preloaded
        with pytest.raises(RuntimeError, match="only once"):
            instance.preload()
    finally:
        instance.finalize()


@pytest.mark.parametrize("mutation", ["clock", "source", "revision", "calibration"])
def test_constructor_rejects_hidden_gaps_or_mixed_source_revisions(sensor, mutation):
    rows, _ = sensor
    if mutation == "clock":
        rows[1]["anchor_us"] += 1
    elif mutation == "source":
        rows[1]["raw_path"] += "other"
    elif mutation == "revision":
        rows[1]["raw_stat"]["mtime_ns"] += 1
    else:
        rows[1]["calibration_sha256"] = "different"
    with pytest.raises(ValueError):
        source(rows)


def test_release_rejects_future_metadata_and_does_not_seal_future_frames(sensor):
    rows, _ = sensor
    rows[0]["bbox_sources"][2]["available_us"] = 1_000_001
    instance = source(rows)
    instance.preload()
    instance.warmup()
    before = instance.stats["unique_frames_frozen"]
    try:
        with pytest.raises(ValueError, match="Future"):
            instance.acquire_raw(0)
        assert instance.stats["unique_frames_frozen"] == before
    finally:
        instance.finalize()


def test_excluded_slots_remain_source_errors_and_indices_never_loop(sensor):
    rows, _ = sensor
    rows[1] = {"anchor_us": rows[1]["anchor_us"], "reason": "missing", "query_id": "excluded"}
    instance = source(rows)
    instance.preload()
    instance.warmup()
    try:
        instance.acquire_raw(0)
        with pytest.raises(ValueError, match="MANIFEST_EXCLUSION"):
            instance.acquire_raw(1)
        with pytest.raises(ValueError, match="contiguous"):
            instance.acquire_raw(1)
        assert instance.acquire_raw(2).index == 2
    finally:
        instance.finalize()


def test_preloaded_source_owner_lifecycle_and_diskless_acquisition(sensor, monkeypatch):
    rows, reads = sensor
    instance = source(rows)
    instance.preload()
    instance.initial_history(20_000)
    monkeypatch.setattr(module, "EvTTCEventReader", lambda *args: pytest.fail("Disk reopened"))
    packets = []

    def consume(arrival):
        packets.append(arrival.payload)
        return {"roi": {"success": True}}

    result = run_paced_replay(
        instance.acquire_raw,
        consume,
        ["roi"],
        updates=10,
        rate_hz=200,
        queue_capacity=10,
        wait_strategy="sleep",
        source_warmup=instance.warmup,
        source_finalize=instance.finalize,
    )
    assert result["source_errors"] == 0 and instance.closed
    assert result["capacity_checks"]["source_lifecycle_succeeded"]
    assert len(packets) == 10 and len(reads) == 1
    assert instance.stats["closed_on_owner_thread"]
    assert all(packet.acquisition_thread != threading.get_ident() for packet in packets)
    with pytest.raises(RuntimeError):
        instance.acquire_raw(0)


def test_cancellation_releases_private_archive_but_payload_remains_valid(sensor):
    rows, _ = sensor
    instance = source(rows)
    instance.preload()
    packets = []

    class Cancelled(BaseException):
        pass

    def cancel(arrival):
        packets.append(arrival.payload)
        raise Cancelled()

    with pytest.raises(Cancelled):
        run_paced_replay(
            instance.acquire_raw,
            cancel,
            ["roi"],
            updates=10,
            source_warmup=instance.warmup,
            source_finalize=instance.finalize,
            wait_strategy="sleep",
        )
    assert instance.closed and not instance._events and not instance._tokens
    assert packets[0].rgb.frames


def test_memory_payloads_match_original_disk_control_on_the_same_hdf5(sensor, monkeypatch):
    from scripts import benchmark_train40_paced as runner

    rows, _ = sensor
    monkeypatch.setattr(runner, "EvTTCEventReader", module.EvTTCEventReader)
    monkeypatch.setattr(runner, "_calibration", module._calibration)
    instance = source(rows)
    instance.preload()
    disk = runner.DiskRawSource(rows, instance.max_payload_bytes)
    actual_initial, expected_initial = (
        instance.initial_history(20_000),
        disk.initial_history(20_000),
    )
    for key in actual_initial:
        np.testing.assert_array_equal(actual_initial[key], expected_initial[key])
    instance.warmup()
    try:
        for index in range(10):
            actual, expected = instance.acquire_raw(index), disk.acquire_raw(index)
            assert actual.row_json == expected.row_json and actual.boxes == expected.boxes
            assert actual.byte_count == expected.byte_count
            assert actual.rgb.triplets == expected.rgb.triplets
            assert dict(actual.rgb.timestamps_us) == dict(expected.rgb.timestamps_us)
            assert actual.rgb.unavailable_reason == expected.rgb.unavailable_reason
            for key in actual.events:
                np.testing.assert_array_equal(actual.events[key], expected.events[key])
                assert actual.events[key].dtype == expected.events[key].dtype
            for key in actual.rgb.frames:
                np.testing.assert_array_equal(actual.rgb.frames[key], expected.rgb.frames[key])
    finally:
        instance.finalize()
