"""CPU-only admission, ownership and failure checks for the paced service CLI."""

from __future__ import annotations

import copy
import json
import threading
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from operational.train40_runtime import inputs as input_module
from operational.train40_runtime.measurement import PREDICTION_COMPONENTS
from operational.train40_runtime.paced import Arrival, run_paced_replay
from scripts import benchmark_train40_paced as runner


def manifest(tmp_path, count=3, excluded=()):
    path = tmp_path / "sensor.h5"
    annotation = tmp_path / "annotation.json"
    annotation.write_text("{}")
    with h5py.File(path, "w") as handle:
        handle["blackflys/left/data"] = np.zeros((7, 4, 4, 3), dtype=np.uint8)
        handle["blackflys/left/ts"] = np.arange(400_000, 1_000_001, 100_000)
        handle["event_t"] = np.array([0, 2_000_000], dtype=np.int64)
    stat = path.stat()
    raw_stat = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    rows, failures = [], []
    for index in range(count):
        anchor = 1_000_000 + index * 20_000
        row = {
            "query_id": f"SENSOR_REPLAY:sequence:{anchor}",
            "sequence_id": "sequence",
            "physical_sequence_id": "sequence",
            "anchor_us": anchor,
            "raw_path": str(path),
            "raw_stat": raw_stat,
            "replay_step_us": 20_000,
            "calibration_sha256": "calibration",
            "windows_us": [
                [anchor - 300_000, anchor - 200_000],
                [anchor - 200_000, anchor - 100_000],
                [anchor - 100_000, anchor],
            ],
            "bbox_sources": [
                {"path": str(annotation), "timestamp_us": value, "available_us": value}
                for value in (800_000, 900_000, 1_000_000)
            ],
        }
        row["metadata_sha256"] = runner._canonical_sha256(row)
        if index in excluded:
            failures.append(
                {
                    "query_id": row["query_id"],
                    "physical_sequence_id": "sequence",
                    "anchor_us": anchor,
                    "reason": "MISSING_ANNOTATION",
                }
            )
        else:
            rows.append(row)
    return {
        "schema": runner.SCHEMA,
        "targets_read": False,
        "optimizer_updates": 0,
        "provenance": {"step_us": 20_000},
        "rows": rows,
        "excluded": failures,
        "inventory": [
            {
                "physical_sequence_id": "sequence",
                "raw_path": str(path),
                "raw_stat": raw_stat,
                "scheduled_updates": count,
                "requested_bounds_us": [1_000_000, 1_000_000 + (count - 1) * 20_000],
            }
        ],
    }


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    data = manifest(tmp_path)
    reads = []

    class Reader:
        intervals_seen = []
        closes = []

        def __init__(self, path):
            self.path = path
            self.layout = SimpleNamespace(t="event_t")

        def __enter__(self):
            return self.open()

        def open(self):
            reads.append(threading.get_ident())
            self.handle = h5py.File(self.path, "r")
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            self.closes.append(threading.get_ident())
            self.handle.close()

        def _require_handle(self):
            return self.handle

        def _bounds(self, start, end):
            return 0, 2

        def iter_window_chunks(self, start, end):
            self.intervals_seen.append((start, end))
            yield {
                "x": np.array([0, 1], dtype=np.int32),
                "y": np.array([0, 1], dtype=np.int32),
                "t": np.array([start, end - 1], dtype=np.int64),
                "p": np.array([0, 1], dtype=np.int8),
            }

    monkeypatch.setattr(runner, "EvTTCEventReader", Reader)
    monkeypatch.setattr(input_module, "EvTTCEventReader", Reader)
    monkeypatch.setattr(
        runner.PersistentSensorInputs, "_box", lambda self, item: runner._rgb_box(item)
    )
    monkeypatch.setattr(runner, "_rgb_box", lambda item: (0.0, 0.0, 3.0, 3.0))
    monkeypatch.setattr(runner, "_calibration", lambda handle: {"test": np.eye(2)})
    monkeypatch.setattr(runner, "_calibration_sha256", lambda value: "calibration")
    geometry_threads = []

    def projection(box, calibration):
        geometry_threads.append(threading.get_ident())
        return box

    monkeypatch.setattr(runner, "project_rgb_box_rotation_only", projection)
    schedule = runner.replay_schedule(data, "sequence")
    source = runner.DiskRawSource(schedule, 10_000)
    return source, reads, geometry_threads


class Service:
    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    def update(self, packet, start, end, regions, rgb):
        self.calls.append((start, end, [item.object_id for item in regions]))
        result = {
            "objects": [
                {
                    "object_id": item.object_id,
                    "predictions": {name: 1.0 for name in PREDICTION_COMPONENTS},
                    "rgb_available": True,
                    "rgb_unavailable_reason": "",
                    "history_lengths": {"event": 4, "rgb": 5},
                }
                for item in regions
            ]
        }
        if self.mutate:
            self.mutate(result)
        return result


def test_schedule_retains_excluded_clock_slots(tmp_path):
    value = manifest(tmp_path, excluded=(1,))
    schedule = runner.replay_schedule(value, "sequence")
    assert [row["anchor_us"] for row in schedule] == [1_000_000, 1_020_000, 1_040_000]
    assert schedule[1]["reason"] == "MISSING_ANNOTATION"
    assert len(runner.replay_schedule(value, "sequence", 2)) == 2


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "digest", "raw", "cadence"])
def test_invalid_or_compressed_schedule_rejected(tmp_path, mutation):
    value = manifest(tmp_path)
    if mutation == "missing":
        value["rows"].pop(1)
    elif mutation == "duplicate":
        value["rows"][1] = copy.deepcopy(value["rows"][0])
    else:
        row = value["rows"][1]
        row[{"digest": "query_id", "raw": "raw_path", "cadence": "replay_step_us"}[mutation]] = (
            "changed"
        )
        if mutation != "digest":
            row["metadata_sha256"] = runner._canonical_sha256(
                {key: item for key, item in row.items() if key != "metadata_sha256"}
            )
    with pytest.raises(ValueError):
        runner.replay_schedule(value, "sequence")


def test_raw_arrays_are_owned_readonly_and_do_not_expose_handles(source_fixture):
    source, _, geometry = source_fixture
    payload = source.acquire_raw(0)
    assert geometry == []
    arrays = [*payload.events.values(), *payload.rgb.frames.values(), *payload.calibration.values()]
    assert all(value.flags.owndata and not value.flags.writeable for value in arrays)
    assert payload.byte_count == len(payload.row_json) + sum(value.nbytes for value in arrays)
    with pytest.raises(ValueError):
        payload.events["x"][0] = 99
    with pytest.raises(TypeError):
        payload.rgb.frames[0] = np.zeros(1)
    assert not hasattr(payload, "reader")


def twenty_hz_schedule(source):
    """Two alternating native100ms cohorts, without reading a real dataset."""
    path = source.schedule[0]["raw_path"]
    clock = np.arange(300_000, 1_300_001, 50_000, dtype=np.int64)
    with h5py.File(path, "a") as handle:
        del handle["blackflys/left/data"]
        del handle["blackflys/left/ts"]
        handle["blackflys/left/data"] = np.stack(
            [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(len(clock))]
        )
        handle["blackflys/left/ts"] = clock
    stat = runner.Path(path).stat()
    rows = []
    for index in range(10):
        row = copy.deepcopy(source.schedule[0])
        row["anchor_us"] = 1_000_000 + index * 20_000
        row["query_id"] = f"query-{index}"
        row["raw_stat"] = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        rows.append(row)
    return rows


def test_persistent_raw_matches_disk_and_reads_each_causal_frame_once(source_fixture):
    disk, reads, geometry = source_fixture
    schedule = twenty_hz_schedule(disk)
    source = runner.PersistentRawSource(schedule, 10_000, max_frame_bytes=16 * 48)
    payloads = []

    def infer(arrival):
        payloads.append(arrival.payload)
        return {
            "a": {"success": True, "predictions": {name: 1.0 for name in PREDICTION_COMPONENTS}}
        }

    result = run_paced_replay(
        source.acquire_raw,
        infer,
        ["a"],
        updates=10,
        rate_hz=200,
        queue_capacity=10,
        wait_strategy="sleep",
        source_warmup=source.warmup,
        source_finalize=source.finalize,
    )
    assert result["grouped"]["passed"]
    assert result["capacity_checks"]["source_lifecycle_succeeded"]
    assert source.closed and source.stats["closed_on_owner_thread"]
    assert source.stats["input_stats"]["source_opens"] == 1
    assert len(reads) == 1 and reads[0] != threading.get_ident()
    assert source.stats["warmup_cutoff_us"] == 980_000
    assert source.stats["warmup_frames"] == 13  # <=950ms, never the 1000ms frame.
    assert source.stats["unique_frames_read"] == 17
    assert source.stats["peak_frame_cache_count"] == 14
    assert source.stats["peak_frame_cache_bytes"] <= 16 * 48
    assert runner.EvTTCEventReader.intervals_seen == [
        (980_000 + index * 20_000, 1_000_000 + index * 20_000) for index in range(10)
    ]
    assert geometry == []  # No ROI/projection work in acquisition or warmup.
    assert payloads[0].source_diagnostics["new_frames_read"] == 1
    assert payloads[1].source_diagnostics["new_frames_read"] == 0
    assert payloads[3].source_diagnostics["new_frames_read"] == 1
    control = runner.DiskRawSource(schedule, 10_000)
    for index, actual in enumerate(payloads):
        expected = control.acquire_raw(index)
        assert actual.row_json == expected.row_json
        assert actual.rgb.triplets == expected.rgb.triplets
        assert actual.boxes == expected.boxes
        assert actual.byte_count == expected.byte_count
        for key in actual.events:
            np.testing.assert_array_equal(actual.events[key], expected.events[key])
        for key in actual.rgb.frames:
            np.testing.assert_array_equal(actual.rgb.frames[key], expected.rgb.frames[key])
            assert actual.rgb.timestamps_us[key] <= actual.rgb.query_us
            assert (
                actual.rgb.frames[key].flags.owndata and not actual.rgb.frames[key].flags.writeable
            )
    common = next(iter(payloads[0].rgb.frames))
    assert not np.shares_memory(payloads[0].rgb.frames[common], payloads[1].rgb.frames[common])


def test_persistent_raw_overflow_fails_and_closes_without_truncation(source_fixture):
    disk, _, _ = source_fixture
    source = runner.PersistentRawSource(twenty_hz_schedule(disk), 10_000, max_frame_bytes=12 * 48)
    result = run_paced_replay(
        source.acquire_raw,
        lambda item: pytest.fail("Warmup should fail"),
        ["a"],
        updates=1,
        source_warmup=source.warmup,
        source_finalize=source.finalize,
    )
    assert source.closed
    assert result["source_lifecycle"]["errors"][0]["phase"] == "warmup"
    assert "BYTE_LIMIT" in result["source_lifecycle"]["errors"][0]["error"]
    assert not result["capacity_checks"]["source_lifecycle_succeeded"]
    assert not result["grouped"]["passed"]


def test_persistent_raw_refuses_unowned_access_and_reopening_after_close(source_fixture):
    disk, _, _ = source_fixture
    source = runner.PersistentRawSource(disk.schedule, 10_000)
    with pytest.raises(RuntimeError, match="owner thread"):
        source.acquire_raw(0)
    source.warmup()
    errors = []

    def wrong_thread():
        for method in (lambda: source.acquire_raw(0), source.finalize):
            try:
                method()
            except RuntimeError as exc:
                errors.append(str(exc))

    worker = threading.Thread(target=wrong_thread)
    worker.start()
    worker.join()
    assert len(errors) == 2
    source.finalize()
    with pytest.raises(RuntimeError, match="only once"):
        source.warmup()


def test_persistent_raw_closes_on_caller_cancellation(source_fixture):
    disk, _, _ = source_fixture
    source = runner.PersistentRawSource(disk.schedule, 10_000)

    class Cancelled(BaseException):
        pass

    def infer(arrival):
        raise Cancelled()

    with pytest.raises(Cancelled):
        run_paced_replay(
            source.acquire_raw,
            infer,
            ["a"],
            updates=3,
            source_warmup=source.warmup,
            source_finalize=source.finalize,
            wait_strategy="sleep",
        )
    assert source.closed and source.stats["closed_on_owner_thread"]
    assert source.inputs._reader is None


def test_persistent_raw_closes_after_acquisition_errors(source_fixture, monkeypatch):
    disk, _, _ = source_fixture
    source = runner.PersistentRawSource(disk.schedule, 10_000)

    def broken_box(item):
        raise ValueError("annotation changed")

    monkeypatch.setattr(runner, "_rgb_box", broken_box)
    result = run_paced_replay(
        source.acquire_raw,
        lambda item: pytest.fail("No usable payload"),
        ["a"],
        updates=3,
        source_warmup=source.warmup,
        source_finalize=source.finalize,
        wait_strategy="sleep",
    )
    assert result["source_errors"] == 3
    assert source.closed and source.stats["closed_on_owner_thread"]
    assert not result["grouped"]["passed"]


def test_source_and_geometry_run_on_separate_threads_with_contiguous_updates(source_fixture):
    source, reads, geometry = source_fixture
    service = Service()
    consumer = runner.ContinuousConsumer(service, source.schedule, 20)
    main = threading.get_ident()
    result = run_paced_replay(
        source.acquire_raw,
        consumer.infer_all,
        [f"roi_{index:02d}" for index in range(20)],
        updates=3,
        rate_hz=50,
        queue_capacity=3,
    )
    assert result["grouped"]["passed"]
    assert len(result["object_records"]) == 60
    assert set(reads) != {main}
    assert set(geometry) == {main}
    assert [(a, b) for a, b, _ in service.calls] == [
        (980_000, 1_000_000),
        (1_000_000, 1_020_000),
        (1_020_000, 1_040_000),
    ]
    assert consumer.failure is None
    assert not result["capacity_checks_passed"]


def test_gap_latches_without_attempting_stale_service_update(source_fixture):
    source, _, _ = source_fixture
    service = Service()
    consumer = runner.ContinuousConsumer(service, source.schedule, 1)
    consumer.infer_all(Arrival("update-0", 0, 0, source.acquire_raw(0)))
    with pytest.raises(ValueError, match="RAW_GAP"):
        consumer.infer_all(Arrival("update-2", 0, 0, source.acquire_raw(2)))
    with pytest.raises(RuntimeError, match="REINITIALIZATION_REQUIRED"):
        consumer.infer_all(Arrival("update-1", 0, 0, source.acquire_raw(1)))
    assert len(service.calls) == 1


@pytest.mark.parametrize("kind", ["duplicate", "nan", "none", "tensor", "missing", "failed"])
def test_invalid_full_service_result_fails_and_latches(source_fixture, kind):
    source, _, _ = source_fixture

    def mutate(result):
        if kind == "duplicate":
            result["objects"][1]["object_id"] = result["objects"][0]["object_id"]
        elif kind == "missing":
            result["objects"].pop()
        elif kind == "failed":
            result["objects"][0]["success"] = False
        else:
            result["objects"][0]["predictions"]["E_CTX_TRAIN40"] = {
                "nan": float("nan"),
                "none": None,
                "tensor": runner.torch.tensor(1.0),
            }[kind]

    consumer = runner.ContinuousConsumer(Service(mutate), source.schedule, 2)
    with pytest.raises(ValueError):
        consumer.infer_all(Arrival("update-0", 0, 0, source.acquire_raw(0)))
    assert consumer.failure is not None
    assert not consumer.service_receipts


def test_payload_limit_remains_visible_as_source_failure(source_fixture):
    source, _, _ = source_fixture
    source.max_payload_bytes = 10
    result = run_paced_replay(
        source.acquire_raw, lambda arrival: {}, ["roi_00"], updates=3, rate_hz=200, queue_capacity=2
    )
    assert result["source_errors"] == 3
    assert not result["grouped"]["passed"]
    assert not result["capacity_checks_passed"]


def test_events_reject_before_any_array_or_chunk_is_allocated():
    class TooLarge:
        def _bounds(self, start, end):
            return 0, 100

        def iter_window_chunks(self, *args):
            pytest.fail("Should reject before reading raw arrays")

    with pytest.raises(ValueError, match="BYTE_LIMIT"):
        runner.bounded_events(TooLarge(), 0, 20_000, 1699)


def test_future_bbox_is_rejected_before_reading_sensor(source_fixture):
    source, reads, _ = source_fixture
    source.schedule[0]["bbox_sources"][2]["available_us"] = 1_000_001
    with pytest.raises(ValueError, match="Future"):
        source.acquire_raw(0)
    assert reads == []


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "drop",
        "source",
        "coverage",
        "arrival",
        "short",
        "parity",
        "unchecked",
        "originals",
        "lifecycle",
    ],
)
def test_capacity_requires_every_gate_including_final_drops(failure):
    paced = {
        "grouped": {"passed": True},
        "dropped_updates": 0,
        "source_errors": 0,
        "capacity_checks_passed": True,
        "capacity_checks": {name: True for name in runner.MILESTONE_CHECKS},
        "object_ids": [f"roi_{index:02d}" for index in range(20)],
        "object_records": [
            {
                "success": True,
                "result": {"predictions": {name: 1.0 for name in PREDICTION_COMPONENTS}},
            }
        ],
    }
    parity = {"performed": True, "passed": True}
    if failure == "drop":
        paced["dropped_updates"] = 1
    elif failure == "source":
        paced["source_errors"] = 1
    elif failure == "coverage":
        paced["grouped"]["passed"] = False
    elif failure == "arrival":
        paced["capacity_checks"]["all_scheduled_arrivals_recorded"] = False
    elif failure == "short":
        paced["capacity_checks_passed"] = False
        paced["capacity_checks"]["offered_50hz_for_at_least_60s"] = False
    elif failure == "parity":
        parity["passed"] = False
    elif failure == "unchecked":
        parity["performed"] = False
    elif failure == "lifecycle":
        paced["capacity_checks"]["source_lifecycle_succeeded"] = False
    result = runner.capacity_admission(paced, parity, failure != "originals")
    assert result["capacity_with_parity_passed"] == (failure is None)
    if failure in ("drop", "source", "coverage", "arrival", "lifecycle"):
        assert not result["complete_measurement_coverage"]


@pytest.mark.parametrize("objects", [1, 2, 20])
@pytest.mark.parametrize("missing_rgb", [False, True])
def test_versioned_milestone_never_relabels_one_or_two_objects_as_twenty(objects, missing_rgb):
    predictions = {name: 1.0 for name in PREDICTION_COMPONENTS}
    if missing_rgb:
        predictions["R_CTX_TRAIN40"] = None
    paced = {
        "grouped": {"passed": True},
        "dropped_updates": 0,
        "source_errors": 0,
        "capacity_checks_passed": objects == 20,
        "capacity_checks": {
            **{name: True for name in runner.MILESTONE_CHECKS},
            "twenty_distinct_objects_per_update": objects == 20,
        },
        "object_ids": [f"roi_{index:02d}" for index in range(objects)],
        "object_records": [{"success": True, "result": {"predictions": predictions}}],
    }
    parity = {"performed": True, "passed": True}
    value = runner.capacity_admission(paced, parity, True, target_object_count=objects)
    assert value["schema"] == "train40_paced_milestone_gate_v2"
    assert value["target_object_count"] == objects
    assert value["milestone_with_parity_passed"] is not missing_rgb
    assert value["capacity_with_parity_passed"] == (objects == 20 and not missing_rgb)
    assert paced["capacity_checks_passed"] == (objects == 20)


@pytest.mark.parametrize("failure", [None, "receipt", "prediction", "missing_update"])
def test_separate_parity_covers_all_updates_and_objects(source_fixture, monkeypatch, failure):
    source, _, _ = source_fixture
    consumer = runner.ContinuousConsumer(Service(), source.schedule, 2)
    prediction = {
        "predictions": {name: 1.0 for name in PREDICTION_COMPONENTS},
        "rgb_available": True,
        "rgb_unavailable_reason": "",
        "history_lengths": {"event": 4, "rgb": 5},
    }
    for index in range(3):
        consumer.service_receipts.append(
            {
                "update_id": f"update-{index}",
                "objects": [
                    {
                        **copy.deepcopy(prediction),
                        "object_id": f"roi_{obj:02d}",
                        "input_receipt": {"checked": True},
                    }
                    for obj in range(2)
                ],
            }
        )
    if failure == "receipt":
        consumer.service_receipts[1]["objects"][1]["input_receipt"] = {"checked": False}
    elif failure == "prediction":
        consumer.service_receipts[1]["objects"][1]["predictions"]["E_CTX_TRAIN40"] = 1.1
    elif failure == "missing_update":
        consumer.service_receipts.pop(1)
    calls = []

    def original_predict(self, row):
        calls.append(row["anchor_us"])
        return copy.deepcopy(prediction)

    monkeypatch.setattr(runner.FinalTransferPredictor, "predict", original_predict)
    monkeypatch.setattr(runner.engine, "prepare_event", lambda row: {})
    monkeypatch.setattr(runner, "reference_rgb", lambda *args: [])
    monkeypatch.setattr(runner, "reference_receipt", lambda *args: {"checked": True})
    result = runner.separate_parity(source, consumer, {}, {})
    assert calls == [1_000_000] * 2 + [1_020_000] * 2 + [1_040_000] * 2
    assert len(result["updates"]) == 3
    assert result["passed"] == (failure is None)
    if failure:
        assert result["updates"][1]["passed"] is False


def test_cli_defaults_and_help_do_not_initialize_cuda(monkeypatch, capsys):
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda: pytest.fail("CUDA touched"))
    args = runner.parser().parse_args(
        [
            "--candidate",
            "candidate",
            "--reference",
            "reference",
            "--base-manifest",
            "base.json",
            "--manifest",
            "replay.json",
            "--output",
            "fresh",
            "--sequence",
            "sequence",
        ]
    )
    assert args.objects == 1 and args.queue_capacity == 2
    assert args.event_query_graphs and args.event_normalization == "topk"
    assert args.verify_parity is False
    assert args.wait_strategy == "sleep"
    legacy = runner.parser().parse_args(
        [
            "--candidate",
            "candidate",
            "--reference",
            "reference",
            "--base-manifest",
            "base.json",
            "--manifest",
            "replay.json",
            "--output",
            "fresh",
            "--sequence",
            "sequence",
            "--wait-strategy",
            "event",
        ]
    )
    assert legacy.wait_strategy == "event"
    with pytest.raises(SystemExit) as caught:
        runner.parser().parse_args(["--help"])
    assert caught.value.code == 0
    assert "--verify-parity" in capsys.readouterr().out


def test_failure_still_verifies_sources_and_writes_summary(tmp_path, monkeypatch):
    replay = manifest(tmp_path)
    base = {"rows": []}
    replay["provenance"]["base_manifest_canonical_sha256"] = runner._canonical_sha256(base)
    base_path, replay_path = tmp_path / "inputs" / "base.json", tmp_path / "inputs" / "replay.json"
    monkeypatch.setattr(runner, "read", lambda path: base if path == base_path else replay)
    monkeypatch.setattr(runner, "verify_candidate", lambda path: ({}, {}))
    monkeypatch.setattr(runner, "verify_frozen_runtime", lambda path: {})
    monkeypatch.setattr(runner, "_validate_reference", lambda *args: ([], [], {}))
    monkeypatch.setattr(runner, "_source_inventory", lambda *args: [])
    monkeypatch.setattr(runner, "_candidate_inventory", lambda *args: [])
    monkeypatch.setattr(runner, "snapshot_files", lambda *args: {"original": {}})
    checks = []
    monkeypatch.setattr(
        runner,
        "verify_snapshot",
        lambda value: checks.append(value) or {"passed": False, "modified": ["original"]},
    )
    monkeypatch.setattr(runner, "sha", lambda path: "sha")
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda: "fake-no-cuda")

    def broken_load(*args):
        raise RuntimeError("model loading failed")

    monkeypatch.setattr(runner, "load_models", broken_load)
    args = runner.parser().parse_args(
        [
            "--candidate",
            str(tmp_path / "candidate"),
            "--reference",
            str(tmp_path / "reference"),
            "--base-manifest",
            str(base_path),
            "--manifest",
            str(replay_path),
            "--output",
            str(tmp_path / "result"),
            "--sequence",
            "sequence",
        ]
    )
    with pytest.raises(RuntimeError, match="model loading failed"):
        runner.run(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert checks == [{"original": {}}]
    assert summary["status"] == "FAILED"
    assert not summary["originals"]["passed"]
    assert not summary["capacity_with_parity_passed"]
