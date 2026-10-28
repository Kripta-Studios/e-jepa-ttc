"""Declared artificial populations and independent sensor-history receipts."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from operational.train40_runtime.multiobject import ObjectRegion, RawRGBHistory
from scripts import benchmark_train40_multiobject as runner
from scripts.benchmark_train40_multiobject import (
    artificial_regions,
    reference_receipt,
    reference_rgb,
)


def test_default_profile_preserves_admitted_batches_and_leaves_experiments_opt_in():
    arguments = [
        "--candidate",
        "candidate",
        "--manifest",
        "manifest",
        "--reference",
        "reference",
        "--output",
        "fresh",
    ]
    options = runner.parser().parse_args(arguments)
    assert options.counts == [1, 2]
    assert options.no_event_reuse and options.rgb_object_chunk_size == 1
    assert options.backend == "prefix" and options.event_query_graphs
    assert options.event_normalization == "topk" and options.parallel
    assert not options.rgb_uint8 and not options.rgb_feature_cache
    assert not options.shared_rgb_images and not options.serial_preprocessing
    assert not options.overlap_event_with_rgb_preparation
    assert not options.immutable_rgb_history
    experimental = runner.parser().parse_args(
        arguments + ["--event-reuse", "--rgb-uint8", "--overlap-event-with-rgb-preparation"]
    )
    assert not experimental.no_event_reuse and experimental.rgb_uint8
    assert experimental.overlap_event_with_rgb_preparation
    immutable = runner.parser().parse_args(
        arguments + ["--immutable-rgb-history", "--rgb-feature-cache"]
    )
    assert immutable.immutable_rgb_history and immutable.rgb_feature_cache


def admission_history():
    frames = {i: np.full((7, 9, 3), i, dtype=np.uint8) for i in (3, 5, 7)}
    history = RawRGBHistory(300, frames, {3: 100, 5: 200, 7: 300}, [(3, 5, 7)])
    row = {
        "anchor_us": 300,
        "raw_path": "source.hdf5",
        "raw_stat": {"size_bytes": 123, "mtime_ns": 456},
        "calibration_sha256": "a" * 64,
    }
    return history, row


def test_immutable_admission_seals_real_factory_tokens_and_preserves_independent_raw():
    from operational.train40_runtime.rgb_immutable_history import FrozenRGBHistory

    history, row = admission_history()
    originals = {key: value.copy() for key, value in history.frames.items()}
    frozen, receipt = runner.immutable_history_for_admission(history, row, "b" * 64)
    assert type(frozen) is FrozenRGBHistory
    assert receipt["all_raw_frames_exact"]
    assert [item["timestamp_us"] for item in receipt["frames"]] == [100, 200, 300]
    assert receipt["hashed_bytes"] == sum(frame.nbytes for frame in history.frames.values())
    assert receipt["sealing_excluded_from_internal_update_clock"]
    assert not receipt["arrival_stream_simulated"]
    assert not receipt["source_revision_is_full_hdf5_hash"]
    for key, frame in history.frames.items():
        np.testing.assert_array_equal(frame, originals[key])
        np.testing.assert_array_equal(frozen.frames[key], originals[key])
        assert not np.shares_memory(frame, frozen.frames[key])
    history.frames[7][:] = 99
    np.testing.assert_array_equal(frozen.frames[7], originals[7])


def test_future_frame_is_rejected_before_any_factory_hash(monkeypatch):
    from operational.train40_runtime import rgb_raw_ring

    history, row = admission_history()
    cast(dict[int, int], history.timestamps_us)[7] = 301
    monkeypatch.setattr(
        rgb_raw_ring, "freeze_rgb_frame", lambda *args, **kwargs: pytest.fail("future hash")
    )
    with pytest.raises(ValueError, match="Future RGB"):
        runner.immutable_history_for_admission(history, row, "b" * 64)


def test_producer_capture_observes_actual_arguments_and_restores_methods_on_failure():
    pixels, delta = torch.zeros((1, 3, 3, 128, 128)), torch.ones((1, 2))

    def block(modality, sensor, clocks, **kwargs):
        return sensor, clocks

    def async_block(modality, sensor, clocks, **kwargs):
        raise RuntimeError("producer failure")

    producer = SimpleNamespace(expert_block=block, expert_block_async=async_block)
    service = SimpleNamespace(producers=producer)
    with pytest.raises(RuntimeError, match="producer failure"):
        with runner.capture_rgb_producer_inputs(cast(Any, service)) as captured:
            producer.expert_block("E", pixels, delta)
            producer.expert_block("R", pixels, delta)
            producer.expert_block_async("R", pixels, delta)
    assert producer.expert_block is block and producer.expert_block_async is async_block
    assert len(captured) == 2
    assert captured[0]["sensor"] is pixels and captured[0]["delta"] is delta
    assert [row["route"] for row in runner.captured_rgb_digests(captured)] == [
        "expert_block",
        "expert_block_async",
    ]


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("tamper", [None, "pixels", "deltas", "missing_chunk", "hit_calls"])
def test_actual_device_argument_attestation_cannot_substitute_cpu_reconstruction(count, tamper):
    references = [
        [(np.full((3, 3, 128, 128), index / 255, np.float32), {"delta_t_s": [0.1, 0.100001]})]
        for index in range(1, count + 1)
    ]
    captured = [
        {
            "route": "expert_block_async",
            "sensor": torch.from_numpy(pixels[None].copy()),
            "delta": torch.tensor([meta["delta_t_s"]], dtype=torch.float32),
        }
        for group in references
        for pixels, meta in group
    ]
    if tamper == "pixels":
        captured[0]["sensor"][0, 0, 0, 0, 0] = 1
    elif tamper == "deltas":
        captured[0]["delta"][0, 0] = 0.5
    elif tamper == "missing_chunk":
        captured.pop()
    result = {
        "rgb_feature_cache": {"status": "HIT" if tamper == "hit_calls" else "MISS"},
        "actual_rgb_producer_inputs": runner.captured_rgb_digests(captured),
    }
    check = runner.attest_actual_rgb_calls(result, references, 1)
    assert check["passed"] == (tamper is None)
    result["rgb_feature_cache"]["status"] = "HIT"
    result["actual_rgb_producer_inputs"] = []
    assert runner.attest_actual_rgb_calls(result, references, 1)["passed"]


def test_artificial_population_keeps_primary_and_creates_distinct_regions():
    event, rgb = [10.0, 20.0, 100.0, 110.0], [2.0, 3.0, 92.0, 93.0]
    original = copy.deepcopy((event, rgb))
    population = artificial_regions(event, rgb, 20)
    assert len({item.object_id for item in population}) == 20
    assert len({item.event_square for item in population}) == 20
    assert len({item.rgb_square for item in population}) == 20
    assert population[0].event_square == tuple(event)
    assert population[0].rgb_square == tuple(rgb)
    assert (event, rgb) == original


@pytest.mark.parametrize("count", [0, 21, -1])
def test_invalid_capacity_population_rejected(count):
    with pytest.raises(ValueError):
        artificial_regions((0, 0, 10, 10), (0, 0, 10, 10), count)


def test_reference_receipt_binds_exact_windows_and_frame_ids():
    row = {
        "anchor_us": 1_000_003,
        "windows_us": [[700_003, 800_003], [800_003, 900_003], [900_003, 1_000_003]],
    }
    obj = ObjectRegion("a", (0.0, 0.0, 32.0, 32.0), (0.0, 0.0, 32.0, 32.0))
    frames = {i: np.full((32, 32, 3), i, dtype=np.uint8) for i in (3, 5, 7)}
    history = RawRGBHistory(1_000_003, frames, {3: 800_001, 5: 900_001, 7: 1_000_001}, [(3, 5, 7)])
    pixels = reference_rgb(history, obj)
    receipt = reference_receipt(row, obj, pixels)
    assert receipt["query_us"] == 1_000_003
    assert receipt["event_windows_us"][0][0] == [400_003, 500_003]
    assert receipt["event_windows_us"][-1][-1] == [900_003, 1_000_003]
    assert receipt["rgb_observations"][0]["frame_indices"] == [3, 5, 7]
    assert receipt["rgb_observations"][0]["timestamps_us"] == [800_001, 900_001, 1_000_001]
    assert pixels[0][0].shape == (3, 3, 128, 128)
    assert pixels[0][0].dtype == np.float32


@pytest.mark.parametrize("immutable_history", [False, True])
def test_cleanup_failures_still_verify_originals_and_write_summary(
    tmp_path, monkeypatch, immutable_history
):
    class BrokenCloser:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            raise RuntimeError("close failure")

    checked = []
    raw_path = tmp_path / "raw.hdf5"
    raw_path.write_bytes(b"fixture never opened")
    stat = raw_path.stat()
    row = {
        "query_id": "q",
        "raw_path": str(raw_path),
        "bbox_sources": [],
        "raw_stat": {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
    }
    monkeypatch.setattr(runner, "verify_candidate", lambda path: ({}, {"components": {}}))
    monkeypatch.setattr(runner, "verify_frozen_runtime", lambda path: {})
    monkeypatch.setattr(runner, "read", lambda path: {"rows": [row]})
    monkeypatch.setattr(runner, "_validate_reference", lambda *args: ([{}], [], {}))
    monkeypatch.setattr(runner, "_source_inventory", lambda *args: [])
    monkeypatch.setattr(runner, "_candidate_inventory", lambda *args: [])
    monkeypatch.setattr(runner, "snapshot_files", lambda *args: {"sentinel": {}})
    monkeypatch.setattr(
        runner, "verify_snapshot", lambda value: checked.append(value) or {"passed": True}
    )
    monkeypatch.setattr(runner, "sha", lambda path: "test-digest")
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda: "mock-no-cuda")
    monkeypatch.setattr(runner, "load_models", lambda *args: ({}, {}))
    monkeypatch.setattr(runner, "IntegralEventHistory", lambda **kwargs: object())
    monkeypatch.setattr(runner, "PersistentSensorInputs", BrokenCloser)
    monkeypatch.setattr(runner, "MultiObjectTrain40Service", BrokenCloser)

    def fail_prime(*args):
        raise RuntimeError("priming failure")

    monkeypatch.setattr(runner, "prime", fail_prime)
    args = argparse.Namespace(
        candidate=tmp_path / "candidate",
        reference=tmp_path / "reference",
        manifest=tmp_path / "data" / "manifest.json",
        output=tmp_path / "result",
        indices=[0],
        counts=[1],
        repeats=2 if immutable_history else 1,
        backend="integral",
        parallel=False,
        rgb_workers=1,
        no_event_reuse=False,
        shared_rgb_images=False,
        rgb_object_chunk_size=None,
        serial_preprocessing=False,
        overlap_event_with_rgb_preparation=False,
        event_query_graphs=False,
        event_normalization="sort",
        rgb_feature_cache=immutable_history,
        rgb_uint8=False,
        immutable_rgb_history=immutable_history,
    )
    with pytest.raises(RuntimeError, match="priming failure"):
        runner.run(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["status"] == "FAILED"
    assert summary["originals"]["passed"] is True
    assert summary["cleanup_errors"] == ["RuntimeError: close failure"] * 2
    assert checked == [{"sentinel": {}}]
    assert summary["immutable_rgb_history"] is immutable_history
    if immutable_history:
        assert summary["scope"] == runner.IMMUTABLE_ADMISSION_SCOPE
        assert summary["raw_source_stat_checks"][0]["passed"]
        assert not summary["schedule_exact"]


@pytest.mark.parametrize("tamper", [None, "pixels", "features", "key", "event", "missing_miss"])
def test_cache_provenance_requires_checked_prior_pixels_and_exact_features(tamper):
    miss = {
        "event_sha256": "event",
        "rgb_sha256": ["canonical"],
        "rgb_cache_key": "key",
        "rgb_producer_features": {"raw": "r", "experts": "e"},
    }
    hit = {**miss, "rgb_sha256": None}
    timings = [{"input_digests": [copy.deepcopy(miss)]}, {"input_digests": [copy.deepcopy(hit)]}]
    if tamper == "pixels":
        timings[0]["input_digests"][0]["rgb_sha256"] = ["wrong"]
    elif tamper == "features":
        timings[1]["input_digests"][0]["rgb_producer_features"]["raw"] = "wrong"
    elif tamper == "key":
        timings[1]["input_digests"][0]["rgb_cache_key"] = "wrong"
    elif tamper == "event":
        timings[0]["input_digests"][0]["event_sha256"] = "wrong"
    elif tamper == "missing_miss":
        timings = timings[1:]
    runner.attest_cached_inputs(timings, [["canonical"]])
    assert all(row["same_inputs_as_tensor_checked_repeat"] for row in timings) == (tamper is None)


@pytest.mark.parametrize(
    "mode", ["valid", "one_miss", "all_miss", "not_stored", "hit_pixels", "hit_only"]
)
def test_cache_experiment_requires_an_exercised_hit_after_stored_miss(mode):
    miss = {
        "rgb_feature_cache": {"status": "MISS", "stored": True, "present_objects": 2},
        "input_digests": [{"rgb_sha256": ["pixels"]}] * 2,
    }
    hit = {
        "rgb_feature_cache": {"status": "HIT", "present_objects": 2},
        "input_digests": [{"rgb_sha256": None}] * 2,
    }
    records = [copy.deepcopy(miss), copy.deepcopy(hit)]
    if mode == "one_miss":
        records = [miss]
    elif mode == "all_miss":
        records = [miss, copy.deepcopy(miss)]
    elif mode == "not_stored":
        records[0]["rgb_feature_cache"]["stored"] = False
    elif mode == "hit_pixels":
        records[1]["input_digests"][0]["rgb_sha256"] = ["pixels"]
    elif mode == "hit_only":
        records = [hit]
    result = runner.cache_exercise_receipt(records)
    assert result["eligible"]
    assert result["passed"] == (mode == "valid")


def test_runtime_import_closure_detects_a_late_experimental_dependency(monkeypatch):
    before = runner.imported_runtime_paths()
    name = "operational.train40_runtime.new_late_test_module"
    module = ModuleType(name)
    module.__file__ = str(runner.ROOT / "operational/train40_runtime/new_late_test_module.py")
    monkeypatch.setitem(sys.modules, name, module)
    added = runner.imported_runtime_paths() - before
    assert added == {runner.ROOT / "operational/train40_runtime/new_late_test_module.py"}
