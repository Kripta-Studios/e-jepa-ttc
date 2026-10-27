"""Declared artificial populations and independent sensor-history receipts."""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pytest

from operational.train40_runtime.multiobject import ObjectRegion, RawRGBHistory
from scripts import benchmark_train40_multiobject as runner
from scripts.benchmark_train40_multiobject import (
    artificial_regions,
    reference_receipt,
    reference_rgb,
)


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


def test_cleanup_failures_still_verify_originals_and_write_summary(tmp_path, monkeypatch):
    class BrokenCloser:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            raise RuntimeError("close failure")

    checked = []
    monkeypatch.setattr(runner, "verify_candidate", lambda path: ({}, {}))
    monkeypatch.setattr(runner, "verify_frozen_runtime", lambda path: {})
    monkeypatch.setattr(runner, "read", lambda path: {"rows": [{"query_id": "q"}]})
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
        repeats=1,
        backend="integral",
        parallel=False,
        rgb_workers=1,
        no_event_reuse=False,
        shared_rgb_images=False,
        rgb_object_chunk_size=None,
        serial_preprocessing=False,
        event_query_graphs=False,
        event_normalization="sort",
        rgb_feature_cache=False,
    )
    with pytest.raises(RuntimeError, match="priming failure"):
        runner.run(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["status"] == "FAILED"
    assert summary["originals"]["passed"] is True
    assert summary["cleanup_errors"] == ["RuntimeError: close failure"] * 2
    assert checked == [{"sentinel": {}}]


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
