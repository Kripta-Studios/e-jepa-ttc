"""CPU parity and ownership gates for original-PIL crop reuse in the real packer."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from operational.train40_runtime import multiobject
from operational.train40_runtime.multiobject import (
    MultiObjectTrain40Service,
    ObjectRegion,
    RawRGBHistory,
)
from operational.train40_runtime.multiobject_cache import RGBBatchCache
from operational.train40_runtime.rgb_crop_cache import CROP_BYTES
from operational.train40_runtime.rgb_immutable_history import FrozenRGBHistory
from operational.train40_runtime.rgb_raw_ring import RGBSourceIdentity, freeze_rgb_frame
from scripts import benchmark_train40_paced as runner


class Producers:
    def __init__(self):
        self.calls = []

    def expert_block(self, modality, sensor, delta, *, history_group_size):
        self.calls.append((modality, tuple(sensor.shape), history_group_size))
        values = torch.arange(len(sensor), dtype=torch.float32)[:, None]
        return values.expand(-1, 17), values.expand(-1, 3)


class Context:
    def __init__(self):
        self.observations = []

    def predict_many(self, observations):
        self.observations = observations
        return [{"object_id": item.object_id, "predictions": {"TTC": 1.0}} for item in observations]


class History:
    def __init__(self):
        self.end = 980_000

    def ingest(self, raw, start, end):
        assert start == self.end and end - start == 20_000
        self.end = end

    def query_many(self, rois):
        return torch.zeros((len(rois), 4, 3, 12, 128, 128))


@pytest.fixture
def services(monkeypatch):
    monkeypatch.setattr(multiobject, "ProducerRuntime", lambda *args, **kwargs: Producers())
    monkeypatch.setattr(multiobject, "BatchedContextRuntime", lambda *args, **kwargs: Context())
    values = []

    def create(enabled=True, **kwargs):
        value = MultiObjectTrain40Service(
            {},
            {},
            History(),
            "cpu",
            graphs=False,
            parallel_branches=False,
            max_objects=2,
            rgb_workers=2,
            rgb_crop_cache=enabled,
            **kwargs,
        )
        values.append(value)
        return value

    yield create
    for value in values:
        value.close()


def objects(count=2):
    return [
        ObjectRegion(str(i), (0.0, 0.0, 32.0, 32.0), (-5.25 + i, 2.125, 49.5 + i, 56.875))
        for i in range(count)
    ]


def tokens(count=15, source=None):
    origin = source or RGBSourceIdentity("test", "a" * 64)
    return {
        i: freeze_rgb_frame(
            np.random.default_rng(i).integers(0, 256, (61, 47, 3), dtype=np.uint8),
            i,
            400_000 + i * 50_000,
            origin,
        )
        for i in range(count)
    }


def frozen(values, ids, query=None):
    return FrozenRGBHistory(
        max(values[i].timestamp_us for i in ids) if query is None else query,
        {i: values[i] for i in ids},
        [tuple(ids[i : i + 3]) for i in range(len(ids) - 2)],
    )


def pack(service, regions, history):
    return service._preprocess.submit(service._pack_rgb, regions, history).result()


def assert_pixels_and_metadata_equal(actual, expected):
    assert len(actual) == len(expected)
    for rows, originals in zip(actual, expected, strict=True):
        assert len(rows) == len(originals)
        for (pixels, meta), (original, reference) in zip(rows, originals, strict=True):
            np.testing.assert_array_equal(pixels, original)
            assert pixels.dtype == np.float32 and pixels.flags.c_contiguous
            assert meta == reference


@pytest.mark.parametrize("count", [1, 2])
def test_causal_cohorts_partial_hits_subset_roi_and_source_match_original_pack(services, count):
    optimized, original = services(), services(False)
    values = tokens()
    regions = objects(count)
    streams = [
        (range(0, 13, 2), regions),
        (range(1, 14, 2), regions),
        (range(3, 14, 2), regions),
        (range(2, 15, 2), regions),
    ]
    hits, tasks = [], []
    for ids, current in streams:
        history = frozen(values, list(ids))
        actual = pack(optimized, current, history)
        assert_pixels_and_metadata_equal(actual, pack(original, current, history))
        hits.append(optimized.last_rgb_profile["crop_cache_hits"])
        tasks.append(optimized.last_rgb_profile["roi_frame_tasks"])
    assert hits == [0, 0, 6 * count, 6 * count]
    assert tasks == [7 * count, 7 * count, 0, count]
    changed = [replace(regions[0], rgb_square=(-5.24999, 2.125, 49.5, 56.875)), *regions[1:]]
    history = frozen(values, list(range(2, 15, 2)))
    assert_pixels_and_metadata_equal(
        pack(optimized, changed, history), pack(original, changed, history)
    )
    assert optimized.last_rgb_profile["crop_cache_misses"] == 7
    assert optimized.last_rgb_profile["crop_cache_hits"] == 7 * (count - 1)
    new_source = frozen(tokens(source=RGBSourceIdentity("new", "b" * 64)), list(range(2, 15, 2)))
    assert_pixels_and_metadata_equal(
        pack(optimized, regions, new_source), pack(original, regions, new_source)
    )
    assert optimized.last_rgb_profile["crop_cache_hits"] == 0
    assert optimized.rgb_crop_cache_stats["crop_bytes"] <= 2 * 16 * CROP_BYTES
    assert optimized.rgb_crop_cache_stats["raw_payload_bytes_hashed"] == 0


def test_legacy_source_bypasses_without_lookup_or_storage(services, monkeypatch):
    optimized, original = services(), services(False)
    history = frozen(tokens(), list(range(0, 13, 2)))
    legacy = RawRGBHistory(
        history.query_us, history.frames, history.timestamps_us, history.triplets
    )
    monkeypatch.setattr(
        optimized._rgb_crop_cache, "get", lambda *args, **kwargs: pytest.fail("get")
    )
    monkeypatch.setattr(
        optimized._rgb_crop_cache, "put", lambda *args, **kwargs: pytest.fail("put")
    )
    assert_pixels_and_metadata_equal(
        pack(optimized, objects(), legacy), pack(original, objects(), legacy)
    )
    assert optimized.last_rgb_profile["crop_cache_bypassed_frames"] == 14
    assert optimized.last_rgb_profile["crop_cache_hits"] == 0
    assert optimized.rgb_crop_cache_stats["entries"] == 0


def test_whole_feature_hit_skips_crop_cache_and_keeps_current_context(services, monkeypatch):
    service = services()
    service.rgb_feature_cache = RGBBatchCache(
        model_sha256="a" * 64,
        runtime_sha256="b" * 64,
        rgb_object_chunk_size=1,
    )
    history = frozen(tokens(), list(range(0, 13, 2)))
    first = service.update({}, 980_000, 1_000_000, objects(), history)
    assert first["rgb_feature_cache"]["status"] == "MISS"
    assert service.producers.calls[-2:] == [("R", (5, 3, 3, 128, 128), 5)] * 2
    before = service.rgb_crop_cache_stats
    monkeypatch.setattr(service, "_pack_rgb", lambda *args: pytest.fail("whole HIT packed pixels"))
    newer = FrozenRGBHistory(1_020_000, history.tokens, history.triplets)
    service.producers.calls.clear()
    second = service.update({}, 1_000_000, 1_020_000, objects(), newer)
    assert second["rgb_feature_cache"]["status"] == "HIT"
    assert second["rgb_preprocess"]["crop_cache_hits"] == 0
    assert second["rgb_preprocess"]["crop_cache_misses"] == 0
    assert second["rgb_crop_cache"] == before
    assert service.producers.calls == [("E", (8, 3, 12, 128, 128), 4)]
    assert all(item.query_us == 1_020_000 for item in service.context.observations)


def test_close_joins_coordinator_and_releases_storage(services):
    service = services()
    rows = pack(service, objects(), frozen(tokens(), list(range(0, 13, 2))))
    assert service.rgb_crop_cache_stats["entries"] == 14
    expected = rows[0][0][0].copy()
    service.close()
    assert service.rgb_crop_cache_stats["closed"]
    assert service.rgb_crop_cache_stats["crop_bytes"] == 0
    np.testing.assert_array_equal(rows[0][0][0], expected)


@pytest.mark.parametrize("options", [{"rgb_uint8": True}, {"shared_rgb_images": True}])
def test_incompatible_pixel_abi_rejected_before_models(options):
    with pytest.raises(ValueError, match="ROI-first"):
        MultiObjectTrain40Service({}, {}, History(), rgb_crop_cache=True, **options)


def cli_args(*extra):
    return runner.parser().parse_args(
        [
            "--candidate",
            "candidate",
            "--reference",
            "reference",
            "--base-manifest",
            "base.json",
            "--manifest",
            "manifest.json",
            "--output",
            "new-output",
            "--sequence",
            "test",
            *extra,
        ]
    )


def test_cli_default_and_runtime_identity_bind_crop_option(monkeypatch):
    default, enabled = cli_args(), cli_args("--rgb-crop-cache", "--objects", "2")
    assert not default.rgb_crop_cache
    assert runner.service_options(default)["rgb_crop_cache"] is False
    assert runner.service_options(enabled)["rgb_crop_cache"] is True
    assert runner.rgb_crop_cache_contract(default)["additional_numeric_upper_bound_bytes"] == 0
    contract = runner.rgb_crop_cache_contract(enabled)
    assert contract["additional_numeric_upper_bound_bytes"] == 33 * CROP_BYTES
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: None)
    binding = {"candidate": {"weights": "preserved"}, "sources": {"runtime": "a" * 64}}
    a, b = runner.cache_identity(binding, default), runner.cache_identity(binding, enabled)
    assert a["model_sha256"] == b["model_sha256"]
    assert a["runtime_sha256"] != b["runtime_sha256"]


@pytest.mark.parametrize(
    "hits,misses,stores,passed", [(0, 7, 7, False), (6, 0, 0, False), (6, 7, 7, True)]
)
def test_crop_exercise_requires_real_hit_and_original_pil_miss(hits, misses, stores, passed):
    assert (
        runner.rgb_crop_cache_exercise(
            [
                {
                    "rgb_preprocess": {
                        "crop_cache_hits": hits,
                        "crop_cache_misses": misses,
                        "crop_cache_stored_crops": stores,
                    }
                }
            ]
        )["passed"]
        is passed
    )


def test_memory_budget_reserves_crop_bytes_before_source_preload(tmp_path, monkeypatch):
    args = cli_args("--rgb-crop-cache", "--raw-source", "memory", "--objects", "2")
    args.output = tmp_path / "new-run"
    monkeypatch.setattr(runner, "_validate_output", lambda *args: None)
    monkeypatch.setattr(runner, "verify_candidate", lambda *args: ({}, {}))
    monkeypatch.setattr(runner, "verify_frozen_runtime", lambda *args: {})
    monkeypatch.setattr(runner, "_validate_reference", lambda *args: ([], [], {}))
    monkeypatch.setattr(runner, "_canonical_sha256", lambda *args: "fixed")
    monkeypatch.setattr(
        runner,
        "read",
        lambda *args: {
            "rows": [],
            "provenance": {"base_manifest_canonical_sha256": "fixed"},
        },
    )
    monkeypatch.setattr(runner, "replay_schedule", lambda *args: [])
    seen = {}

    def source(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("STOP_BEFORE_PRELOAD")

    monkeypatch.setattr(runner, "PreloadedRawSource", source)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: pytest.fail("CUDA touched"))
    with pytest.raises(RuntimeError, match="STOP_BEFORE_PRELOAD"):
        runner.run(args)
    assert seen["max_total_bytes"] == args.max_preloaded_source_bytes - 33 * CROP_BYTES
