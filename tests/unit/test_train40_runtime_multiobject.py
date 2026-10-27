"""CPU contracts for shared raw updates and isolated object histories."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from e_jepa_ttc.rgb_port.data import crop_uint8
from operational.train40_runtime.multiobject import (
    MultiObjectTrain40Service,
    ObjectRegion,
    PreparedRGBBatch,
    RawRGBHistory,
    _effective_device,
    _validate_producer_device,
    _validate_regions,
    _validate_rgb,
    rgb_metadata,
)
from operational.train40_runtime.multiobject_cache import RGBBatchCache
from operational.train40_runtime.rgb_uint8 import RGBUInt8Runtime


def regions():
    return [
        ObjectRegion("a", (0.0, 0.0, 20.0, 20.0), (0.0, 0.0, 20.0, 20.0)),
        ObjectRegion("b", (10.0, 10.0, 30.0, 30.0), (10.0, 10.0, 30.0, 30.0)),
    ]


def raw_rgb(query=1_000_000):
    frame = np.tile(np.arange(32, dtype=np.uint8)[None, :, None], (32, 1, 3))
    frames = {i: frame + i for i in range(7)}
    for value in frames.values():
        value.setflags(write=False)
    return RawRGBHistory(
        query,
        frames,
        {i: query - (6 - i) * 100_000 for i in range(7)},
        [(i, i + 1, i + 2) for i in range(5)],
    )


class FakeProducers:
    def __init__(self):
        self.calls = []
        self.device = torch.device("cpu")
        self.block_graphs = False
        self.reuse_by_modality = {"E": False, "R": True}
        self.models = {}

    def expert_block(self, modality, sensor, delta, *, history_group_size):
        self.calls.append((modality, tuple(sensor.shape), history_group_size))
        values = torch.arange(len(sensor), dtype=torch.float32)[:, None]
        return values.expand(-1, 17), values.expand(-1, 3)


class FakeContext:
    def __init__(self):
        self.observations = []

    def predict_many(self, observations):
        self.observations = observations
        return [{"object_id": item.object_id, "predictions": {"TTC": 1.0}} for item in observations]


class FakeHistory:
    def __init__(self):
        self.end = 980_000
        self.ingested = []
        self.rois = []

    def ingest(self, raw, start, end):
        if start != self.end or end - start != 20_000:
            raise ValueError("nonconsecutive packet")
        self.end = end
        self.ingested.append((start, end, raw))

    def query_many(self, rois):
        self.rois.append(rois)
        return torch.zeros((len(rois), 4, 3, 12, 128, 128))


@pytest.fixture
def service():
    value = MultiObjectTrain40Service.__new__(MultiObjectTrain40Service)
    value.device = torch.device("cpu")
    value.max_objects = 20
    value.producers = FakeProducers()
    value.context = FakeContext()
    value.history = FakeHistory()
    value.parallel_branches = False
    value.shared_rgb_images = False
    value.rgb_object_chunk_size = None
    value.parallel_preprocessing = True
    value.rgb_feature_cache = None
    value.last_rgb_profile = {}
    value._closed = False
    value.retain_input_diagnostics = False
    value.last_rgb_inputs = None
    value.last_rgb_prepared_inputs = None
    value.last_rgb_input_delta = None
    value._diagnostic_rgb_lease = None
    value._rgb_uint8 = None
    value._crops = ThreadPoolExecutor(max_workers=2)
    value._preprocess = ThreadPoolExecutor(max_workers=1)
    yield value
    value.close()


def test_shared_update_ingests_once_and_keeps_object_boundaries(service):
    objects = regions()
    output = service.update({}, 980_000, 1_000_000, objects, raw_rgb())
    assert output["object_count"] == 2
    assert output["completed_ns"] >= output["started_ns"]
    assert output["complete_wall_ms"] > 0
    assert len(service.history.ingested) == 1
    assert service.history.rois == [[item.event_square for item in objects]]
    assert service.producers.calls == [
        ("E", (8, 3, 12, 128, 128), 4),
        ("R", (10, 3, 3, 128, 128), 5),
    ]
    a, b = service.context.observations
    assert a.object_id == "a" and b.object_id == "b"
    assert a.event_raw[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert b.event_raw[:, 0].tolist() == [4.0, 5.0, 6.0, 7.0]
    assert a.rgb_raw[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert b.rgb_raw[:, 0].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_roi_is_applied_to_raw_frames_independently(service):
    output = service._rgb(regions(), raw_rgb())
    assert len(output) == 2 and all(len(rows) == 5 for rows in output)
    assert not np.array_equal(output[0][0][0], output[1][0][0])
    np.testing.assert_array_equal(output[0][0][0][1:], output[0][1][0][:-1])


def test_object_birth_disappearance_and_missing_rgb_do_not_reuse_context(service):
    first = regions()[0]
    service.update({}, 980_000, 1_000_000, [first], raw_rgb())
    second = ObjectRegion("new", first.event_square, None)
    output = service.update({}, 1_000_000, 1_020_000, [second], raw_rgb(1_020_000))
    assert [row["object_id"] for row in output["objects"]] == ["new"]
    assert output["objects"][0]["rgb_unavailable_reason"] == "RGB_ROI_UNAVAILABLE"
    assert service.context.observations[0].rgb_raw is None


def test_gap_or_rollback_rejected(service):
    with pytest.raises(ValueError, match="nonconsecutive"):
        service.update({}, 960_000, 980_000, regions(), raw_rgb(980_000))


@pytest.mark.parametrize("case", ["duplicate", "empty", "nan", "negative"])
def test_invalid_regions_rejected(case):
    items = regions()
    if case == "duplicate":
        items[1] = items[0]
    elif case == "empty":
        items = []
    else:
        items[0] = ObjectRegion(
            "a", (float("nan"), 0.0, 1.0, 1.0) if case == "nan" else (2.0, 0.0, 1.0, 1.0), None
        )
    with pytest.raises(ValueError):
        _validate_regions(items, 20)


def test_rgb_clock_identity_and_causality_rejected():
    history = raw_rgb()
    with pytest.raises(ValueError, match="different update"):
        _validate_rgb(history, history.query_us + 1)
    history.timestamps_us[6] += 1
    with pytest.raises(ValueError, match="causal"):
        _validate_rgb(history, history.query_us)


def test_native_two_frame_startup_history_supported():
    history = raw_rgb()
    partial = RawRGBHistory(history.query_us, history.frames, history.timestamps_us, [(5, 6)])
    _validate_rgb(partial, partial.query_us)


def test_shared_pil_frames_match_original_for_twenty_regions(service):
    generator = np.random.default_rng(4)
    frames = {i: generator.integers(0, 256, (128, 160, 3), dtype=np.uint8) for i in range(7)}
    for frame in frames.values():
        frame.setflags(write=False)
    history = RawRGBHistory(
        1_000_000,
        frames,
        {i: 400_000 + i * 100_000 for i in range(7)},
        [(i, i + 1, i + 2) for i in range(5)],
    )
    objects = [
        ObjectRegion(
            str(i), (0.0, 0.0, 10.0, 10.0), (-5.2 + i * 2, 1.1 + i, 40.8 + i * 2, 65.4 + i)
        )
        for i in range(20)
    ]
    service.shared_rgb_images = True
    proposed = service._rgb(objects, history)
    assert service.last_rgb_profile["pil_images_created"] == 7
    assert service.last_rgb_profile["roi_frame_tasks"] == 140
    for obj, observations in zip(objects, proposed, strict=True):
        for group, (actual, _) in zip(history.triplets, observations, strict=True):
            expected = np.ascontiguousarray(
                np.stack(
                    [crop_uint8(history.frames[index], obj.rgb_square) for index in group]
                ).astype(np.float32)
                / 255
            )
            np.testing.assert_array_equal(actual, expected)


def test_rgb_chunks_keep_each_object_history_and_feature_order(service):
    service.rgb_object_chunk_size = 1
    output = service.update({}, 980_000, 1_000_000, regions(), raw_rgb())
    assert output["object_count"] == 2
    assert service.producers.calls == [
        ("E", (8, 3, 12, 128, 128), 4),
        ("R", (5, 3, 3, 128, 128), 5),
        ("R", (5, 3, 3, 128, 128), 5),
    ]
    assert [item.object_id for item in service.context.observations] == ["a", "b"]
    for observation in service.context.observations:
        assert observation.rgb_raw[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_shared_pil_invalid_region_drains_tasks_and_raises(service):
    service.shared_rgb_images = True
    objects = regions() + [ObjectRegion("bad", (0.0, 0.0, 1.0, 1.0), (100.0, 100.0, 110.0, 110.0))]
    with pytest.raises(ValueError, match="empty"):
        service._rgb(objects, raw_rgb())


def test_serial_preprocessing_remains_inside_complete_clock(service):
    service.parallel_preprocessing = False
    result = service.update({}, 980_000, 1_000_000, regions(), raw_rgb())
    job = result["rgb_preprocess"]
    assert result["started_ns"] <= job["started_ns"] <= job["completed_ns"]
    assert job["completed_ns"] <= result["completed_ns"]
    assert result["event_enqueue_host_ms"] > result["event_dispatch_host_ms"]
    assert result["complete_wall_ms"] >= job["total_seconds"] * 1000


def enable_cache(service):
    service.rgb_feature_cache = RGBBatchCache(
        model_sha256="a" * 64,
        runtime_sha256="b" * 64,
        rgb_object_chunk_size=service.rgb_object_chunk_size,
    )
    history = raw_rgb()
    for frame in history.frames.values():
        frame.setflags(write=False)
    return history


def test_exact_cache_hit_skips_pixels_and_rgb_but_rebuilds_current_context(service, monkeypatch):
    history = enable_cache(service)
    service.retain_input_diagnostics = True
    first = service.update({}, 980_000, 1_000_000, regions(), history)
    assert first["rgb_feature_cache"]["status"] == "MISS"
    expected_features = [item.rgb_raw.clone() for item in service.context.observations]
    service.producers.calls.clear()

    def forbid_crop(*args):
        raise AssertionError("A complete feature hit must not prepare pixels")

    monkeypatch.setattr(service, "_rgb", forbid_crop)
    second = service.update(
        {}, 1_000_000, 1_020_000, regions(), replace(history, query_us=1_020_000)
    )
    assert second["rgb_feature_cache"]["status"] == "HIT"
    assert second["rgb_preprocess"]["roi_frame_tasks"] == 0
    assert service.last_rgb_inputs is None
    assert service.producers.calls == [("E", (8, 3, 12, 128, 128), 4)]
    for observation, expected in zip(service.context.observations, expected_features, strict=True):
        assert observation.query_us == 1_020_000
        assert all(meta["available_us"] == 1_020_000 for meta in observation.rgb_metadata)
        torch.testing.assert_close(observation.rgb_raw, expected, rtol=0, atol=0)


def test_cache_changed_roi_recomputes_whole_same_batch(service):
    history = enable_cache(service)
    service.update({}, 980_000, 1_000_000, regions(), history)
    objects = regions()
    objects[1] = replace(objects[1], rgb_square=(11.0, 11.0, 31.0, 31.0))
    service.producers.calls.clear()
    result = service.update({}, 1_000_000, 1_020_000, objects, replace(history, query_us=1_020_000))
    assert result["rgb_feature_cache"]["status"] == "MISS"
    assert service.producers.calls[-1] == ("R", (10, 3, 3, 128, 128), 5)


def test_cache_rgb_absence_invalidates_and_event_fallback_remains(service):
    history = enable_cache(service)
    service.update({}, 980_000, 1_000_000, regions(), history)
    objects = [replace(item, rgb_square=None) for item in regions()]
    result = service.update({}, 1_000_000, 1_020_000, objects, replace(history, query_us=1_020_000))
    assert result["rgb_feature_cache"]["status"] == "NO_RGB"
    assert all(observation.rgb_raw is None for observation in service.context.observations)
    assert service.rgb_feature_cache.stats["entries"] == 0


def test_prepared_pixel_tensor_preserves_object_metadata_and_chunk_shape(service):
    history = raw_rgb()
    rgb = service._rgb(regions(), history)
    pixels = torch.stack([torch.from_numpy(pixels) for rows in rgb for pixels, _ in rows])
    prepared = PreparedRGBBatch(rgb_metadata(regions(), history), pixels=pixels)
    service.rgb_object_chunk_size = 1
    service.infer_prepared(
        regions(), history.query_us, torch.zeros((2, 4, 3, 12, 128, 128)), prepared
    )
    assert service.producers.calls == [
        ("E", (8, 3, 12, 128, 128), 4),
        ("R", (5, 3, 3, 128, 128), 5),
        ("R", (5, 3, 3, 128, 128), 5),
    ]
    assert [item.rgb_metadata for item in service.context.observations] == prepared.metadata


def test_cache_chunk_identity_must_match_actual_producer_configuration():
    cache = RGBBatchCache(model_sha256="a" * 64, runtime_sha256="b" * 64, rgb_object_chunk_size=1)
    with pytest.raises(ValueError, match="chunk shape"):
        MultiObjectTrain40Service(
            {},
            {},
            FakeHistory(),
            device="cpu",
            graphs=False,
            parallel_branches=False,
            rgb_object_chunk_size=None,
            rgb_feature_cache=cache,
        )


@pytest.mark.parametrize("count", [1, 2])
def test_uint8_full_update_matches_original_pixels_delta_and_lease(service, count):
    from scripts.benchmark_train40_multiobject import input_digests, reference_rgb

    service._rgb_uint8 = RGBUInt8Runtime(
        "cpu", max_objects=count, crop_workers=2, shared_images=False
    )
    service.retain_input_diagnostics = True
    objects = regions()[:count]
    history = raw_rgb()
    result = service.update({}, 980_000, 1_000_000, objects, history)
    assert result["rgb_uint8"] and result["rgb_diagnostic_lease_retained"]
    lease = service._diagnostic_rgb_lease
    lease.assert_valid()
    assert service.last_rgb_inputs is None
    assert len(input_digests(service)) == count
    for position, obj in enumerate(objects):
        expected = reference_rgb(history, obj)
        actual = service.last_rgb_prepared_inputs.pixels[position * 5 : (position + 1) * 5]
        np.testing.assert_array_equal(actual.numpy(), np.stack([pixels for pixels, _ in expected]))
        np.testing.assert_array_equal(
            service.last_rgb_input_delta[position * 5 : (position + 1) * 5].numpy(),
            np.asarray([meta["delta_t_s"] for _, meta in expected], dtype=np.float32),
        )
    service.update({}, 1_000_000, 1_020_000, objects, raw_rgb(1_020_000))
    with pytest.raises(ValueError, match="released"):
        lease.assert_valid()


def test_uint8_gap_discards_finished_cpu_pack_and_recovers(service):
    service._rgb_uint8 = RGBUInt8Runtime("cpu", max_objects=2)
    with pytest.raises(ValueError, match="nonconsecutive"):
        service.update({}, 960_000, 980_000, regions(), raw_rgb(980_000))
    result = service.update({}, 980_000, 1_000_000, regions(), raw_rgb())
    assert result["object_count"] == 2
    assert not result["rgb_diagnostic_lease_retained"]


def test_uint8_context_failure_releases_borrowed_buffers(service, monkeypatch):
    service._rgb_uint8 = RGBUInt8Runtime("cpu", max_objects=2)
    original = service.context.predict_many

    def fail_context(*args):
        raise RuntimeError("context failure")

    monkeypatch.setattr(service.context, "predict_many", fail_context)
    with pytest.raises(RuntimeError, match="context failure"):
        service.update({}, 980_000, 1_000_000, regions(), raw_rgb())
    monkeypatch.setattr(service.context, "predict_many", original)
    assert (
        service.update({}, 1_000_000, 1_020_000, regions(), raw_rgb(1_020_000))["object_count"] == 2
    )


def test_uint8_cache_hit_has_no_pixel_lease_and_fresh_context(service):
    service._rgb_uint8 = RGBUInt8Runtime("cpu", max_objects=2)
    service.retain_input_diagnostics = True
    history = enable_cache(service)
    service.update({}, 980_000, 1_000_000, regions(), history)
    old_lease = service._diagnostic_rgb_lease
    result = service.update(
        {}, 1_000_000, 1_020_000, regions(), replace(history, query_us=1_020_000)
    )
    assert result["rgb_feature_cache"]["status"] == "HIT"
    assert service.last_rgb_prepared_inputs.pixels is None
    assert service._diagnostic_rgb_lease is None
    assert all(item.query_us == 1_020_000 for item in service.context.observations)
    with pytest.raises(ValueError, match="released"):
        old_lease.assert_valid()


def test_new_service_invalidates_cache_from_previous_producer_lifetime(service, monkeypatch):
    from operational.train40_runtime import multiobject

    history = enable_cache(service)
    service.update({}, 980_000, 1_000_000, regions(), history)
    cache = service.rgb_feature_cache
    assert cache.stats["entries"] == 2
    monkeypatch.setattr(multiobject, "BatchedContextRuntime", lambda *args, **kwargs: FakeContext())
    monkeypatch.setattr(multiobject, "ProducerRuntime", lambda *args, **kwargs: FakeProducers())
    fresh = MultiObjectTrain40Service(
        {},
        {},
        FakeHistory(),
        device="cpu",
        graphs=False,
        parallel_branches=False,
        rgb_feature_cache=cache,
    )
    try:
        assert cache.stats["entries"] == 0
    finally:
        fresh.close()


def test_external_producer_and_cache_cannot_claim_unbound_shared_identity():
    cache = RGBBatchCache(model_sha256="a" * 64, runtime_sha256="b" * 64)
    with pytest.raises(ValueError, match="separately bound backend"):
        MultiObjectTrain40Service(
            {},
            {},
            FakeHistory(),
            device="cpu",
            graphs=False,
            parallel_branches=False,
            producer_runtime=FakeProducers(),
            rgb_feature_cache=cache,
        )


def test_external_producer_can_be_injected_without_feature_cache(monkeypatch):
    from operational.train40_runtime import multiobject

    monkeypatch.setattr(multiobject, "BatchedContextRuntime", lambda *args, **kwargs: FakeContext())
    injected = FakeProducers()
    fresh = MultiObjectTrain40Service(
        {},
        {},
        FakeHistory(),
        device="cpu",
        graphs=False,
        parallel_branches=False,
        producer_runtime=injected,
    )
    try:
        assert fresh.producers is injected
    finally:
        fresh.close()


@pytest.mark.parametrize(
    "runtime_device,current,model_device",
    [
        ("cuda:1", 0, "cuda:1"),
        ("cuda", 1, "cuda:1"),
        ("cuda", 0, "cuda:1"),
    ],
)
def test_injected_cuda_device_rejects_implicit_and_resident_mismatch_without_gpu(
    monkeypatch, runtime_device, current, model_device
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)
    tensor = SimpleNamespace(device=torch.device(model_device))
    model = SimpleNamespace(parameters=lambda: iter([tensor]), buffers=lambda: iter([]))
    runtime = SimpleNamespace(device=torch.device(runtime_device), models={"model": model})
    with pytest.raises(ValueError, match="service device"):
        _validate_producer_device(runtime, torch.device("cuda:0"))
    assert _effective_device("cuda") == torch.device("cuda", current)
