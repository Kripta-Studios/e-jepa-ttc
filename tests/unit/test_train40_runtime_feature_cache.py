"""CPU checks for causal identity, mutation safety and bounded RGB reuse."""

from __future__ import annotations

import dataclasses
import threading

import numpy as np
import pytest
import torch

from operational.train40_runtime.feature_cache import (
    ExactRGBFeatureCache,
    bind_rgb_history,
    rgb_feature_key,
)


def raw_history():
    frames = {index: np.full((4, 5, 3), index, dtype=np.uint8) for index in range(7)}
    for frame in frames.values():
        frame.flags.writeable = False
    clocks = {index: (index + 1) * 100_000 for index in frames}
    groups = tuple((index, index + 1, index + 2) for index in range(5))
    return frames, clocks, groups


def identity(*, query=700_000, readonly=True, **kwargs):
    frames, clocks, groups = raw_history()
    for frame in frames.values():
        frame.flags.writeable = not readonly
    return bind_rgb_history(frames, clocks, groups, query_us=query, **kwargs)


def key(object_id="car", *, history=None, slot=0, objects=1, roi=(0.0, 0.0, 4.0, 5.0)):
    return rgb_feature_key(
        object_id,
        roi,
        history or identity(),
        model_sha256="a" * 64,
        runtime_sha256="b" * 64,
        batch_shape=(5 * objects, 3, 3, 128, 128),
        object_slot=slot,
    )


def tensors(objects=1):
    return torch.arange(objects * 85).reshape(objects, 5, 17).float(), torch.ones(objects, 5, 3)


def test_content_identity_is_shared_and_ignores_only_current_context_time():
    first, second = identity(), identity(query=720_000)
    assert first.digest == second.digest
    assert first.hashed_bytes == first.raw_bytes == 7 * 4 * 5 * 3
    assert key(history=first).digest == key(history=second).digest
    assert first.fingerprint_seconds >= 0


@pytest.mark.parametrize("change", ["pixels", "timestamp", "frame_id", "history", "shape"])
def test_any_producer_input_change_invalidates_shared_identity(change):
    frames, clocks, groups = raw_history()
    before = bind_rgb_history(frames, clocks, groups, query_us=700_000)
    if change == "pixels":
        frames[0].flags.writeable = True
        frames[0][0, 0, 0] = 111
        frames[0].flags.writeable = False
    elif change == "timestamp":
        clocks[2] += 1
    elif change == "frame_id":
        frames[8], clocks[8] = frames.pop(0), clocks.pop(0)
        groups = ((8, 1, 2), *groups[1:])
    elif change == "history":
        groups = groups[1:]
    else:
        frames[0] = frames[0].reshape(5, 4, 3)
    after = bind_rgb_history(frames, clocks, groups, query_us=700_000)
    assert before.digest != after.digest


def test_stat_mode_checks_source_and_requires_readonly_reader_frames(tmp_path):
    path = tmp_path / "source.h5"
    path.write_bytes(b"sealed fixture bytes")
    stat = path.stat()
    expected = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    with pytest.raises(ValueError, match="read-only"):
        identity(readonly=False, source_path=path, expected_source_stat=expected)
    bound = identity(readonly=True, source_path=path, expected_source_stat=expected)
    assert bound.source_binding[0] == "READONLY_READER_STAT"
    assert bound.hashed_bytes == 0
    assert bound.digest != identity(readonly=True).digest
    path.write_bytes(b"changed and longer source")
    with pytest.raises(ValueError, match="source stat"):
        identity(readonly=True, source_path=path, expected_source_stat=expected)


def test_incomplete_stat_attestation_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="both"):
        identity(source_path=tmp_path / "source.h5")


@pytest.mark.parametrize("query", [699_999, 750_001])
def test_future_or_overage_frames_cannot_be_bound(query):
    with pytest.raises(ValueError, match="causal support"):
        identity(query=query)


def test_frame_memory_bound_precedes_hashing_unbounded_history():
    with pytest.raises(ValueError, match="byte bound"):
        identity(max_raw_bytes=100)


def test_source_frame_mutation_is_detected_without_object_address_caching():
    frames, clocks, groups = raw_history()
    first = bind_rgb_history(frames, clocks, groups, query_us=700_000)
    frames[1].flags.writeable = True
    frames[1][:] = 200
    frames[1].flags.writeable = False
    second = bind_rgb_history(frames, clocks, groups, query_us=700_000)
    assert first.digest != second.digest


def test_content_mode_rejects_writable_sensor_reuse_buffers():
    with pytest.raises(ValueError, match="read-only"):
        identity(readonly=False)


@pytest.mark.parametrize("change", ["roi", "model", "runtime", "shape", "position", "object"])
def test_complete_key_changes_invalidate_cached_object(change):
    cache = ExactRGBFeatureCache()
    original = key(objects=2)
    raw, experts = tensors()
    cache.store_many([original], raw, experts, guard_passed=True)
    proposed = original
    if change == "roi":
        proposed = key(objects=2, roi=(0.01, 0.0, 4.0, 5.0))
    elif change == "model":
        proposed = dataclasses.replace(original, model_sha256="c" * 64)
    elif change == "runtime":
        proposed = dataclasses.replace(original, runtime_sha256="c" * 64)
    elif change == "shape":
        proposed = key(objects=1)
    elif change == "position":
        proposed = key(objects=2, slot=1)
    else:
        proposed = key("pedestrian", objects=2)
    assert cache.lookup_many([proposed]) is None
    assert cache.stats["batch_misses"] == 1


def test_owned_copies_prevent_producer_or_consumer_mutation_from_poisoning_cache():
    cache = ExactRGBFeatureCache()
    entry = key()
    raw, experts = tensors()
    expected_raw, expected_experts = raw.clone(), experts.clone()
    assert cache.store_many([entry], raw, experts, guard_passed=True)
    raw.fill_(99)
    experts.fill_(88)
    first = cache.lookup_many([entry])
    assert first is not None
    torch.testing.assert_close(first[0].raw, expected_raw[0], rtol=0, atol=0)
    torch.testing.assert_close(first[0].experts, expected_experts[0], rtol=0, atol=0)
    first[0].raw.fill_(-1)
    second = cache.lookup_many([entry])
    assert second is not None
    torch.testing.assert_close(second[0].raw, expected_raw[0], rtol=0, atol=0)
    assert cache.stats["served_object_hits"] == 2
    assert cache.stats["tensor_bytes"] == 5 * 20 * 4


def test_partial_key_match_cannot_claim_an_avoided_rgb_batch():
    cache = ExactRGBFeatureCache()
    keys = [key("a", objects=2), key("b", objects=2, slot=1)]
    cache.store_many(keys, *tensors(2), guard_passed=True)
    different = key("b", objects=2, slot=1, roi=(1.0, 1.0, 4.0, 5.0))
    assert cache.lookup_many([keys[0], different]) is None
    assert cache.stats["key_matches"] == 1
    assert cache.stats.get("served_object_hits", 0) == 0
    assert cache.stats["batch_misses"] == 1
    assert cache.stats["invalidations"] == 1


def test_twenty_objects_remain_distinct_and_hits_follow_request_order():
    cache = ExactRGBFeatureCache()
    common = identity()
    keys = [key(f"object-{i}", history=common, objects=20, slot=i) for i in range(20)]
    raw, experts = tensors(20)
    assert cache.store_many(keys, raw, experts, guard_passed=True)
    result = cache.lookup_many(list(reversed(keys)))
    assert result is not None
    for index, item in enumerate(result):
        torch.testing.assert_close(item.raw, raw[19 - index], rtol=0, atol=0)
    assert cache.stats["entries"] == 20
    assert cache.stats["tensor_bytes"] == 8000


def test_lru_and_byte_limits_bound_backing_storage():
    cache = ExactRGBFeatureCache(max_entries=2, max_bytes=800)
    a, b, c = key("a"), key("b"), key("c")
    for entry in (a, b):
        assert cache.store_many([entry], *tensors(), guard_passed=True)
    assert cache.lookup_many([a]) is not None
    assert cache.store_many([c], *tensors(), guard_passed=True)
    assert cache.lookup_many([b]) is None
    assert cache.stats["evictions"] == 1
    assert cache.stats["tensor_bytes"] == 800
    cache.retain_objects(["c"])
    assert cache.stats["entries"] == 1
    assert cache.stats["tensor_bytes"] == 400
    cache.clear()
    assert cache.stats["tensor_bytes"] == 0


def test_oversized_lot_is_rejected_without_partial_eviction():
    cache = ExactRGBFeatureCache(max_entries=2, max_bytes=400)
    old = key("old")
    cache.store_many([old], *tensors(), guard_passed=True)
    keys = [key("a", objects=2), key("b", objects=2, slot=1)]
    assert not cache.store_many(keys, *tensors(2), guard_passed=True)
    assert cache.lookup_many([old]) is not None
    assert cache.stats["entries"] == 1


@pytest.mark.parametrize("damage", ["guard", "nan", "dtype", "grad", "shape"])
def test_invalid_features_never_enter_the_cache(damage):
    cache = ExactRGBFeatureCache()
    raw, experts = tensors()
    guard = True
    if damage == "guard":
        guard = False
    elif damage == "nan":
        experts[0, 0, 0] = float("nan")
    elif damage == "dtype":
        raw = raw.double()
    elif damage == "grad":
        raw.requires_grad_(True)
    else:
        raw = raw[..., :16]
    with pytest.raises(ValueError):
        cache.store_many([key()], raw, experts, guard_passed=guard)
    assert cache.stats["entries"] == 0


def test_cache_is_owned_by_one_thread():
    cache = ExactRGBFeatureCache()
    errors = []

    def wrong_thread():
        try:
            cache.lookup_many([key()])
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=wrong_thread)
    worker.start()
    worker.join()
    assert errors == ["RGB feature cache has a single owning thread"]


@pytest.mark.parametrize("override", [{"model_sha256": "weak"}, {"precision": "BF16"}])
def test_weak_model_or_precision_binding_is_rejected(override):
    kwargs = {
        "model_sha256": "a" * 64,
        "runtime_sha256": "b" * 64,
        "batch_shape": (5, 3, 3, 128, 128),
        "object_slot": 0,
        **override,
    }
    with pytest.raises(ValueError):
        rgb_feature_key("car", (0, 0, 4, 5), identity(), **kwargs)
