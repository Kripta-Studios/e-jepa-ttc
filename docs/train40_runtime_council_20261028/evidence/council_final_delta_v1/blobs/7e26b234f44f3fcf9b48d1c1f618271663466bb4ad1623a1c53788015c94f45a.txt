"""Exact original PIL pixel reuse, causal keys, bounded bytes and worker ownership."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from operational.train40_runtime import rgb_raw_ring
from operational.train40_runtime.inputs import _normalized_crop
from operational.train40_runtime.rgb_crop_cache import CROP_BYTES, CROP_SHAPE, ExactRGBCropCache
from operational.train40_runtime.rgb_raw_ring import RGBSourceIdentity, freeze_rgb_frame

ROI = (-5.25, 2.125, 49.5, 56.875)


def frame(index=0, *, stamp=None, origin=None, pixels=None):
    return freeze_rgb_frame(
        np.random.default_rng(index + 7).integers(0, 256, (61, 47, 3), dtype=np.uint8)
        if pixels is None
        else pixels,
        index,
        100_000 + index * 50_000 if stamp is None else stamp,
        origin or RGBSourceIdentity("sensor", "a" * 64),
    )


def store(cache, token, *, roi=ROI, object_id="car", query=None):
    pixels = _normalized_crop(token.array, roi)
    assert cache.put(
        object_id,
        token,
        roi,
        pixels,
        query_us=token.timestamp_us if query is None else query,
    )
    return pixels


def test_original_pil_output_roundtrips_bitexact_and_cannot_be_mutated():
    cache = ExactRGBCropCache(max_objects=1)
    token = frame()
    assert cache.get("car", token, ROI, query_us=100_000) is None
    expected = store(cache, token)
    frozen = cache.get("car", token, ROI, query_us=120_000)
    np.testing.assert_array_equal(frozen, expected)
    assert frozen.dtype == np.float32 and frozen.shape == CROP_SHAPE
    assert not np.shares_memory(frozen, expected)
    expected[:] = 0
    assert frozen.any()
    frozen.shape = (128, 128, 3)
    assert cache.get("car", token, ROI, query_us=140_000).shape == CROP_SHAPE
    with pytest.raises(ValueError):
        frozen.setflags(write=True)
    cache.close()
    assert frozen.any() and cache.stats["crop_bytes"] == 0
    assert cache.stats["raw_payload_bytes_hashed"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        cache.get("car", token, ROI, query_us=140_000)


@pytest.mark.parametrize("changed", ["roi", "object", "source", "pixels", "frame_id", "timestamp"])
def test_every_input_identity_change_misses(changed):
    cache = ExactRGBCropCache()
    token = frame()
    store(cache, token)
    roi, object_id = ROI, "car"
    if changed == "roi":
        roi = (ROI[0] + 1e-7, *ROI[1:])
    elif changed == "object":
        object_id = "pedestrian"
    else:
        pixels = token.array.copy()
        if changed == "pixels":
            pixels[0, 0, 0] ^= 1
        token = frame(
            int(changed == "frame_id"),
            stamp=100_000 + int(changed == "timestamp"),
            pixels=pixels,
            origin=RGBSourceIdentity("other", "b" * 64) if changed == "source" else token.source,
        )
    assert cache.get(object_id, token, roi, query_us=120_000) is None


def test_equivalent_factory_tokens_hit_without_any_raw_rescan(monkeypatch):
    token = frame()
    equivalent = frame(pixels=token.array.copy())
    cache = ExactRGBCropCache()
    expected = store(cache, token)
    monkeypatch.setattr(rgb_raw_ring.hashlib, "sha256", lambda *args: pytest.fail("RAW rehash"))
    np.testing.assert_array_equal(cache.get("car", equivalent, ROI, query_us=110_000), expected)


@pytest.mark.parametrize("query", [99_999, 750_001, -1, True])
def test_causality_checked_even_for_existing_entries(query):
    cache = ExactRGBCropCache()
    token = frame()
    store(cache, token)
    with pytest.raises(ValueError):
        cache.get("car", token, ROI, query_us=query)
    with pytest.raises(ValueError):
        cache.put("car", token, ROI, np.zeros(CROP_SHAPE, np.float32), query_us=query)


@pytest.mark.parametrize(
    "invalid", ["raw_array", "forged", "roi_nan", "roi_bool", "roi_empty", "id_long"]
)
def test_unattested_tokens_or_invalid_roi_identity_are_rejected(invalid):
    cache = ExactRGBCropCache()
    token = frame()
    roi, object_id = ROI, "car"
    if invalid == "raw_array":
        token = token.array
    elif invalid == "forged":
        token = object.__new__(rgb_raw_ring.FrozenRGBFrame)
    elif invalid == "roi_nan":
        roi = (float("nan"), *ROI[1:])
    elif invalid == "roi_bool":
        roi = (True, *ROI[1:])
    elif invalid == "roi_empty":
        roi = (1, 1, 1, 2)
    else:
        object_id = "x" * 257
    with pytest.raises(ValueError):
        cache.get(object_id, token, roi, query_us=100_000)


def test_lru_respects_per_object_capacity_and_returned_views_survive_eviction():
    cache = ExactRGBCropCache(max_objects=2, max_crops_per_object=2)
    a, b, c = frame(0), frame(1), frame(2)
    first = store(cache, a)
    store(cache, b)
    retained = cache.get("car", a, ROI, query_us=200_000)
    store(cache, c)
    assert cache.get("car", b, ROI, query_us=200_000) is None
    assert cache.get("car", a, ROI, query_us=200_000) is not None
    store(cache, b, object_id="pedestrian")
    assert cache.stats["entries"] == 3 and cache.stats["objects"] == 2
    assert cache.stats["crop_bytes"] == 3 * CROP_BYTES
    cache.clear()
    np.testing.assert_array_equal(retained, first)
    assert cache.stats["entries"] == cache.stats["crop_bytes"] == 0


def test_global_byte_bound_and_object_bound_evict_without_reusing_other_identity():
    cache = ExactRGBCropCache(max_objects=2, max_bytes=2 * CROP_BYTES)
    a, b, c = frame(0), frame(1), frame(2)
    store(cache, a, object_id="a")
    store(cache, b, object_id="b")
    store(cache, c, object_id="b")
    assert cache.stats["crop_bytes"] == 2 * CROP_BYTES
    assert cache.get("a", a, ROI, query_us=200_000) is None
    store(cache, a, object_id="c")
    assert cache.stats["objects"] <= 2 and cache.stats["crop_bytes"] <= 2 * CROP_BYTES
    narrow = ExactRGBCropCache(max_objects=1)
    store(narrow, a, object_id="a")
    borrowed = narrow.get("a", a, ROI, query_us=200_000)
    store(narrow, b, object_id="b")
    assert narrow.stats["objects"] == 1
    assert narrow.get("a", a, ROI, query_us=200_000) is None and borrowed.any()


def test_oversized_put_does_not_allocate_or_retain_an_entry():
    cache = ExactRGBCropCache(max_bytes=CROP_BYTES - 1)
    token = frame()
    assert not cache.put("car", token, ROI, _normalized_crop(token.array, ROI), query_us=100_000)
    assert cache.stats["oversized_rejections"] == 1 and cache.stats["crop_bytes"] == 0


@pytest.mark.parametrize("invalid", ["shape", "dtype", "nan", "low", "high"])
def test_bad_crop_never_replaces_a_valid_entry(invalid):
    cache = ExactRGBCropCache()
    token = frame()
    expected = store(cache, token)
    wrong = expected.copy()
    if invalid == "shape":
        wrong = wrong[:, :-1]
    elif invalid == "dtype":
        wrong = wrong.astype(np.float64)
    else:
        wrong.flat[0] = {"nan": float("nan"), "low": -1, "high": 1.01}[invalid]
    with pytest.raises(ValueError):
        cache.put("car", token, ROI, wrong, query_us=100_000)
    np.testing.assert_array_equal(cache.get("car", token, ROI, query_us=100_000), expected)


@pytest.mark.parametrize("count", [1, 2])
def test_both_twenty_hz_cohorts_new_frame_and_history_subset_keep_original_pixels(count):
    cache = ExactRGBCropCache(max_objects=count)
    tokens = [frame(index) for index in range(15)]
    updates = [
        tuple(range(0, 13, 2)),
        tuple(range(1, 14, 2)),
        tuple(range(3, 14, 2)),
        tuple(range(2, 15, 2)),
    ]
    misses = []
    for ids in updates:
        query = max(tokens[index].timestamp_us for index in ids)
        update_misses = 0
        for obj in range(count):
            roi = (ROI[0] + obj, ROI[1], ROI[2] + obj, ROI[3])
            crops = {}
            for index in ids:
                actual = cache.get(str(obj), tokens[index], roi, query_us=query)
                if actual is None:
                    update_misses += 1
                    actual = store(cache, tokens[index], roi=roi, object_id=str(obj), query=query)
                crops[index] = actual
                np.testing.assert_array_equal(actual, _normalized_crop(tokens[index].array, roi))
            actual_rows = np.stack(
                [
                    np.stack([crops[index] for index in ids[pos : pos + 3]])
                    for pos in range(len(ids) - 2)
                ]
            )
            expected_rows = np.stack(
                [
                    np.stack(
                        [_normalized_crop(tokens[index].array, roi) for index in ids[pos : pos + 3]]
                    )
                    for pos in range(len(ids) - 2)
                ]
            )
            np.testing.assert_array_equal(actual_rows, expected_rows)
        misses.append(update_misses)
    assert misses == [7 * count, 7 * count, 0, count]
    assert cache.stats["entries"] == 15 * count
    assert cache.stats["crop_bytes"] <= count * 16 * CROP_BYTES


def test_lazy_coordinator_ownership_does_not_bind_construction_to_service_thread():
    cache = ExactRGBCropCache(max_objects=1)
    token = frame()
    result = []
    ready, release = threading.Event(), threading.Event()

    def coordinate():
        assert cache.get("car", token, ROI, query_us=100_000) is None
        with ThreadPoolExecutor(max_workers=2) as workers:
            crop = workers.submit(_normalized_crop, token.array, ROI).result()
        assert cache.put("car", token, ROI, crop, query_us=100_000)
        result.append(cache.get("car", token, ROI, query_us=100_000))
        ready.set()
        release.wait(10)

    worker = threading.Thread(target=coordinate)
    worker.start()
    assert ready.wait(10)
    try:
        assert cache.stats["entries"] == 1
        with pytest.raises(RuntimeError, match="owning preprocessing"):
            cache.get("car", token, ROI, query_us=100_000)
        with pytest.raises(RuntimeError, match="Join"):
            cache.close()
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    cache.close()
    np.testing.assert_array_equal(result[0], _normalized_crop(token.array, ROI))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_objects": 21},
        {"max_objects": True},
        {"max_objects": 0},
        {"max_crops_per_object": 17},
        {"max_crops_per_object": 0},
        {"max_bytes": 0},
    ],
)
def test_invalid_capacity_rejected_before_storage(kwargs):
    with pytest.raises(ValueError):
        ExactRGBCropCache(**kwargs)
