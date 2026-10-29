"""CPU checks for metadata-only identity of authentic immutable RGB histories."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from operational.train40_runtime import rgb_frozen_identity as module
from operational.train40_runtime.feature_cache import (
    ExactRGBFeatureCache,
    bind_rgb_history,
    rgb_feature_key,
)
from operational.train40_runtime.rgb_frozen_identity import bind_frozen_rgb_history
from operational.train40_runtime.rgb_raw_ring import (
    FrozenRGBFrame,
    RGBSourceIdentity,
    freeze_rgb_frame,
)


def history(*, shape=(4, 5, 3), source=None):
    origin = source or RGBSourceIdentity("sensor", "a" * 64)
    frames = {
        index: freeze_rgb_frame(
            np.full(shape, index, np.uint8), index, (index + 1) * 100_000, origin
        )
        for index in range(7)
    }
    groups = tuple((index, index + 1, index + 2) for index in range(5))
    return frames, groups


def identity(*, query=700_000):
    frames, groups = history()
    return bind_frozen_rgb_history(frames, groups, query_us=query)


def key(bound):
    return rgb_feature_key(
        "car",
        (0, 0, 4, 5),
        bound,
        model_sha256="b" * 64,
        runtime_sha256="c" * 64,
        batch_shape=(5, 3, 3, 128, 128),
        object_slot=0,
    )


def test_query_changes_only_availability_and_not_producer_identity():
    first, later = identity(), identity(query=720_000)
    assert first.digest == later.digest
    assert key(first).digest == key(later).digest
    assert first.source_binding == ("IMMUTABLE_OWNED_RGB_V1", "sensor", "a" * 64)
    assert first.raw_bytes == 7 * 4 * 5 * 3
    assert first.hashed_bytes == 0
    assert first.fingerprint_seconds >= 0


def test_native_metadata_and_deltas_match_content_hashed_core_oracle():
    frames, groups = history()
    immutable = bind_frozen_rgb_history(frames, groups, query_us=700_000)
    oracle = bind_rgb_history(
        {index: token.array for index, token in frames.items()},
        {index: token.timestamp_us for index, token in frames.items()},
        groups,
        query_us=700_000,
    )
    assert immutable.triplets == oracle.triplets
    assert immutable.frame_clocks_us == oracle.frame_clocks_us
    assert immutable.delta_seconds_hex == oracle.delta_seconds_hex
    assert immutable.raw_bytes == oracle.raw_bytes
    assert immutable.digest != oracle.digest


def test_only_a_small_descriptor_is_hashed_even_for_large_raw_payloads(monkeypatch):
    frames, groups = history(shape=(180, 240, 3))
    original = module.hashlib.sha256
    sizes = []

    def small_only(data):
        assert len(data) < 16 * 1024
        sizes.append(len(data))
        return original(data)

    monkeypatch.setattr(module.hashlib, "sha256", small_only)
    result = bind_frozen_rgb_history(frames, groups, query_us=700_000)
    assert result.raw_bytes == 907_200
    assert result.hashed_bytes == 0 and len(sizes) == 1


@pytest.mark.parametrize(
    "change", ["pixels", "timestamp", "frame_id", "shape", "history", "source"]
)
def test_every_bound_input_change_invalidates_feature_key(change):
    frames, groups = history()
    original = bind_frozen_rgb_history(frames, groups, query_us=700_000)
    if change in {"pixels", "timestamp", "shape"}:
        token = frames[2]
        pixels = token.array.copy()
        if change == "pixels":
            pixels[0, 0, 0] = 99
        if change == "shape":
            pixels = pixels.reshape(5, 4, 3)
        frames[2] = freeze_rgb_frame(
            pixels, 2, token.timestamp_us + int(change == "timestamp"), token.source
        )
    elif change == "frame_id":
        token = frames.pop(0)
        frames[8] = freeze_rgb_frame(token.array, 8, token.timestamp_us, token.source)
        groups = ((8, 1, 2), *groups[1:])
    elif change == "history":
        groups = ((0, 1, 2), (0, 2, 3), *groups[2:])
    else:
        frames, groups = history(source=RGBSourceIdentity("sensor", "d" * 64))
    proposed = bind_frozen_rgb_history(frames, groups, query_us=700_000)
    assert original.digest != proposed.digest
    assert key(original).digest != key(proposed).digest


def test_cache_reuses_guarded_features_at_new_query_with_current_token_pixels():
    frames, groups = history()
    cache = ExactRGBFeatureCache()
    first = bind_frozen_rgb_history(frames, groups, query_us=700_000)
    raw, experts = torch.arange(85).reshape(1, 5, 17).float(), torch.ones(1, 5, 3)
    assert cache.store_many([key(first)], raw, experts, guard_passed=True)
    later = bind_frozen_rgb_history(frames, groups, query_us=720_000)
    hit = cache.lookup_many([key(later)])
    assert hit is not None and torch.equal(hit[0].raw, raw[0])
    assert all(not token.array.flags.writeable for token in frames.values())


def test_mutating_original_array_does_not_change_token_pixels_or_identity():
    origin = RGBSourceIdentity("sensor", "a" * 64)
    mutable = np.full((4, 5, 3), 9, np.uint8)
    a = freeze_rgb_frame(mutable, 1, 100_000, origin)
    b = freeze_rgb_frame(mutable, 2, 200_000, origin)
    first = bind_frozen_rgb_history({1: a, 2: b}, ((1, 2),), query_us=200_000)
    mutable[:] = 200
    later = bind_frozen_rgb_history({1: a, 2: b}, ((1, 2),), query_us=220_000)
    assert first.digest == later.digest
    assert np.all(a.array == 9)
    assert not np.array_equal(a.array, mutable)


@pytest.mark.parametrize("invalid", ["array", "forged", "mapping_id", "missing", "source"])
def test_weak_or_inconsistent_frame_bindings_fail_closed(invalid):
    frames, groups = history()
    if invalid == "array":
        frames[1] = frames[1].array
    elif invalid == "forged":
        frames[1] = object.__new__(FrozenRGBFrame)
    elif invalid == "mapping_id":
        frames[1] = frames[0]
    elif invalid == "missing":
        frames.pop(1)
    else:
        old = frames[1]
        frames[1] = freeze_rgb_frame(
            old.array, old.frame_id, old.timestamp_us, RGBSourceIdentity("other", "a" * 64)
        )
    with pytest.raises(ValueError):
        bind_frozen_rgb_history(frames, groups, query_us=700_000)


@pytest.mark.parametrize("query", [699_999, 750_001, True, -1])
def test_query_causality_and_type_are_checked_every_call(query):
    frames, groups = history()
    with pytest.raises(ValueError):
        bind_frozen_rgb_history(frames, groups, query_us=query)


@pytest.mark.parametrize(
    "invalid", ["empty", "overlong", "mixed", "one_step", "reversed", "anchors", "boolean"]
)
def test_history_shape_and_chronology_guards(invalid):
    frames, groups = history()
    if invalid == "empty":
        groups = ()
    elif invalid == "overlong":
        groups = groups + groups
    elif invalid == "mixed":
        groups = ((0, 1), *groups[1:])
    elif invalid == "one_step":
        groups = ((0,),)
    elif invalid == "reversed":
        groups = ((2, 1, 0),)
    elif invalid == "anchors":
        groups = (groups[1], groups[0])
    else:
        groups = ((False, 1, 2),)
    with pytest.raises(ValueError):
        bind_frozen_rgb_history(frames, groups, query_us=700_000)


def test_more_than_ten_distinct_frames_is_rejected():
    origin = RGBSourceIdentity("sensor", "a" * 64)
    frames = {
        index: freeze_rgb_frame(np.zeros((2, 2, 3), np.uint8), index, index * 10_000, origin)
        for index in range(12)
    }
    groups = tuple((index, index + 1) for index in range(0, 12, 2))
    with pytest.raises(ValueError, match="ten distinct"):
        bind_frozen_rgb_history(frames, groups, query_us=200_000)


def test_raw_byte_bound_counts_only_referenced_immutable_frames():
    frames, groups = history()
    with pytest.raises(ValueError, match="byte bound"):
        bind_frozen_rgb_history(frames, groups, query_us=700_000, max_raw_bytes=419)
    result = bind_frozen_rgb_history(frames, ((4, 5, 6),), query_us=700_000, max_raw_bytes=180)
    assert result.raw_bytes == 180


def test_oversized_source_metadata_is_rejected_before_hashing(monkeypatch):
    frames, groups = history(source=RGBSourceIdentity("x" * 17_000, "a" * 64))
    monkeypatch.setattr(module.hashlib, "sha256", lambda *a, **k: pytest.fail("oversize hashed"))
    with pytest.raises(ValueError, match="16 KiB"):
        bind_frozen_rgb_history(frames, groups, query_us=700_000)
