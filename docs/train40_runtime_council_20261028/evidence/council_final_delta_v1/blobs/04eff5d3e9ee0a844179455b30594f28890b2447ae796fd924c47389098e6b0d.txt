"""CPU checks for immutable arrival tokens, ring eviction and lease lifetimes."""

from __future__ import annotations

import gc
import hashlib
import threading
import weakref
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from operational.rgb_port_revision.transfer_inputs import sensor_triplets
from operational.train40_runtime import rgb_raw_ring as module
from operational.train40_runtime.rgb_raw_ring import (
    FrozenRGBFrame,
    ResidentRGBFrameRing,
    RGBSourceIdentity,
    freeze_rgb_frame,
    validate_frozen_rgb_frame,
)


def source(name="sensor", revision="a"):
    return RGBSourceIdentity(name, revision * 64)


def frame(index, *, value=None, origin=None, timestamp=None, shape=(4, 5, 3)):
    pixels = np.full(shape, index if value is None else value, dtype=np.uint8)
    return freeze_rgb_frame(
        pixels, index, index * 100 if timestamp is None else timestamp, origin or source()
    )


def ring(capacity=3, **kwargs):
    return ResidentRGBFrameRing(frame_shape=(4, 5, 3), capacity=capacity, **kwargs)


def finish(instance, binding):
    instance.mark_consumed(binding)
    return binding


def test_freeze_copies_writable_inputs_and_captures_source_by_value():
    pixels = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)
    expected = pixels.copy()
    origin = source()
    token = freeze_rgb_frame(pixels, np.int64(7), np.int64(123), origin)
    pixels[:] = 0
    object.__setattr__(origin, "name", "changed")
    assert np.array_equal(token.array, expected)
    assert token.source == source()
    assert token.frame_id == 7 and token.timestamp_us == 123
    assert token.shape == (4, 5, 3)
    assert token.sha256 == hashlib.sha256(expected.tobytes()).hexdigest()
    assert token.content_sha256 == token.sha256
    assert token.hashed_bytes == 60
    assert token.freeze_seconds >= 0
    assert isinstance(token.payload, bytes)


def test_token_and_its_array_base_cannot_be_made_writable():
    token = frame(1)
    view = token.array
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view.setflags(write=True)
    with pytest.raises(ValueError):
        view.base.setflags(write=True)
    with pytest.raises(ValueError):
        view[0, 0, 0] = 8
    view.shape = (60,)
    assert token.array.shape == (4, 5, 3)
    with pytest.raises(AttributeError):
        object.__setattr__(token, "_payload", b"altered")
    with pytest.raises(AttributeError):
        token.frame_id = 99


def test_only_factory_tokens_are_admitted_and_registry_does_not_leak():
    with pytest.raises(TypeError, match="freeze_rgb_frame"):
        FrozenRGBFrame()
    forged = object.__new__(FrozenRGBFrame)
    with pytest.raises(ValueError, match="forged"):
        validate_frozen_rgb_frame(forged)
    instance = ring()
    with pytest.raises(ValueError, match="forged"):
        instance.bind([forged], 100)
    token = frame(1)
    reference = weakref.ref(token)
    del token
    gc.collect()
    assert reference() is None
    instance.close()


def test_noncontiguous_arrival_is_frozen_in_c_order():
    pixels = np.arange(120, dtype=np.uint8).reshape(4, 10, 3)[:, ::2]
    assert not pixels.flags.c_contiguous
    token = freeze_rgb_frame(pixels, 1, 100, source())
    assert token.array.flags.c_contiguous
    assert np.array_equal(token.array, pixels)


@pytest.mark.parametrize("change", ["pixels", "frame_id", "timestamp", "source_name", "revision"])
def test_every_provenance_or_content_change_changes_identity(change):
    original = frame(1)
    kwargs = {}
    index = 1
    if change == "pixels":
        kwargs["value"] = 2
    elif change == "frame_id":
        index, kwargs["value"], kwargs["timestamp"] = 2, 1, 100
    elif change == "timestamp":
        kwargs["timestamp"] = 101
    elif change == "source_name":
        kwargs["origin"] = source("other")
    else:
        kwargs["origin"] = source(revision="b")
    assert original.identity_sha256 != frame(index, **kwargs).identity_sha256


@pytest.mark.parametrize("revision", ["", "a" * 63, "A" * 64, "g" * 64])
def test_invalid_source_revision_rejected(revision):
    with pytest.raises(ValueError, match="SHA-256"):
        RGBSourceIdentity("sensor", revision)


def test_source_is_readonly_and_requires_nonempty_name():
    origin = source()
    with pytest.raises(FrozenInstanceError):
        origin.name = "other"
    with pytest.raises(ValueError, match="nonempty"):
        source("")


@pytest.mark.parametrize("invalid", ["dtype", "channels", "empty", "id", "clock", "bound"])
def test_invalid_arrival_rejected_before_copy(invalid):
    pixels, index, clock, limit = np.zeros((4, 5, 3), np.uint8), 1, 100, 1000
    if invalid == "dtype":
        pixels = pixels.astype(np.float32)
    elif invalid == "channels":
        pixels = pixels[:, :, :2]
    elif invalid == "empty":
        pixels = pixels[:0]
    elif invalid == "id":
        index = True
    elif invalid == "clock":
        clock = -1
    else:
        limit = 59
    with pytest.raises(ValueError):
        freeze_rgb_frame(pixels, index, clock, source(), max_bytes=limit)


def test_bind_copies_payload_in_input_order_and_hits_without_sha(monkeypatch):
    frames = [frame(1), frame(2), frame(3)]
    instance = ring()
    pointer = instance.tensor.data_ptr()
    monkeypatch.setattr(module.hashlib, "sha256", lambda *a, **k: pytest.fail("bind rehashed"))
    for token in frames:
        assert validate_frozen_rgb_frame(token) is None
    first = instance.bind(frames, 350)
    assert first.slots == (0, 1, 2)
    assert first.frame_ids == (1, 2, 3)
    assert first.source == source()
    for token, slot in zip(frames, first.slots, strict=True):
        assert np.array_equal(first.tensor[slot].numpy(), token.array)
    assert first.receipt["uploads"] == 3
    assert first.receipt["uploaded_bytes"] == 180
    assert first.receipt["h2d_bytes"] == 0
    finish(instance, first)
    second = instance.bind([frames[2], frames[0]], 360)
    assert second.slots == (2, 0)
    assert second.receipt["hits"] == 2
    assert second.receipt["uploads"] == second.receipt["uploaded_bytes"] == 0
    assert instance.tensor.data_ptr() == pointer
    finish(instance, second)
    assert instance.stats["frame_bytes_hashed_inside_bind"] == 0
    instance.close()


def test_independently_frozen_identical_arrival_hits_but_same_id_new_pixels_misses():
    instance = ring()
    finish(instance, instance.bind([frame(1)], 100))
    same = instance.bind([frame(1)], 100)
    assert same.receipt["hits"] == 1
    finish(instance, same)
    changed = instance.bind([frame(1, value=9)], 100)
    assert changed.receipt["hits"] == 0 and changed.receipt["uploads"] == 1
    assert np.all(changed.tensor[changed.slots[0]].numpy() == 9)
    finish(instance, changed)
    instance.close()


def test_eviction_protects_required_frames_even_when_misses_come_first():
    a, b, c, d = [frame(index) for index in range(1, 5)]
    instance = ring()
    finish(instance, instance.bind([a, b, c], 400))
    current = instance.bind([d, c, a], 410)
    assert current.slots == (1, 2, 0)
    assert current.receipt["hits"] == 2
    assert current.receipt["uploads"] == current.receipt["evictions"] == 1
    assert np.array_equal(current.tensor[0].numpy(), a.array)
    assert np.array_equal(current.tensor[2].numpy(), c.array)
    finish(instance, current)
    instance.close()


def test_lru_eviction_and_capacity_remain_bounded_during_many_arrivals():
    instance = ring(capacity=3)
    frames = [frame(index) for index in range(25)]
    for index in range(2, len(frames)):
        binding = instance.bind(frames[index - 2 : index + 1], index * 100)
        assert binding.receipt["uploads"] == (3 if index == 2 else 1)
        assert instance.stats["resident_frames"] == 3
        assert instance.stats["resident_payload_bytes"] == 180
        finish(instance, binding)
    assert instance.stats["uploads"] == 25
    assert instance.stats["evictions"] == 22
    instance.close()


def test_source_change_and_query_rollback_reset_explicitly():
    instance = ring()
    finish(instance, instance.bind([frame(1), frame(2)], 300))
    newer_source = [frame(1, origin=source(revision="b"))]
    changed = instance.bind(newer_source, 310)
    assert changed.receipt["source_reset"]
    assert not changed.receipt["query_rollback_reset"]
    assert changed.receipt["uploads"] == 1 and changed.receipt["hits"] == 0
    finish(instance, changed)
    rollback = instance.bind(newer_source, 200)
    assert rollback.receipt["query_rollback_reset"]
    assert not rollback.receipt["source_reset"]
    assert rollback.receipt["uploads"] == 1 and rollback.receipt["hits"] == 0
    finish(instance, rollback)
    assert instance.stats["source_resets"] == instance.stats["rollback_resets"] == 1
    assert instance.stats["invalidated_frames"] == 3
    instance.close()


def test_foreign_released_and_live_bindings_are_guarded():
    first, second = ring(), ring()
    binding = first.bind([frame(1)], 100)
    foreign = second.bind([frame(1)], 100)
    with pytest.raises(ValueError, match="foreign"):
        first.mark_consumed(foreign)
    with pytest.raises(RuntimeError, match="mark_consumed"):
        first.bind([frame(2)], 200)
    with pytest.raises(RuntimeError, match="mark_consumed"):
        first.reset()
    with pytest.raises(RuntimeError, match="mark_consumed"):
        first.close()
    finish(first, binding)
    with pytest.raises(ValueError, match="released"):
        binding.assert_valid()
    with pytest.raises(ValueError, match="released"):
        _ = binding.tensor
    with pytest.raises(ValueError):
        first.mark_consumed(binding)
    finish(second, foreign)
    first.close()
    second.close()


@pytest.mark.parametrize(
    "invalid", ["empty", "overcapacity", "duplicate", "source", "shape", "future"]
)
def test_invalid_binding_does_not_destroy_previous_residency(invalid):
    instance = ring()
    original = frame(1)
    finish(instance, instance.bind([original], 100))
    supplied, query = [frame(2)], 300
    if invalid == "empty":
        supplied = []
    elif invalid == "overcapacity":
        supplied = [frame(index) for index in range(4)]
    elif invalid == "duplicate":
        supplied = [frame(1), frame(1, value=8)]
    elif invalid == "source":
        supplied = [frame(1), frame(2, origin=source("other"))]
    elif invalid == "shape":
        supplied = [frame(2, shape=(5, 4, 3))]
    else:
        query = 199
    with pytest.raises(ValueError):
        instance.bind(supplied, query)
    hit = instance.bind([original], 100)
    assert hit.receipt["hits"] == 1
    finish(instance, hit)
    instance.close()


def test_cpu_path_never_calls_cuda_and_reports_allocation_bound(monkeypatch):
    for name in ("current_device", "current_stream", "Event", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: pytest.fail("CPU called CUDA"))
    instance = ring(max_bytes=180)
    assert instance.frame_shape == (4, 5, 3) and instance.capacity == 3
    receipt = instance.allocation_receipt
    assert receipt["resident_buffer_bytes"] == receipt["total_preallocated_tensor_bytes"] == 180
    assert receipt["pinned_host_buffer_bytes"] == 0
    finish(instance, instance.bind([frame(1)], 100))
    instance.reset()
    assert instance.stats["explicit_resets"] == 1
    assert instance.stats["resident_frames"] == 0
    instance.close()
    instance.close()
    with pytest.raises(RuntimeError, match="closed"):
        instance.bind([frame(1)], 100)
    with pytest.raises(ValueError, match="before allocation"):
        ring(max_bytes=179)


def test_ring_rejects_calls_from_another_thread():
    instance = ring()
    frames = [frame(1)]
    errors = []

    def attempt():
        try:
            instance.bind(frames, 100)
        except RuntimeError as error:
            errors.append(str(error))

    worker = threading.Thread(target=attempt)
    worker.start()
    worker.join()
    assert len(errors) == 1 and "owning thread" in errors[0]
    instance.close()


def test_upload_failure_poisoning_prevents_partial_history_reuse(monkeypatch):
    instance = ring()
    token = frame(1)

    def fail(*args, **kwargs):
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(module.np, "copyto", fail)
    with pytest.raises(RuntimeError, match="injected"):
        instance.bind([token], 100)
    assert instance.stats["failed_binds"] == 1
    assert instance.stats["resident_frames"] == 0
    with pytest.raises(RuntimeError, match="unusable"):
        instance.bind([token], 100)
    instance.close()


def _native_twenty_hz_replay(*, capacity=None):
    options = {} if capacity is None else {"capacity": capacity}
    instance = ResidentRGBFrameRing(frame_shape=(8, 8, 3), **options)
    clock = np.arange(0, 2_050_000, 50_000, dtype=np.int64)
    origin = source()
    tokens = {}
    records = []
    for query in range(650_000, 2_010_001, 20_000):
        # Only frames already available at this query are frozen. The initial
        # source history is causal; neither alternating resident chain is primed.
        available = int(np.searchsorted(clock, query, side="right"))
        for index in range(len(tokens), available):
            tokens[index] = freeze_rgb_frame(
                np.full((8, 8, 3), index, np.uint8), index, int(clock[index]), origin
            )
        groups = sensor_triplets(clock, query)
        ids = tuple(sorted({int(index) for group in groups for index in group}))
        assert len(ids) == 7 and all(tokens[index].timestamp_us <= query for index in ids)
        binding = instance.bind([tokens[index] for index in ids], query)
        records.append(
            {
                "frame_ids": ids,
                "uploads": binding.receipt["uploads"],
                "hits": binding.receipt["hits"],
                "resident_frames": instance.stats["resident_frames"],
            }
        )
        finish(instance, binding)
    receipt = instance.allocation_receipt
    instance.close()
    return records, receipt


def test_default_history_capacity_retains_both_native_alternating_chains():
    records, allocation = _native_twenty_hz_replay()
    assert allocation["capacity"] == 16
    assert allocation["max_frames_per_bind"] == 10
    warmed = set()
    for record in records:
        chain = record["frame_ids"][-1] % 2
        if chain not in warmed:
            assert record["uploads"] == 7
            warmed.add(chain)
        else:
            assert record["uploads"] <= 1
        assert record["resident_frames"] <= 16
    assert warmed == {0, 1}
    assert any(record["uploads"] == 0 for record in records)
    assert any(record["uploads"] == 1 for record in records)


def test_ten_historical_slots_reproduce_alternating_chain_reloads():
    records, _ = _native_twenty_hz_replay(capacity=10)
    warmed = set()
    reloaded = []
    for record in records:
        chain = record["frame_ids"][-1] % 2
        if chain in warmed:
            reloaded.append(record["uploads"])
        warmed.add(chain)
    assert max(reloaded) > 1


def test_historical_capacity_and_per_query_demand_have_separate_bounds(monkeypatch):
    instance = ResidentRGBFrameRing(frame_shape=(4, 5, 3), capacity=32)
    assert instance.capacity == 32
    with pytest.raises(ValueError, match="min\\(capacity, 10\\)"):
        instance.bind([frame(index) for index in range(11)], 1_100)
    binding = instance.bind([frame(index) for index in range(10)], 1_000)
    assert len(binding.slots) == 10
    finish(instance, binding)
    instance.close()
    with pytest.raises(ValueError, match="32 frames"):
        ResidentRGBFrameRing(frame_shape=(4, 5, 3), capacity=33)
    monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("allocated over byte bound"))
    with pytest.raises(ValueError, match="before allocation"):
        ResidentRGBFrameRing(frame_shape=(1200, 1920, 3), capacity=32)
