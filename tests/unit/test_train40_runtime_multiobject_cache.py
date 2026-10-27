"""CPU integration receipts for RGB cache chunking and update lifetimes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from operational.train40_runtime.multiobject import ObjectRegion, RawRGBHistory
from operational.train40_runtime.multiobject_cache import RGBBatchCache


def history(query=700_000):
    frames = {index: np.full((4, 5, 3), index, np.uint8) for index in range(7)}
    for frame in frames.values():
        frame.flags.writeable = False
    return RawRGBHistory(
        query,
        frames,
        {index: (index + 1) * 100_000 for index in frames},
        tuple((index, index + 1, index + 2) for index in range(5)),
    )


def objects(count=3):
    return [ObjectRegion(str(index), (0, 0, 4, 5), (0, 0, 4, 5)) for index in range(count)]


def cache(**kwargs):
    return RGBBatchCache(model_sha256="a" * 64, runtime_sha256="b" * 64, **kwargs)


def store(instance, ticket):
    rows = len(ticket.present_indices) * 5
    raw = torch.arange(rows * 17, dtype=torch.float32).reshape(rows, 17)
    experts = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
    assert instance.store(ticket, raw, experts, guard_passed=True)
    return raw, experts


def test_roundtrip_preserves_flat_order_and_recomputes_only_query_metadata():
    instance = cache()
    first = instance.begin(objects(), history())
    assert first.cached_features is None
    raw, experts = store(instance, first)
    hit = instance.begin(objects(), history(720_000))
    assert hit.present_indices == (0, 1, 2)
    assert hit.cached_features is not None
    assert torch.equal(torch.cat([item.raw for item in hit.cached_features]), raw)
    assert torch.equal(torch.cat([item.experts for item in hit.cached_features]), experts)
    assert hit.identity is not None and first.identity is not None
    assert hit.identity.digest == first.identity.digest
    assert hit.diagnostics["hashed_bytes"] == 7 * 4 * 5 * 3
    assert instance.stats["served_object_hits"] == 3
    assert instance.stats["tensor_bytes"] == 3 * 5 * 20 * 4


def test_chunk_keys_use_present_position_and_actual_partial_chunk_shape():
    regions = objects(5)
    regions[1] = replace(regions[1], rgb_square=None)
    ticket = cache(rgb_object_chunk_size=3).begin(regions, history())
    assert ticket.present_indices == (0, 2, 3, 4)
    assert [key.batch_shape for key in ticket.keys] == [
        (15, 3, 3, 128, 128),
        (15, 3, 3, 128, 128),
        (15, 3, 3, 128, 128),
        (5, 3, 3, 128, 128),
    ]
    assert [key.object_slot for key in ticket.keys] == [0, 1, 2, 0]


def test_chunk_contract_is_readonly_and_available_to_service_validation():
    instance = cache(rgb_object_chunk_size=1)
    assert instance.rgb_object_chunk_size == 1
    assert cache().rgb_object_chunk_size is None
    with pytest.raises(AttributeError):
        instance.rgb_object_chunk_size = 2  # type: ignore[misc]


def test_one_roi_change_requires_recomputing_whole_lot():
    instance = cache()
    regions = objects()
    store(instance, instance.begin(regions, history()))
    regions[1] = replace(regions[1], rgb_square=(0.001, 0, 4, 5))
    miss = instance.begin(regions, history(720_000))
    assert miss.cached_features is None
    assert len(miss.keys) == 3
    assert instance.stats["key_matches"] == 2
    assert instance.stats["invalidations"] == 1
    assert instance.stats.get("served_object_hits", 0) == 0


def test_reordering_same_chunk_invalidates_object_slots():
    instance = cache()
    regions = objects()
    store(instance, instance.begin(regions, history()))
    assert instance.begin(list(reversed(regions)), history()).cached_features is None


def test_absent_region_and_departed_object_release_feature_bytes():
    instance = cache(rgb_object_chunk_size=1)
    regions = objects()
    store(instance, instance.begin(regions, history()))
    reduced = [regions[0], replace(regions[1], rgb_square=None)]
    ticket = instance.begin(reduced, history())
    assert ticket.cached_features is not None
    assert ticket.present_indices == (0,)
    assert instance.stats["entries"] == 1
    assert instance.stats["invalidations"] == 2


def test_unavailable_history_clears_cache_without_hashing():
    instance = cache()
    store(instance, instance.begin(objects(), history()))
    unavailable = RawRGBHistory(720_000, {}, {}, (), "INSUFFICIENT_CAUSAL_RGB_HISTORY")
    ticket = instance.begin(objects(), unavailable)
    assert ticket.cached_features == []
    assert ticket.keys == ticket.present_indices == ()
    assert ticket.identity is None
    assert ticket.diagnostics["hashed_bytes"] == 0
    assert instance.stats["entries"] == 0
    assert instance.begin(objects(), history()).cached_features is None


def test_contradictory_availability_and_weak_writable_frames_fail_closed():
    instance = cache()
    packet = history()
    with pytest.raises(ValueError, match="Unavailable"):
        instance.begin(objects(), replace(packet, unavailable_reason="MISSING"))
    packet.frames[0].flags.writeable = True
    with pytest.raises(ValueError, match="read-only"):
        instance.begin(objects(), packet)


@pytest.mark.parametrize("value", [0, 21, True, 1.5])
def test_invalid_chunk_size_rejected_at_construction(value):
    with pytest.raises(ValueError, match="chunk_size"):
        cache(rgb_object_chunk_size=value)


@pytest.mark.parametrize("field", ["model_sha256", "runtime_sha256"])
def test_weak_model_or_runtime_identity_rejected_at_construction(field):
    kwargs = {"model_sha256": "a" * 64, "runtime_sha256": "b" * 64}
    kwargs[field] = "unverified"
    with pytest.raises(ValueError, match=field):
        RGBBatchCache(**kwargs)


def test_foreign_stale_reused_and_hit_tickets_cannot_be_stored():
    instance = cache()
    foreign = cache().begin(objects(), history())
    current = instance.begin(objects(), history())
    raw, experts = torch.zeros(15, 17), torch.zeros(15, 3)
    with pytest.raises(ValueError, match="latest unused"):
        instance.store(foreign, raw, experts, guard_passed=True)
    newer = instance.begin(objects(), history())
    with pytest.raises(ValueError, match="latest unused"):
        instance.store(current, raw, experts, guard_passed=True)
    store(instance, newer)
    with pytest.raises(ValueError, match="latest unused"):
        instance.store(newer, raw, experts, guard_passed=True)
    hit = instance.begin(objects(), history())
    with pytest.raises(ValueError, match="missed nonempty"):
        instance.store(hit, raw, experts, guard_passed=True)


@pytest.mark.parametrize("failure", ["shape", "guard", "nan"])
def test_invalid_insertions_fail_and_consume_ticket(failure):
    instance = cache()
    ticket = instance.begin(objects(), history())
    raw, experts = torch.zeros(15, 17), torch.zeros(15, 3)
    if failure == "shape":
        raw = raw[:-1]
    elif failure == "nan":
        raw[0, 0] = float("nan")
    with pytest.raises(ValueError):
        instance.store(ticket, raw, experts, guard_passed=failure != "guard")
    assert instance.stats["entries"] == 0
    with pytest.raises(ValueError, match="latest unused"):
        instance.store(ticket, raw, experts, guard_passed=True)


def test_small_byte_budget_rejects_lot_and_clear_invalidates_pending_ticket():
    instance = cache(max_bytes=100)
    ticket = instance.begin(objects(), history())
    raw, experts = torch.zeros(15, 17), torch.zeros(15, 3)
    assert not instance.store(ticket, raw, experts, guard_passed=True)
    assert instance.stats["tensor_bytes"] == 0
    pending = instance.begin(objects(), history())
    instance.clear()
    with pytest.raises(ValueError, match="latest unused"):
        instance.store(pending, raw, experts, guard_passed=True)


@pytest.mark.parametrize("kind", ["empty", "overbound", "duplicate"])
def test_object_contract_rejected_before_lookup(kind):
    regions = objects(21 if kind == "overbound" else 3)
    if kind == "empty":
        regions = []
    elif kind == "duplicate":
        regions[1] = regions[0]
    with pytest.raises(ValueError):
        cache().begin(regions, history())
