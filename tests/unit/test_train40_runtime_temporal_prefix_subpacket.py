"""Exact 20ms/5ms state, query cadence and bounded-capacity tests (CPU only)."""

from __future__ import annotations

import hashlib
from typing import TypeVar, overload

import numpy as np
import pytest
import torch

from operational.train40_runtime.temporal_prefix import TemporalPrefixEventHistory
from operational.train40_runtime.temporal_prefix_subpacket import (
    SubpacketTemporalPrefixEventHistory,
)

STATE_TENSORS = (
    "_counts",
    "_moments",
    "_cumulative_count",
    "_cumulative_moment",
    "_previous_count",
    "_previous_moment",
)
ROIS = [(0.0, 0.0, 12.0, 10.0), (-1.5, 1.25, 10.8, 11.0)]
History = TypeVar("History", bound=TemporalPrefixEventHistory)


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@overload
def history(*, normalization="sort", capacity=1024) -> SubpacketTemporalPrefixEventHistory: ...


@overload
def history(cls: type[History], *, normalization="sort", capacity=1024) -> History: ...


def history(
    cls: type[TemporalPrefixEventHistory] = SubpacketTemporalPrefixEventHistory,
    *,
    normalization="sort",
    capacity=1024,
) -> TemporalPrefixEventHistory:
    return cls(
        device="cpu",
        width=12,
        height=10,
        roi_size=8,
        max_packet_events=capacity,
        max_objects=2,
        roi_batch_size=2,
        normalization=normalization,
    )


def events(start, duration=1_200_000):
    rng = np.random.default_rng(264)
    edges = np.arange(0, duration, 5000, dtype=np.int64)
    # Repeated exact boundaries, final microseconds, and integer polarities
    # outside int8 exercise original half-open/sign normalization semantics.
    times = np.sort(np.concatenate((edges, edges, edges + 4999, rng.integers(0, duration, 9000))))
    count = len(times)
    return {
        "x": rng.integers(0, 12, count, dtype=np.int32),
        "y": rng.integers(0, 10, count, dtype=np.int32),
        "t": times + start,
        "p": rng.choice(np.array([-(2**31), 0, 2**31], dtype=np.int64), count),
    }


def empty():
    return {
        key: np.array([], dtype=dtype)
        for key, dtype in (("x", np.int32), ("y", np.int32), ("t", np.int64), ("p", np.int8))
    }


def window(raw, start, end):
    first, last = np.searchsorted(raw["t"], [start, end], side="left")
    return {key: value[first:last] for key, value in raw.items()}


def assert_state_equal(actual, expected):
    for name in STATE_TENSORS:
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
    for name in ("end_us", "_total_buckets", "_epoch_start_us", "_filled", "_ingested_packets"):
        assert getattr(actual, name) == getattr(expected, name), name


def digest(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("start", [738, 2**52 + 738])
@pytest.mark.parametrize("normalization", ["sort", "topk"])
def test_every_completed_group_preserves_integer_state_and_hashes_at_120_124_240(
    start, normalization
):
    raw = events(start)
    original = history(TemporalPrefixEventHistory, normalization=normalization)
    candidate = history(normalization=normalization)
    original.reset(start)
    candidate.reset(start)
    checked = []
    for left in range(start, start + 1_200_000, 20_000):
        original.ingest(window(raw, left, left + 20_000), left, left + 20_000)
        for sub_start in range(left, left + 20_000, 5000):
            candidate.ingest_subpacket(
                window(raw, sub_start, sub_start + 5000), sub_start, sub_start + 5000
            )
        assert_state_equal(candidate, original)
        if candidate._total_buckets in (120, 124, 240):
            expected = original.query_many(ROIS).clone()
            actual = candidate.query_many(ROIS).clone()
            assert torch.equal(actual, expected)
            assert digest(actual) == digest(expected)
            checked.append(candidate._total_buckets)
    assert checked == [120, 124, 240]
    assert candidate.allocated_bytes == original.allocated_bytes
    assert candidate.stats["subpacket_calls_since_reset"] == 240
    assert candidate.stats["subpacket_completed_updates_since_reset"] == 60
    assert candidate.stats["subpacket_pending_events"] == 0
    assert candidate.stats["bulk_20ms_calls_since_reset"] == 0


def test_prefill_600_plus_15ms_keeps_the_original_reset_origin_and_last_bucket_on_update():
    start = 738
    raw = events(start)
    original, candidate = history(TemporalPrefixEventHistory), history()
    for value in (original, candidate):
        value.initialize(window(raw, start, start + 600_000), start, start + 600_000)
    assert_state_equal(candidate, original)
    borrowed = candidate.query_many(ROIS)
    previous_pixels = borrowed.clone()
    for offset, left in enumerate(range(start + 600_000, start + 615_000, 5000), 1):
        candidate.ingest_subpacket(window(raw, left, left + 5000), left, left + 5000)
        assert candidate.subpacket_phase == offset
        assert not candidate.ready
        assert candidate._epoch_start_us == start + 600_000
        assert torch.equal(borrowed, previous_pixels)
        with pytest.raises(ValueError, match="four 5ms"):
            candidate.query(ROIS[0])
        with pytest.raises(ValueError, match="four 5ms"):
            candidate.query_many(ROIS)
    original.ingest(window(raw, start + 600_000, start + 620_000), start + 600_000, start + 620_000)
    # Same call shape as service.update(last_5ms, ...): no special model API.
    candidate.ingest(
        window(raw, start + 615_000, start + 620_000), start + 615_000, start + 620_000
    )
    assert candidate.ready and candidate.subpacket_phase == 0
    assert_state_equal(candidate, original)
    assert torch.equal(candidate.query_many(ROIS), original.query_many(ROIS))
    assert candidate.stats["bulk_20ms_calls_since_reset"] == 30
    assert candidate.stats["subpacket_calls_since_reset"] == 4


def repeated(start, count):
    return {
        "x": np.zeros(count, dtype=np.int32),
        "y": np.zeros(count, dtype=np.int32),
        "t": np.full(count, start, dtype=np.int64),
        "p": np.ones(count, dtype=np.int8),
    }


def test_capacity_is_aggregate_per_20ms_and_rejected_last_bucket_latches_until_reset():
    candidate = history(capacity=10)
    candidate.initialize(empty(), 738, 600_738)
    for left, count in ((600_738, 4), (605_738, 4), (610_738, 2)):
        candidate.ingest_subpacket(repeated(left, count), left, left + 5000)
    before = {name: getattr(candidate, name).clone() for name in STATE_TENSORS}
    with pytest.raises(ValueError, match="aggregate 20ms"):
        candidate.ingest_subpacket(repeated(615_738, 1), 615_738, 620_738)
    assert candidate.end_us == 615_738 and candidate._total_buckets == 123
    assert candidate.stats["subpacket_pending_events"] == 10
    assert all(torch.equal(getattr(candidate, name), value) for name, value in before.items())
    for call in (
        lambda: candidate.query_many(ROIS),
        lambda: candidate.ingest_subpacket(empty(), 615_738, 620_738),
    ):
        with pytest.raises(RuntimeError, match="RESET_REQUIRED"):
            call()
    candidate.reset(1_000_738)
    assert not candidate.ready and not candidate.stats["subpacket_invalid"]
    assert candidate.stats["subpacket_calls_since_reset"] == 0
    assert candidate.stats["subpacket_pending_events"] == 0
    assert candidate._total_buckets == 0
    candidate.initialize(empty(), 1_000_738, 1_600_738)
    assert candidate.ready
    assert not torch.count_nonzero(candidate.query_many(ROIS))


def test_exact_capacity_then_empty_subpackets_stays_admissible_and_resets_group_budget():
    candidate, original = history(capacity=10), history(TemporalPrefixEventHistory, capacity=10)
    for value in (candidate, original):
        value.initialize(empty(), 738, 600_738)
    original.ingest(repeated(600_738, 10), 600_738, 620_738)
    for index, left in enumerate(range(600_738, 620_738, 5000)):
        candidate.ingest_subpacket(repeated(left, 10) if index == 0 else empty(), left, left + 5000)
    assert_state_equal(candidate, original)
    assert candidate.stats["last_completed_subpacket_update_events"] == 10
    assert candidate.stats["subpacket_pending_events"] == 0
    candidate.ingest(empty(), 620_738, 640_738)
    original.ingest(empty(), 620_738, 640_738)
    assert_state_equal(candidate, original)
    assert torch.equal(candidate.query_many(ROIS), original.query_many(ROIS))


@pytest.mark.parametrize("fault", ["gap", "rollback", "duration", "at_end", "decreasing"])
def test_invalid_subpacket_preserves_prefix_bytes_then_latches(fault):
    candidate = history()
    candidate.initialize(empty(), 738, 600_738)
    left, right = 600_738, 605_738
    raw = repeated(left, 2)
    if fault == "gap":
        left += 1
        right += 1
    elif fault == "rollback":
        left -= 5000
        right -= 5000
    elif fault == "duration":
        right += 1
    elif fault == "at_end":
        raw["t"][-1] = right
    else:
        raw["t"][:] = [left + 1, left]
    before = {name: getattr(candidate, name).clone() for name in STATE_TENSORS}
    with pytest.raises(ValueError):
        candidate.ingest_subpacket(raw, left, right)
    assert candidate.end_us == 600_738 and candidate._total_buckets == 120
    assert all(torch.equal(getattr(candidate, name), value) for name, value in before.items())
    assert candidate.stats["subpacket_invalid"]
    with pytest.raises(RuntimeError, match="RESET_REQUIRED"):
        candidate.query(ROIS[0])


def test_bulk_update_cannot_silently_finish_a_partial_group():
    candidate = history()
    candidate.initialize(empty(), 738, 600_738)
    candidate.ingest_subpacket(empty(), 600_738, 605_738)
    with pytest.raises(ValueError, match="partial subpacket group"):
        candidate.ingest(empty(), 605_738, 625_738)
    assert candidate.end_us == 605_738 and candidate._total_buckets == 121
    assert candidate.stats["subpacket_invalid"]


@pytest.mark.parametrize("duration", [5000, 20_000])
def test_keyboard_interrupt_after_mutation_propagates_and_requires_reset(monkeypatch, duration):
    candidate = history()
    candidate.initialize(empty(), 738, 600_738)
    raw = repeated(600_739, 3)
    original_cumsum = torch.cumsum
    interruption = KeyboardInterrupt("injected after an ingestion mutation")
    calls = 0
    # In the bulk path, the fifth cumsum belongs to the second bucket: the
    # first prefix was already committed. In the 5ms path the fourth call has
    # copied RAW and mutated histogram/prefix scratch without committing yet.
    fail_on_call = 4 if duration == 5000 else 5

    def interrupt_after_cumsum(*args, **kwargs):
        nonlocal calls
        result = original_cumsum(*args, **kwargs)
        calls += 1
        if calls == fail_on_call:
            assert torch.equal(candidate._raw["t"][:3], torch.from_numpy(raw["t"]))
            if duration == 20_000:
                assert candidate._total_buckets == 121
                assert torch.count_nonzero(candidate._counts[121 % 121])
            else:
                assert torch.count_nonzero(candidate._count_work[0])
                assert torch.count_nonzero(candidate._moment_work[2])
            raise interruption
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch, "cumsum", interrupt_after_cumsum)
        with pytest.raises(KeyboardInterrupt) as caught:
            if duration == 5000:
                candidate.ingest_subpacket(raw, 600_738, 605_738)
            else:
                candidate.ingest(raw, 600_738, 620_738)
        assert caught.value is interruption
    assert calls == fail_on_call
    assert candidate.stats["subpacket_invalid"]
    assert not candidate.ready
    with pytest.raises(RuntimeError, match="RESET_REQUIRED"):
        candidate.query_many(ROIS)
    with pytest.raises(RuntimeError, match="RESET_REQUIRED"):
        candidate.ingest(empty(), 600_738, 620_738)
    candidate.reset(1_000_738)
    assert not candidate.stats["subpacket_invalid"]
    assert candidate._total_buckets == 0
    candidate.initialize(empty(), 1_000_738, 1_600_738)
    assert candidate.ready
    assert not torch.count_nonzero(candidate.query_many(ROIS))
