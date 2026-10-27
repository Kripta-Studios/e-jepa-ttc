"""Bounded epochs and canonical tensor parity for temporal event prefixes."""

import os

import numpy as np
import pytest
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from operational.train40_runtime.integral import IntegralEventHistory
from operational.train40_runtime.temporal_prefix import TemporalPrefixEventHistory


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def events(start, end, count=8000):
    rng = np.random.default_rng(20261027)
    return {
        "x": rng.integers(0, 32, count, dtype=np.int32),
        "y": rng.integers(0, 24, count, dtype=np.int32),
        "t": np.sort(rng.integers(start, end, count, dtype=np.int64)),
        "p": rng.choice(np.array([-1, 1], dtype=np.int8), count),
    }


def window(raw, start, end):
    lo, hi = np.searchsorted(raw["t"], [start, end])
    return {key: value[lo:hi] for key, value in raw.items()}


def oracle(raw, end, roi, size=8):
    mapped = map_roi(raw, roi, size, 0.0)
    six = torch.stack(
        [
            encode_mapped(mapped, left, left + 100_000, size, "prefix-fixture")
            for left in range(end - 600_000, end, 100_000)
        ]
    )
    return torch.stack([six[index : index + 3] for index in range(4)])


def history(cls=TemporalPrefixEventHistory, device="cpu", **kwargs):
    return cls(device=device, width=32, height=24, roi_size=8, max_packet_events=1024, **kwargs)


@pytest.mark.parametrize("start", [0, 738, 2**52 + 738])
@pytest.mark.parametrize("normalization", ["sort", "topk"])
def test_epochs_wrap_without_changing_any_integer_moments_or_canonical_inputs(start, normalization):
    raw = events(start, start + 1_260_000, count=14000)
    prefix, ring = history(normalization=normalization), history(IntegralEventHistory)
    initial = window(raw, start, start + 600_000)
    for backend in (prefix, ring):
        backend.initialize(initial, start, start + 600_000)
    rois = [(1.25, 3.7, 28.6, 22.2), (-3.5, 4.25, 23.7, 25.5)]
    for end in range(start + 600_000, start + 1_260_001, 20_000):
        if end > start + 600_000:
            packet = window(raw, end - 20_000, end)
            for backend in (prefix, ring):
                backend.ingest(packet, end - 20_000, end)
        for roi in rois:
            actual = prefix.query(roi)
            torch.testing.assert_close(actual, ring.query(roi), rtol=0.0, atol=0.0)
            torch.testing.assert_close(actual, oracle(raw, end, roi), rtol=0.0, atol=0.0)
    assert prefix.stats["epoch_start_us"] == start + 1_200_000
    assert prefix.stats["queried_boundaries"] == 25
    assert prefix.allocated_bytes < ring.allocated_bytes


@pytest.mark.parametrize("normalization", ["sort", "topk"])
def test_twenty_distinct_rois_and_partial_workspace_preserve_order(normalization):
    raw = events(738, 620_738)
    prefix = history(normalization=normalization)
    prefix.initialize(window(raw, 738, 600_738), 738, 600_738)
    prefix.ingest(window(raw, 600_738, 620_738), 600_738, 620_738)
    rois = [(i * 0.3, i * 0.1, 24.0 + i * 0.2, 20.0 + i * 0.1) for i in range(20)]
    for selected in (rois, list(reversed(rois[:5]))):
        actual = prefix.query_many(selected)
        assert actual.shape == (len(selected), 4, 3, 12, 8, 8)
        for index, roi in enumerate(selected):
            torch.testing.assert_close(actual[index], oracle(raw, 620_738, roi), rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="1..20"):
        prefix.query_many(rois + [rois[0]])


@pytest.mark.parametrize("normalization", ["sort", "topk"])
def test_dense_epoch_and_crossing_quarter_do_not_overflow_int32_products(normalization):
    # Prefix count * 600000 exceeds 2**31, even though each count fits int32.
    start = 738
    raw = events(start, start + 620_000, count=16000)
    raw["x"].fill(0)
    raw["y"].fill(0)
    raw["p"].fill(1)
    prefix = history(normalization=normalization)
    prefix.initialize(window(raw, start, start + 600_000), start, start + 600_000)
    prefix.ingest(window(raw, start + 600_000, start + 620_000), start + 600_000, start + 620_000)
    roi = (0, 0, 32, 24)
    torch.testing.assert_close(prefix.query(roi), oracle(raw, start + 620_000, roi), rtol=0, atol=0)


def test_empty_histories_outside_rois_and_reset_never_retain_old_epoch_data():
    prefix = history()
    prefix.initialize(events(738, 600_738), 738, 600_738)
    assert not torch.count_nonzero(prefix.query((33.0, 30.0, 50.0, 47.0)))
    prefix.reset(739)
    assert not prefix.ready
    with pytest.raises(ValueError, match="600ms"):
        prefix.query((0.0, 0.0, 32.0, 24.0))
    prefix.initialize(events(739, 600_739, count=0), 739, 600_739)
    assert not torch.count_nonzero(prefix.query((0.0, 0.0, 32.0, 24.0)))
    prefix.ingest(events(600_739, 620_739, count=0), 600_739, 620_739)
    assert not torch.count_nonzero(prefix.query((0.0, 0.0, 32.0, 24.0)))


def test_validation_and_capacity_reject_before_mutating_history():
    prefix = history()
    prefix.initialize(events(738, 600_738, count=0), 738, 600_738)
    raw = events(600_738, 620_738, count=10)
    for start, end in ((600_739, 620_739), (580_738, 600_738), (600_738, 620_739)):
        with pytest.raises(ValueError, match="rollback/gap"):
            prefix.ingest(raw, start, end)
    raw["t"][[0, 1]] = raw["t"][[1, 0]]
    with pytest.raises(ValueError, match="monotonic"):
        prefix.ingest(raw, 600_738, 620_738)
    with pytest.raises(ValueError, match="capacity"):
        prefix.ingest(events(600_738, 620_738, count=1025), 600_738, 620_738)
    assert prefix.end_us == 600_738
    assert prefix.stats["total_buckets"] == 120
    assert not torch.count_nonzero(prefix.query((0.0, 0.0, 32.0, 24.0)))


def test_budget_and_epoch_overflow_are_checked_before_allocation():
    with pytest.raises(ValueError, match="above budget"):
        history(max_memory_bytes=100)
    with pytest.raises(ValueError, match="int32"):
        TemporalPrefixEventHistory(device="cpu", max_packet_events=2**31 // 30 + 1)


@pytest.mark.parametrize("normalization", ["sort", "topk"])
def test_bucket_quarter_epoch_edges_are_half_open_and_storage_accounting_is_exact(normalization):
    start = 738
    times = start + np.sort(
        np.concatenate((np.arange(0, 620_000, 5_000), np.arange(4_999, 620_000, 5_000)))
    )
    raw = {
        "x": np.zeros(len(times), dtype=np.int32),
        "y": np.zeros(len(times), dtype=np.int32),
        "t": times,
        # Preserve the sign even when input polarity cannot fit in int8.
        "p": np.full(len(times), 2**31, dtype=np.int64),
    }
    prefix = history(normalization=normalization)
    prefix.initialize(window(raw, start, start + 600_000), start, start + 600_000)
    roi = (0.0, 0.0, 32.0, 24.0)
    for end in (start + 600_000, start + 620_000):
        if end > start + 600_000:
            prefix.ingest(window(raw, end - 20_000, end), end - 20_000, end)
        torch.testing.assert_close(prefix.query(roi), oracle(raw, end, roi), rtol=0.0, atol=0.0)
    assert prefix.allocated_bytes == sum(
        value.numel() * value.element_size() for value in prefix._buffers
    )


@pytest.mark.cuda_fp32
@pytest.mark.skipif(os.environ.get("RUN_TRAIN40_CUDA_TESTS") != "1", reason="GPU slot opt-in")
@pytest.mark.parametrize("async_transfers", [False, True])
@pytest.mark.parametrize("normalization", ["sort", "topk"])
def test_cuda_prefix_matches_canonical_through_epoch_boundary(async_transfers, normalization):
    raw = events(738, 620_738)
    prefix = history(device="cuda", async_transfers=async_transfers, normalization=normalization)
    prefix.initialize(window(raw, 738, 600_738), 738, 600_738)
    prefix.ingest(window(raw, 600_738, 620_738), 600_738, 620_738)
    rois = [(1.25, 3.7, 28.6, 22.2), (-3.5, 4.25, 23.7, 25.5)]
    actual = prefix.query_many(rois).cpu()
    for index, roi in enumerate(rois):
        torch.testing.assert_close(actual[index], oracle(raw, 620_738, roi), rtol=0.0, atol=0.0)
