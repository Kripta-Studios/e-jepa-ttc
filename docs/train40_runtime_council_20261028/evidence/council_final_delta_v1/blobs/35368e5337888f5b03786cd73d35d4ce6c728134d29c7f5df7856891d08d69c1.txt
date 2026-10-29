"""Causality, boundedness and canonical-input parity of streaming moments."""

import os

import numpy as np
import pytest
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from operational.train40_runtime.integral import IntegralEventHistory, projection_bounds


def events(start, end, count=6000):
    rng = np.random.default_rng(2026)
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
            encode_mapped(mapped, left, left + 100_000, size, "fixture")
            for left in range(end - 600_000, end, 100_000)
        ]
    )
    return torch.stack([six[index : index + 3] for index in range(4)])


def history(device="cpu", **kwargs):
    return IntegralEventHistory(
        device=device, width=32, height=24, roi_size=8, max_packet_events=512, **kwargs
    )


def test_projection_boundaries_reproduce_canonical_pixel_assignment():
    for low, high in ((0.0, 32.0), (2.5, 29.3), (-7.5, 20.3), (33.0, 41.0)):
        bounds = projection_bounds(32, low, high, 8)
        for index in range(8):
            selected = [
                coordinate
                for coordinate in range(32)
                if low <= coordinate < high
                and int(np.floor((coordinate - low) * 8 / (high - low))) == index
            ]
            assert list(range(int(bounds[index]), int(bounds[index + 1]))) == selected


@pytest.mark.parametrize("roi", [(0.0, 0.0, 32.0, 24.0), (1.25, 3.7, 28.6, 22.2)])
def test_integer_moment_inputs_match_canonical_for_new_packet_and_moving_roi(roi):
    start = 1_000_738
    raw = events(start, start + 640_000)
    runtime = history()
    runtime.initialize(window(raw, start, start + 600_000), start, start + 600_000)
    for end in (start + 600_000, start + 620_000, start + 640_000):
        if end > start + 600_000:
            runtime.ingest(window(raw, end - 20_000, end), end - 20_000, end)
        expected = oracle(raw, end, roi)
        torch.testing.assert_close(runtime.query(roi), expected, rtol=0.0, atol=0.0)
        moved = (-3.5, 4.25, 23.7, 25.5)
        torch.testing.assert_close(
            runtime.query(moved), oracle(raw, end, moved), rtol=0.0, atol=0.0
        )
    assert runtime.end_us == start + 640_000
    assert runtime.stats["cuda_graph_captures"] == 0
    assert runtime.allocated_bytes < 10 * 1024**2


def test_empty_packets_and_unoccupied_roi_produce_zero_inputs():
    raw = events(738, 600_738, count=0)
    runtime = history()
    runtime.initialize(raw, 738, 600_738)
    assert not torch.count_nonzero(runtime.query((0.0, 0.0, 32.0, 24.0)))
    populated = events(738, 600_738)
    runtime.initialize(populated, 738, 600_738)
    assert not torch.count_nonzero(runtime.query((33.0, 30.0, 50.0, 47.0)))


def test_irregular_phase_requires_explicit_reset_and_never_rounds_clock():
    raw = events(738, 600_738, count=0)
    runtime = history()
    runtime.initialize(raw, 738, 600_738)
    for start, end in ((600_739, 620_739), (580_738, 600_738), (600_738, 620_739)):
        with pytest.raises(ValueError, match="rollback/gap"):
            runtime.ingest(raw, start, end)
    assert runtime.end_us == 600_738
    runtime.reset(739)
    assert not runtime.ready
    with pytest.raises(ValueError, match="600ms"):
        runtime.query((0.0, 0.0, 32.0, 24.0))
    runtime.initialize(events(739, 600_739, count=0), 739, 600_739)
    assert runtime.end_us == 600_739


def test_validation_and_capacity_fail_before_mutating_history():
    runtime = history()
    runtime.initialize(events(0, 600_000, count=0), 0, 600_000)
    packet = events(600_000, 620_000, count=10)
    packet["t"][[0, 1]] = packet["t"][[1, 0]]
    with pytest.raises(ValueError, match="monotonic"):
        runtime.ingest(packet, 600_000, 620_000)
    packet = events(600_000, 620_000, count=10)
    packet["x"][0] = -1
    with pytest.raises(ValueError, match="canvas"):
        runtime.ingest(packet, 600_000, 620_000)
    with pytest.raises(ValueError, match="capacity"):
        runtime.ingest(events(600_000, 620_000, count=513), 600_000, 620_000)
    assert runtime.end_us == 600_000
    assert not torch.count_nonzero(runtime.query((0.0, 0.0, 32.0, 24.0)))


def test_memory_budget_is_checked_before_allocation():
    with pytest.raises(ValueError, match="above budget"):
        history(max_memory_bytes=100)


def test_query_many_preserves_twenty_distinct_roi_order_and_full_coverage():
    raw = events(738, 600_738, count=9000)
    runtime = history()
    runtime.initialize(raw, 738, 600_738)
    rois = [
        (index * 0.3, index * 0.1, 24.0 + index * 0.2, 20.0 + index * 0.1) for index in range(20)
    ]
    actual = runtime.query_many(rois)
    assert actual.shape == (20, 4, 3, 12, 8, 8)
    for index, roi in enumerate(rois):
        torch.testing.assert_close(actual[index], oracle(raw, 600_738, roi), rtol=0.0, atol=0.0)
    # An odd number exercises a partially filled final workspace batch.
    subset = runtime.query_many(list(reversed(rois[:5])))
    for index, roi in enumerate(reversed(rois[:5])):
        torch.testing.assert_close(subset[index], oracle(raw, 600_738, roi), rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="1..20"):
        runtime.query_many(rois + [rois[0]])


@pytest.mark.cuda_fp32
@pytest.mark.skipif(os.environ.get("RUN_TRAIN40_CUDA_TESTS") != "1", reason="GPU slot opt-in")
@pytest.mark.parametrize("async_transfers", [False, True])
def test_cuda_integral_input_matches_canonical(async_transfers):
    raw = events(738, 620_738)
    runtime = history("cuda", async_transfers=async_transfers)
    runtime.initialize(window(raw, 738, 600_738), 738, 600_738)
    runtime.ingest(window(raw, 600_738, 620_738), 600_738, 620_738)
    roi = (1.25, 3.7, 28.6, 22.2)
    torch.testing.assert_close(
        runtime.query(roi).cpu(), oracle(raw, 620_738, roi), rtol=0.0, atol=0.0
    )
    moved = (-3.5, 4.25, 23.7, 25.5)
    first = runtime.query(roi).clone()
    second = runtime.query(moved)
    torch.testing.assert_close(first.cpu(), oracle(raw, 620_738, roi), rtol=0.0, atol=0.0)
    torch.testing.assert_close(second.cpu(), oracle(raw, 620_738, moved), rtol=0.0, atol=0.0)
