"""Causal, capacity and exact sensor-byte contracts for resident replay."""

import numpy as np
import pytest

from operational.efficient_context.r1_reader import ResidentReplay


class Source:
    def __init__(self, times, chunk=2):
        self.raw = dict(t=np.asarray(times, np.int64))
        self.raw.update(
            x=np.arange(len(times), dtype=np.int32),
            y=np.ones(len(times), np.int32),
            p=np.where(np.arange(len(times)) % 2, 1, -1).astype(np.int8),
        )
        self.chunk = chunk
        self.requests = []

    def iter_window_chunks(self, start_us, end_us, *, chunk_events=250_000):
        self.requests.append((start_us, end_us))
        mask = (self.raw["t"] >= start_us) & (self.raw["t"] < end_us)
        raw = {k: v[mask] for k, v in self.raw.items()}
        for i in range(0, len(raw["t"]), min(chunk_events, self.chunk)):
            yield {k: v[i : i + min(chunk_events, self.chunk)] for k, v in raw.items()}


def collect(reader, start, end):
    chunks = list(reader.iter_window_chunks(start, end))
    return {
        k: np.concatenate([c[k] for c in chunks]) if chunks else np.array([])
        for k in ("x", "y", "t", "p")
    }


@pytest.mark.parametrize("capacity", [1, 34, 68, 1024])
def test_r1_ties_capacity_and_gap_ingestion(capacity):
    source = Source([0, 1, 1, 2, 4, 4, 6, 10, 10, 11, 12])
    replay = ResidentReplay(source, capacity)
    for start, end in [(0, 2), (1, 7), (10, 12)]:
        replay.advance(start, end)
        actual, expected = collect(replay, start, end), collect(Source(source.raw["t"]), start, end)
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])
        assert replay.retained_bytes <= capacity
    assert (7, 12) in source.requests
    assert replay.stats["ingested_events"] == 10
    assert replay.stats["stream_seconds"] == pytest.approx(12 / 1e6)


def test_r1_no_future_and_query_rollback():
    replay = ResidentReplay(Source([0, 1, 2]))
    replay.advance(0, 2)
    with pytest.raises(ValueError, match="cutoff"):
        collect(replay, 0, 3)
    with pytest.raises(ValueError, match="rollback"):
        replay.advance(0, 1)


def test_r1_cross_chunk_rollback():
    replay = ResidentReplay(Source([0, 2, 1, 3], chunk=2))
    with pytest.raises(ValueError, match="rollback"):
        replay.advance(0, 4)


def test_r1_large_timestamps_and_earlier_history_fallback():
    base = 2**54
    replay = ResidentReplay(Source([base, base + 1, base + 2, base + 3]))
    replay.advance(base + 1, base + 3)
    replay.advance(base, base + 4)
    np.testing.assert_array_equal(
        collect(replay, base, base + 4)["t"], np.arange(4, dtype=np.int64) + base
    )
    assert replay.stats["capacity_fallback_reads"] == 1


@pytest.mark.parametrize("roi", [(0, 0, 8, 8), (-5, -5, 3, 3), (900, 900, 908, 908)])
@pytest.mark.parametrize("offset", [0.0, 0.5, -1.0])
def test_r1_tensor_parity_with_changed_roi_and_offset(roi, offset):
    import torch

    from e_jepa_ttc.efficient_context.mapped_union import encode_union
    from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union

    source = Source([0, 1, 1, 3, 4, 5, 5, 6, 7, 9, 10, 12], chunk=3)
    replay = ResidentReplay(source, 68)
    replay.advance(0, 13)
    windows = np.array([[1, 5], [5, 9], [9, 13]], np.int64)
    lags = np.zeros(16, np.int64)
    valid = np.ones(16, bool)
    options = dict(sequence_id="synthetic", roi_size=8, event_pixel_diff=offset)
    expected = encode_context_union(source, windows, lags, valid, roi, **options)
    actual = encode_union(replay, windows, lags, valid, roi, **options)
    assert torch.equal(expected, actual)
