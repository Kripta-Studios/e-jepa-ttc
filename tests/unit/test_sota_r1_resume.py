"""Resume must preserve sensor state and keep restoration out of new requests."""

import gc
import weakref
from pathlib import Path

import numpy as np
import pytest

from operational.efficient_context import r1_profile
from operational.efficient_context.r1_reader import ResidentReplay
from operational.sota_eval.r1_resume import (
    ResumeReader,
    installed,
    receipt_snapshot,
    validate_receipt,
    verify_snapshot,
)


class Source:
    def __init__(self, times, path="sensor.h5"):
        self.path = Path(path)
        self.times = np.asarray(times, np.int64)
        self.calls = []

    def iter_window_chunks(self, start_us, end_us, *, chunk_events=250_000):
        self.calls.append((start_us, end_us))
        t = self.times[(self.times >= start_us) & (self.times < end_us)]
        for i in range(0, len(t), min(2, chunk_events)):
            piece = t[i : i + min(2, chunk_events)]
            yield dict(
                t=piece,
                x=(piece % 8).astype(np.int32),
                y=np.ones(len(piece), np.int32),
                p=np.ones(len(piece), np.int8),
            )


def group(intervals, saved_count, source_path="sensor.h5"):
    return dict(
        source_path=source_path,
        saved_count=saved_count,
        complete=saved_count == len(intervals),
        queries=[
            dict(start_us=a, cutoff_us=b, saved=i < saved_count)
            for i, (a, b) in enumerate(intervals)
        ],
    )


def state(reader):
    return (
        reader.cutoff,
        reader.coverage_start,
        reader.retained_bytes,
        [{key: value.tolist() for key, value in chunk.items()} for chunk in reader.chunks],
    )


def test_complete_group_never_ingests_or_opens_source():
    source = Source(range(100))
    reader = ResumeReader(source, 1000, group([(0, 10), (20, 30)], 2))
    reader.advance(0, 10)
    reader.advance(20, 30)
    assert source.calls == []
    assert reader.retained_bytes == 0
    assert reader.cutoff is None


@pytest.mark.parametrize("capacity", [1, 34, 1000])
def test_partial_prefix_restored_at_last_saved_boundary(capacity):
    intervals = [(0, 5), (2, 8), (5, 10)]
    times = [0, 1, 1, 3, 4, 5, 5, 7, 8, 9]
    source, original_source = Source(times), Source(times)
    restored = ResumeReader(source, capacity, group(intervals, 2))
    original = ResidentReplay(original_source, capacity)
    restored.advance(*intervals[0])
    assert source.calls == []
    restored.advance(*intervals[1])
    for bounds in intervals[:2]:
        original.advance(*bounds)
    assert source.calls == original_source.calls
    assert state(restored) == state(original)
    before = len(source.calls)
    restored.advance(*intervals[2])
    original.advance(*intervals[2])
    assert source.calls[before:] == [(8, 10)]
    assert state(restored) == state(original)
    assert restored.stats == original.stats | {"ingestion_ms": restored.stats["ingestion_ms"]}


def test_unseen_group_begins_with_its_own_history_only():
    source = Source(range(100, 150))
    reader = ResumeReader(source, 1000, group([(120, 130)], 0))
    reader.advance(120, 130)
    assert source.calls == [(120, 130)]


def test_resume_rejects_different_chronological_bounds():
    reader = ResumeReader(Source(range(10)), 1000, group([(0, 5)], 1))
    with pytest.raises(ValueError, match="chronological plan"):
        reader.advance(0, 6)


def test_factory_keeps_routes_independent_and_does_not_retain_buffers():
    g = group([(0, 5)], 0)
    original = r1_profile.ResidentReplay
    refs = []
    with installed(dict(groups=[g], capacity_bytes_per_route=1000)) as audits:
        for _ in range(2):
            reader = r1_profile.ResidentReplay(Source(range(10)), 1000)
            reader.advance(0, 5)
            refs.append(weakref.ref(reader))
            del reader
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert audits == [dict(position=1, planned_queries=1)] * 2
    assert r1_profile.ResidentReplay is original


def test_factory_restores_binding_on_error():
    original = r1_profile.ResidentReplay
    with pytest.raises(ValueError, match="source"):
        with installed(dict(groups=[group([(0, 5)], 0)], capacity_bytes_per_route=1000)):
            r1_profile.ResidentReplay(Source(range(10), path="other.h5"), 1000)
    assert r1_profile.ResidentReplay is original


def test_saved_receipt_snapshot_detects_mutation(tmp_path):
    folder = tmp_path / "measurement/fragments"
    folder.mkdir(parents=True)
    path = folder / "sample.json"
    path.write_text("{}")
    saved = receipt_snapshot(tmp_path)
    verify_snapshot(tmp_path, saved)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot(tmp_path, saved)


def test_failed_or_wrong_saved_pair_rejected():
    row = dict(
        block="b", label="H1", query="q", original_query=0, regime="R1_REPLAY", milliseconds=1.0
    )
    receipt = dict(
        requests=[row | dict(route="reference"), row | dict(route="mapped")],
        parity=dict(status="FAILED_INTEGRITY"),
    )
    with pytest.raises(ValueError, match="failed parity"):
        validate_receipt(receipt, block="b", arm="H1", query=dict(sample_token="q"), qi=0)
    with pytest.raises(ValueError, match="identity"):
        validate_receipt(receipt, block="b", arm="H8", query=dict(sample_token="q"), qi=0)
