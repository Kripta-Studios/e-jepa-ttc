"""Input timer attribution without raw data, expert inference or optimizer work."""

import importlib.util
from pathlib import Path

import numpy as np


def test_reader_timer_excludes_consumer_and_preserves_chunks(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/profile_simplex_t_input_stages.py"
    spec = importlib.util.spec_from_file_location("input_timer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chunks = [{"t": np.array([1, 2])}, {"t": np.array([3])}]
    calls = []

    def fixture_chunks(self, start_us, end_us, *, chunk_events):
        calls.append((start_us, end_us, chunk_events))
        yield from chunks

    monkeypatch.setattr(module.CachedEventReader, "iter_window_chunks", fixture_chunks)
    ticks = iter([0.0, 2.0, 10.0, 13.0, 20.0, 21.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    reader = object.__new__(module.TimedReader)
    output = list(reader.iter_window_chunks(1, 4, chunk_events=7))
    assert calls == [(1, 4, 7)]
    assert all(left is right for left, right in zip(output, chunks, strict=True))
    assert reader.read_seconds == 6.0
    assert reader.events_read == 3
    other_reader = object.__new__(module.TimedReader)
    assert other_reader.read_seconds == 0.0
    assert other_reader.events_read == 0
