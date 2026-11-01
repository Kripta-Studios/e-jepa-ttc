"""Deterministic selection, cache-preserving instrumentation and explicit queue semantics."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from operational.final_refit.test12 import FinalV16Train40Predictor
from scripts import release_native_local_latency as runtime
from scripts.release_native_local_latency import queue_delay, select_queries


def test_selection_uses_time_prefixes_per_sequence_without_cost_or_prediction():
    rows = [
        {"sample_token": f"{s}:{i:02d}", "sequence_id": s, "query_time_us": i}
        for s in ("b", "a")
        for i in reversed(range(12))
    ]
    selected = select_queries(rows)
    assert list(selected) == ["a", "b"]
    assert [r["query_time_us"] for r in selected["a"]] == list(range(10))
    assert len(selected["b"]) == 10


def test_physical_alias_duplicates_are_not_independent_latency_samples():
    row = {"sample_token": "a", "sequence_id": "s", "query_time_us": 0}
    with pytest.raises(ValueError, match="aliases"):
        select_queries([row, row])


def test_queue_accumulates_shared_arrivals_and_clears_after_idle_gap():
    assert queue_delay([0.0, 0.0, 10.0, 100.0], [6.0, 6.0, 6.0, 6.0]) == [6.0, 12.0, 8.0, 6.0]


@pytest.mark.parametrize(
    "arrivals,service",
    [
        ([], []),
        ([1.0, 0.0], [1.0, 1.0]),
        ([0.0], [-1.0]),
        ([0.0], [np.nan]),
        ([0.0], [1.0, 2.0]),
    ],
)
def test_queue_rejects_undefined_or_misaligned_service(arrivals, service):
    with pytest.raises(ValueError):
        queue_delay(arrivals, service)


@pytest.mark.parametrize("instrumented", [False, True])
def test_timing_keeps_outputs_and_does_not_recompute_cached_observations(monkeypatch, instrumented):
    native = SimpleNamespace(event_values={}, rgb_values={})
    calls = []

    def expert(*args):
        calls.append(args[1])
        return (np.ones(17), np.ones(3))

    def observation(cache, prefix):
        def read(identity):
            if identity not in cache:
                cache[identity] = runtime.engine._expert_block(
                    {}, prefix, runtime.torch.empty(0), runtime.torch.empty(0)
                )
            return cache[identity]

        return read

    native._event_observation = observation(native.event_values, "E")
    native._rgb_observation = observation(native.rgb_values, "R")

    def predict(query, inputs):
        native._event_observation("e")
        native._event_observation("e")
        native._rgb_observation("r")
        return 3.0

    native.predict = predict
    monkeypatch.setattr(runtime.engine, "_expert_block", expert)
    monkeypatch.setattr(runtime.torch.cuda, "synchronize", lambda: None)
    first = runtime.timed_predict(cast(FinalV16Train40Predictor, native), {}, {}, instrumented)
    second = runtime.timed_predict(cast(FinalV16Train40Predictor, native), {}, {}, instrumented)
    assert first["prediction"] == second["prediction"] == 3.0
    assert calls == ["E", "R"]
    assert first["new_event_observations"] == first["new_rgb_observations"] == 1
    assert second["new_event_observations"] == second["new_rgb_observations"] == 0
    assert ("producers_ms" in first) == instrumented
    if instrumented:
        assert first["preparation_and_cache_ms"] >= 0
        assert first["context_fusion_host_ms"] >= 0
