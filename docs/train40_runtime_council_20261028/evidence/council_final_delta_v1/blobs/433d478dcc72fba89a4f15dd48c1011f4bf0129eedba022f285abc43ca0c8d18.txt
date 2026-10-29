"""Integer-selector parity, bounded storage and explicit synthetic compilation."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from operational.train40_runtime.compiled_prefix import (
    CompiledTemporalPrefixEventHistory,
    _select_into,
)
from operational.train40_runtime.temporal_prefix import TemporalPrefixEventHistory


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def events(start, end, count=16000):
    rng = np.random.default_rng(20261028)
    return {
        "x": rng.integers(0, 32, count, dtype=np.int32),
        "y": rng.integers(0, 24, count, dtype=np.int32),
        "t": np.sort(rng.integers(start, end, count, dtype=np.int64)),
        "p": rng.choice(np.array([-1, 1], dtype=np.int8), count),
    }


def window(raw, start, end):
    lo, hi = np.searchsorted(raw["t"], [start, end])
    return {key: value[lo:hi] for key, value in raw.items()}


def history(cls=CompiledTemporalPrefixEventHistory, device="cpu", **kwargs):
    return cls(device=device, width=32, height=24, roi_size=8, max_packet_events=1024, **kwargs)


def oracle(raw, end, roi):
    mapped = map_roi(raw, roi, 8, 0.0)
    six = torch.stack(
        [
            encode_mapped(mapped, left, left + 100_000, 8, "selector-fixture")
            for left in range(end - 600_000, end, 100_000)
        ]
    )
    return torch.stack([six[index : index + 3] for index in range(4)])


@pytest.mark.parametrize("start", [738, 2**52 + 738])
@pytest.mark.parametrize("normalization", ["sort", "topk"])
@pytest.mark.parametrize("compile_backend", [None, "eager"])
def test_selector_is_exact_across_shapes_phases_and_large_absolute_clocks(
    start, normalization, compile_backend
):
    raw = events(start, start + 1_260_000)
    selected = history(normalization=normalization, compile_backend=compile_backend)
    baseline = history(TemporalPrefixEventHistory, normalization=normalization)
    selected.prime_selectors([1, 5, 20])
    assert selected.stats["queries"] == 0
    assert selected.stats["selector_compiled_shapes"] == "1,4"
    for runtime in (selected, baseline):
        runtime.initialize(window(raw, start, start + 600_000), start, start + 600_000)
    checkpoints = {600_000, 620_000, 640_000, 1_180_000, 1_200_000, 1_220_000, 1_260_000}
    for end in range(start + 600_000, start + 1_260_001, 20_000):
        if end > start + 600_000:
            for runtime in (selected, baseline):
                runtime.ingest(window(raw, end - 20_000, end), end - 20_000, end)
        if end - start not in checkpoints:
            continue
        for count in (1, 5, 20):
            rois = [(i * 0.3, i * 0.1, 24.0 + i * 0.2, 20.0 + i * 0.1) for i in range(count)]
            actual = selected.query_many(rois)
            expected = baseline.query_many(rois)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if count == 1:
                torch.testing.assert_close(actual[0], oracle(raw, end, rois[0]), rtol=0, atol=0)


def test_old_gather_storage_is_actually_released_and_memory_accounting_is_exact():
    selected = history(compile_backend=None)
    assert selected.released_gather_bytes > 0
    assert selected.allocated_bytes < selected.initial_buffer_bytes_before_release
    assert selected._count_gather == [] and selected._moment_gather == []
    for name in ("_address", "_quarter_work", "_count_gather_0", "_moment_gather_1"):
        assert not hasattr(selected, name)
    assert selected.allocated_bytes == sum(
        value.numel() * value.element_size() for value in selected._buffers
    )
    assert selected.stats["max_take_temporary_bytes"] == 0


def test_compilation_must_be_primed_before_sensor_data_and_never_hides_first_query():
    selected = history(compile_backend="eager")
    selected.prime_selectors([20])
    assert selected.stats["selector_compiled_shapes"] == "4"
    raw = events(738, 600_738, count=0)
    selected.initialize(raw, 738, 600_738)
    with pytest.raises(ValueError, match="not primed"):
        selected.query((0.0, 0.0, 32.0, 24.0))
    with pytest.raises(ValueError, match="before any sensor history"):
        selected.prime_selectors([1])
    assert selected.stats["queries"] == 0


def test_synthetic_priming_and_real_queries_share_the_same_compiler_execution_mode():
    selected = history(compile_backend="eager")
    compilations = []

    def counting_backend(graph, inputs):
        compilations.append(
            tuple(value.shape for value in inputs if isinstance(value, torch.Tensor))
        )
        return graph.forward

    selected._selector = torch.compile(
        _select_into, backend=counting_backend, fullgraph=True, dynamic=False
    )
    selected.prime_selectors([1, 20])
    primed_count = len(compilations)
    assert primed_count > 0
    raw = events(738, 620_738)
    selected.initialize(window(raw, 738, 600_738), 738, 600_738)
    rois = [(i * 0.3, i * 0.1, 24.0 + i * 0.2, 20.0 + i * 0.1) for i in range(20)]
    for end in (600_738, 620_738):
        if end > 600_738:
            selected.ingest(window(raw, end - 20_000, end), end - 20_000, end)
        for count in (1, 20):
            selected.query_many(rois[:count])
    assert len(compilations) == primed_count


def test_empty_boundary_events_dense_counts_and_invalid_packets_keep_contracts():
    start = 738
    raw = events(start, start + 620_000)
    raw["x"].fill(0)
    raw["y"].fill(0)
    raw["p"].fill(1)
    selected = history(compile_backend=None)
    selected.initialize(window(raw, start, start + 600_000), start, start + 600_000)
    selected.ingest(window(raw, start + 600_000, start + 620_000), start + 600_000, start + 620_000)
    roi = (0.0, 0.0, 32.0, 24.0)
    torch.testing.assert_close(
        selected.query(roi), oracle(raw, start + 620_000, roi), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="rollback/gap"):
        selected.ingest(events(620_738, 640_738, count=0), 620_739, 640_739)
    with pytest.raises(ValueError, match="ROI"):
        selected.query((float("nan"), 0.0, 32.0, 24.0))
    selected.initialize(events(start, start + 600_000, count=0), start, start + 600_000)
    assert not torch.count_nonzero(selected.query(roi))


@pytest.mark.cuda_fp32
@pytest.mark.skipif(
    os.environ.get("RUN_TRAIN40_COMPILED_PREFIX_CUDA_TESTS") != "1",
    reason="exclusive GPU/compiler slot opt-in",
)
def test_cuda_compiled_selector_graph_matches_eager_through_epoch_transition(tmp_path):
    from scripts.benchmark_train40_transport import configure_isolated_compiler

    configure_isolated_compiler(Path(os.environ["EJEPA_COMPILER_DEPS"]), tmp_path / "compiler")
    raw = events(738, 1_240_738)
    selected = history(device="cuda", normalization="topk", graphs=True)
    baseline = history(TemporalPrefixEventHistory, device="cuda", normalization="topk")
    selected.prime_query_graphs([1, 5, 20])
    for runtime in (selected, baseline):
        runtime.initialize(window(raw, 738, 600_738), 738, 600_738)
    for end in range(600_738, 1_240_739, 20_000):
        if end > 600_738:
            for runtime in (selected, baseline):
                runtime.ingest(window(raw, end - 20_000, end), end - 20_000, end)
        if end not in (600_738, 620_738, 640_738, 1_200_738, 1_220_738):
            continue
        for count in (20, 1, 5):
            rois = [(i * 0.3, i * 0.1, 24.0 + i * 0.2, 20.0 + i * 0.1) for i in range(count)]
            actual = selected.query_many(rois).cpu()
            expected = baseline.query_many(rois).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
