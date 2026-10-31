"""Full solver parity and bounded mutable CUDA-cache behavior."""

import numpy as np
import pytest
import torch

from operational.phase_event_expansion.accelerated import CudaObjectiveCache, estimate_accelerated
from operational.phase_event_expansion.model import SolverConfig, estimate_expansion
from operational.phase_event_expansion.synthetic import make_fixture


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_full_solver_matches_cpu_on_development_expansion():
    f = make_fixture(7, "expansion")
    config = SolverConfig(objective="temporal_ncc")
    expected = estimate_expansion(f.events, f.roi, f.query_us, config)
    cache = CudaObjectiveCache()
    actual = estimate_accelerated(f.events, f.roi, f.query_us, config, cache)
    assert (actual.valid, actual.reason) == (expected.valid, expected.reason)
    np.testing.assert_allclose(
        [actual.q, *actual.translation], [expected.q, *expected.translation], rtol=0, atol=1e-5
    )
    assert cache.capture_count == 2
    assert cache.replay_count > 3
    cache.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_same_shape_cache_reuses_graphs_and_reloads_new_observations():
    first = make_fixture(7, "expansion", count=512)
    second = make_fixture(7, "receding", count=512)
    cache = CudaObjectiveCache()
    p = np.array([[3.0, 0.2, 0.1]])
    cache.prepare(
        (first.events[:, :2] - 128) / 256,
        (first.query_us - first.events[:, 2]) / 1e6,
        first.events[:, 3],
    )
    one = cache.evaluate(p)
    cache.prepare(
        (second.events[:, :2] - 128) / 256,
        (second.query_us - second.events[:, 2]) / 1e6,
        second.events[:, 3],
    )
    two = cache.evaluate(p)
    assert not np.array_equal(one, two)
    assert cache.capture_count == 2
    assert cache.cached_shapes == 1
    cache.prepare(np.zeros((256, 2)), np.linspace(0, 0.04, 256), np.ones(256))
    assert cache.capture_count == 4
    assert cache.cached_shapes == 1
    cache.close()
    assert cache.cached_shapes == 0


def test_cpu_backend_is_explicit_and_preserves_original_result():
    f = make_fixture(7, "edge")
    config = SolverConfig(objective="temporal_ncc")
    result = estimate_accelerated(f.events, f.roi, f.query_us, config, backend="cpu")
    assert result.reason == "insufficient_spatial_support"
    assert result.diagnostics["execution_backend"] == "cpu_explicit"


def test_full_solver_comparison_does_not_hide_optimizer_branch_divergence():
    from operational.phase_event_expansion.model import ExpansionResult
    from scripts.benchmark_phase_event_expansion_accelerated import compare_results

    cpu = ExpansionResult(
        2.0, (0.1, 0.2), 0.5, True, "available", 0.1, {"objective_evaluations": 400}
    )
    changed = ExpansionResult(
        2.0, (0.1, 0.2), None, False, "optimizer_failure", 0.1, {"objective_evaluations": 401}
    )
    assert not compare_results(cpu, changed)["passed"]
    assert not compare_results(cpu, changed)["objective_count_matches"]
