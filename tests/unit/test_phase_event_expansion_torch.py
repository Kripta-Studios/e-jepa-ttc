"""Parity with the frozen NumPy/SciPy objective, including boundary semantics."""

from itertools import product

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from operational.phase_event_expansion.model import temporal_agreement, warp_endpoint
from operational.phase_event_expansion.torch_objective import (
    CachedTorchObjective,
    SymmetricGaussian,
)


def reference(points, age, polarity, parameters):
    early = age >= (age.min() + age.max()) / 2
    values = []
    for parameter in parameters:
        scale = 1 + parameter[0] * age
        if np.any(scale <= 0):
            values.append(1000 + np.maximum(0, -scale).sum())
        else:
            score, lost = temporal_agreement(
                warp_endpoint(points, age, parameter), polarity, early, 128, 1.0, 0.8
            )
            values.append(-score + 100 * lost)
    return np.asarray(values)


def test_symmetric_gaussian_matches_scipy_at_all_canvas_boundaries():
    image = np.zeros((2, 4, 128, 128))
    image[0, 0, 0, 0] = 1
    image[0, 1, -1, 0] = 2
    image[1, 2, 0, -1] = 3
    image[1, 3, -1, -1] = 4
    actual = SymmetricGaussian(128, dtype=torch.float64)(torch.from_numpy(image)).numpy()
    expected = gaussian_filter(image, sigma=(0, 0, 0.8, 0.8), mode="reflect", truncate=4)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


def test_batched_objective_matches_reference_with_losses_and_invalid_scales():
    rng = np.random.default_rng(7)
    points = rng.uniform(-1.1, 1.1, (256, 2))
    points[:4] = [[-1.0, -1.0], [1.0, 1.0], [-0.999999, 0.999999], [0.999999, -0.999999]]
    age = np.linspace(0, 0.2, len(points))
    polarity = rng.choice([-1, 1], len(points))
    parameters = np.array(list(product((-10.0, 0.0, 10.0), (-5.0, 0.0, 5.0), (-5.0, 0.0, 5.0))))
    model = CachedTorchObjective(points, age, polarity, dtype=torch.float64)
    actual = model(torch.from_numpy(parameters)).numpy()
    expected = reference(points, age, polarity, parameters)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(np.argsort(actual), np.argsort(expected))


def test_no_energy_or_empty_bin_returns_finite_same_reference_objective():
    points = np.full((128, 2), 3.0)
    age = np.zeros(128)
    polarity = np.ones(128)
    parameters = np.zeros((1, 3))
    model = CachedTorchObjective(points, age, polarity, dtype=torch.float64)
    np.testing.assert_allclose(
        model(torch.from_numpy(parameters)).numpy(),
        reference(points, age, polarity, parameters),
        atol=1e-12,
    )


def test_cached_observations_do_not_alias_numpy_inputs():
    points = np.zeros((128, 2))
    age = np.linspace(0, 0.04, 128)
    polarity = np.ones(128)
    model = CachedTorchObjective(points, age, polarity, dtype=torch.float64)
    parameters = torch.zeros((1, 3), dtype=torch.float64)
    before = model(parameters).clone()
    points[:] = 3
    age[:] = 10
    polarity[:] = -1
    assert torch.equal(before, model(parameters))


@pytest.mark.parametrize("count", [0, 20_001])
def test_cache_rejects_exceeding_supported_event_shape(count):
    with pytest.raises(ValueError, match="event"):
        CachedTorchObjective(np.zeros((count, 2)), np.zeros(count), np.ones(count))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_cuda_graph_replays_changed_parameters_with_reference_parity():
    from operational.phase_event_expansion.torch_objective import CudaGraphObjective

    rng = np.random.default_rng(7)
    points = rng.uniform(-1.01, 1.01, (512, 2))
    age = np.linspace(0, 0.2, len(points))
    polarity = rng.choice([-1, 1], len(points))
    model = CachedTorchObjective(points, age, polarity, dtype=torch.float64, device="cuda")
    graph = CudaGraphObjective(model, 3)
    parameters = np.array([[0.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [10.0, -5.0, 5.0]])
    first = graph(torch.tensor(parameters, dtype=torch.float64, device="cuda")).cpu().numpy().copy()
    np.testing.assert_allclose(
        first, reference(points, age, polarity, parameters), atol=1e-10, rtol=0
    )
    parameters[0] = [1.0, 2.0, -1.0]
    second = graph(torch.tensor(parameters, dtype=torch.float64, device="cuda")).cpu().numpy()
    np.testing.assert_allclose(
        second, reference(points, age, polarity, parameters), atol=1e-10, rtol=0
    )
    assert first[0] != second[0]
    assert graph.capture_count == 1
    assert graph.replay_count == 2


def test_order_comparison_reports_material_reversals_and_ties_separately():
    from scripts.benchmark_phase_event_expansion_objective import compare_objectives

    reference_values = np.array([1.0, 1.0 + 1e-8, 2.0])
    tied = compare_objectives(reference_values, np.array([1.0 + 1e-8, 1.0, 2.0]))
    assert tied["within_tolerance"] and tied["material_order_matches"]
    assert not tied["exact_order_matches"]
    wrong = compare_objectives(reference_values, np.array([2.0, 1.0, 1.0]))
    assert not wrong["within_tolerance"] and not wrong["material_order_matches"]
