"""The capture candidate must preserve training arithmetic and failure guards."""

import copy
from dataclasses import fields

import pytest
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from operational.joint_measurement.geometry_capture import (
    CaptureCore,
    affine_tensor,
    checked_phase,
    contiguous_microbatch,
    contiguous_subset,
    owned_backward,
)
from operational.joint_measurement.geometry_model import GeometricPairMeasurement, weighted_affine
from operational.joint_measurement.qa import exact, fixture, subset
from operational.joint_measurement.train import backward


@pytest.mark.parametrize("seed", [7, 23])
def test_capture_core_same_phase_gradients_rng_and_parameter_identity(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    original = GeometricPairMeasurement(CausalScaleTTC()).train()
    candidate = copy.deepcopy(original)
    core = CaptureCore(candidate)
    assert {k: id(p) for k, p in core.named_parameters()} == {
        k: id(p) for k, p in candidate.named_parameters()
    }
    inputs = torch.randn(2, 2, 12, 128, 128)
    dt = torch.tensor([0.05, 0.1])
    state = torch.get_rng_state()
    baseline = original(inputs, dt)["raw_phase"]
    baseline.square().sum().backward()
    final_rng = torch.get_rng_state()
    torch.set_rng_state(state)
    result = checked_phase(core(inputs, dt))["raw_phase"]
    result.square().sum().backward()
    assert torch.equal(baseline, result)
    assert torch.equal(final_rng, torch.get_rng_state())
    for (name, left), (right_name, right) in zip(
        original.named_parameters(), core.named_parameters(), strict=True
    ):
        assert name == right_name
        assert exact(left.grad, right.grad), name


def test_affine_tensor_same_matrix_and_gradients_with_nonuniform_support():
    torch.manual_seed(13)
    flow = torch.randn(2, 2, 8, 8, requires_grad=True)
    weights = torch.rand(2, 8, 8, requires_grad=True)
    baseline, _ = weighted_affine(flow, weights)
    old_grads = torch.autograd.grad(baseline.square().sum(), (flow, weights))
    candidate, valid = affine_tensor(flow, weights)
    new_grads = torch.autograd.grad(candidate.square().sum(), (flow, weights))
    assert valid
    assert torch.equal(baseline, candidate)
    assert all(torch.equal(a, b) for a, b in zip(old_grads, new_grads, strict=True))


@pytest.mark.parametrize("invalid", [0.0, -0.1, float("inf"), float("nan")])
def test_invalid_native_time_rejected_before_loss_or_optimizer(invalid):
    torch.set_num_threads(1)
    core = CaptureCore(GeometricPairMeasurement(CausalScaleTTC()))
    with pytest.raises(ValueError, match="finite/domain"):
        checked_phase(core(torch.zeros(1, 2, 12, 128, 128), torch.tensor([invalid])))
    assert all(p.grad is None for p in core.parameters())


def test_invalid_affine_weights_preserve_original_rejection():
    flow = torch.zeros(1, 2, 8, 8)
    weights = -torch.ones(1, 8, 8)
    _, valid = affine_tensor(flow, weights)
    assert not valid
    with pytest.raises(ValueError):
        weighted_affine(flow, weights)


def test_contiguous_microbatch_same_fields_and_shared_tensor_storage():
    batch = fixture(128)
    actual = contiguous_microbatch(batch, 2, 6)
    expected = subset(batch, [2, 3, 4, 5])
    for field in fields(batch):
        value = getattr(actual, field.name)
        assert exact(value, getattr(expected, field.name)), field.name
        if isinstance(value, torch.Tensor):
            parent = getattr(batch, field.name)
            assert value.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()


@pytest.mark.parametrize("start,stop", [(-1, 2), (0, 0), (3, 2), (0, 9)])
def test_microbatch_invalid_bounds_fail(start, stop):
    with pytest.raises(ValueError):
        contiguous_microbatch(fixture(128), start, stop)


def test_arbitrary_sampler_ids_keep_duplicates_and_order():
    batch = fixture(128)
    expected = subset(batch, [4, 2, 4, 0])
    actual = contiguous_subset(batch, [4, 2, 4, 0])
    for field in fields(batch):
        assert exact(getattr(actual, field.name), getattr(expected, field.name))


def test_owned_gradient_accumulation_matches_original_multiple_microbatches():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    original = GeometricPairMeasurement(CausalScaleTTC()).train()
    candidate = copy.deepcopy(original)
    batch = fixture(128)
    parts = [contiguous_microbatch(batch, 0, 4), contiguous_microbatch(batch, 4, 8)]
    weights = torch.full((8,), 1 / 8)
    rng = torch.get_rng_state()
    baseline = backward(original, parts, weights, 8)
    final_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual = owned_backward(candidate, parts, weights, 8)
    assert exact(actual, baseline)
    assert torch.equal(final_rng, torch.get_rng_state())
    for left, right in zip(original.parameters(), candidate.parameters(), strict=True):
        assert exact(left.grad, right.grad)
