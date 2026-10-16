"""CPU arithmetic, aggregation and lifetime contracts; actual CUDA admission is separate."""

import copy
from types import SimpleNamespace

import pytest
import torch

from operational.joint_measurement import captured_phase_loss as candidate
from operational.joint_measurement import phase_transport_loss as reference


def test_ordered_weighted_loss_and_derivative_are_exact_on_boundaries():
    raw = torch.tensor([-9.0, -0.7, -0.69, -0.01, 0.0, 0.01, 1.0, 8.0, 12.0], requires_grad=True)
    truth = torch.tensor([-10.0, -2.0, -0.1, 0.10001, 0.5, 1.0, 3.0, 6.0, 10.0])
    weights = torch.linspace(0.001, 0.009, 9)
    terms, valid = reference.phase_terms(raw, truth)
    scaled = 53890 * weights
    expected = ((terms["emitted"] + terms["raw"]) * scaled).sum() / 32
    diagnostics = torch.stack(
        [expected.detach(), *[(value.detach() * scaled).sum() / 32 for value in terms.values()]]
    )
    loss, observed, guard = candidate.LossCore(53890, 32)(raw, truth, weights)
    assert bool(valid & guard)
    assert torch.equal(loss, expected) and torch.equal(diagnostics, observed)
    assert torch.equal(torch.autograd.grad(expected, raw)[0], torch.autograd.grad(loss, raw)[0])
    assert not observed.requires_grad


@pytest.mark.parametrize(
    ("raw_value", "target"),
    [
        (0.1, 0.1),
        (0.1, 0.0),
        (0.1, float("nan")),
        (float("inf"), 1.0),
        (float("nan"), -1.0),
    ],
)
def test_guard_rejects_before_backward(raw_value, target):
    raw = torch.tensor([raw_value], requires_grad=True)
    with pytest.raises(FloatingPointError):
        candidate.checked(candidate.LossCore(1, 1)(raw, torch.tensor([target]), torch.ones(1)))
    assert raw.grad is None


def test_diagnostic_owns_storage_across_replay():
    storage = torch.arange(4, dtype=torch.float32)
    _, saved = candidate.checked((torch.tensor(2.0), storage, torch.tensor(True)))
    storage.fill_(-999)
    assert torch.equal(saved, torch.arange(4, dtype=torch.float32))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.04]))
        self._v18_capture_active = True

    def forward(self, events, dt):
        return {"raw_phase": events[:, 0] * self.weight}


class EagerTestLoss:
    def __init__(self, raw, truth, weights, population, batch_size):
        self.population, self.batch_size = population, batch_size
        self.core = candidate.LossCore(population, batch_size)

    def __call__(self, raw, truth, weights):
        return candidate.checked(self.core(raw, truth, weights))


def test_microbatch_aggregation_gradients_and_parameter_topology_exact(monkeypatch):
    monkeypatch.setattr(candidate, "CapturedLoss", EagerTestLoss)
    batches = [
        SimpleNamespace(
            events=torch.tensor([[-2.0], [1.0], [0.0], [1.5]]) + i / 10,
            delta_t_s=torch.ones(4) * 0.04,
            target_ttc_s=torch.tensor([-1.0, 1.0, 10.0, 0.2]),
        )
        for i in range(8)
    ]
    weights = torch.linspace(0.0001, 0.0032, 32)
    original, model = Model(), Model()
    before = copy.deepcopy(model.state_dict())
    ids = [id(p) for p in model.parameters()]
    expected = reference.backward(original, batches, weights, 53890)
    observed = candidate.backward(model, batches, weights, 53890)
    assert observed == expected and torch.equal(model.weight.grad, original.weight.grad)
    assert ids == [id(p) for p in model.parameters()]
    assert set(model.state_dict()) == set(before)
    assert torch.equal(model.weight, before["weight"])
    loss_object = model._v18_phase_loss
    model.zero_grad(set_to_none=True)
    assert candidate.backward(model, batches, weights, 53890) == expected
    assert model._v18_phase_loss is loss_object
    with pytest.raises(ValueError, match="population"):
        candidate.backward(model, batches, weights, 12)


def test_cpu_is_not_admitted_as_cuda_capture():
    with pytest.raises(ValueError, match="CUDA FP32"):
        candidate.CapturedLoss(
            torch.ones(4, requires_grad=True), torch.ones(4), torch.ones(4), 8, 32
        )


def test_signature_checks_differentiability_and_dtype():
    args = (torch.ones(4, requires_grad=True), torch.ones(4), torch.ones(4))
    assert candidate.signature(args) != candidate.signature((args[0].detach(), *args[1:]))
    assert candidate.signature(args) != candidate.signature((args[0].double(), *args[1:]))
