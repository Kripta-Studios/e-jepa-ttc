"""Preserve exact phase terms, derivatives and original invalid-input rejection."""

import pytest
import torch

from e_jepa_ttc.joint_measurement.loss import phase_components
from e_jepa_ttc.simplex_t.phase import ttc_to_phase
from operational.joint_measurement.phase_transport_loss import phase_terms


def test_terms_and_gradients_exact():
    raw = torch.tensor([-9.0, -0.7, -0.69, -0.01, 0.0, 0.01, 1.0, 8.0, 12.0], requires_grad=True)
    truth = torch.tensor([-10.0, -2.0, -0.1, 0.10001, 0.5, 1.0, 3.0, 6.0, 10.0])
    reference = phase_components(raw, ttc_to_phase(truth))
    candidate, valid = phase_terms(raw, truth)
    assert bool(valid)
    assert list(candidate) == list(reference)
    for key in reference:
        assert torch.equal(reference[key], candidate[key])
        a = torch.autograd.grad(reference[key].sum(), raw, retain_graph=True)[0]
        b = torch.autograd.grad(candidate[key].sum(), raw, retain_graph=True)[0]
        assert torch.equal(a, b)


@pytest.mark.parametrize("truth", [0.0, 0.1, 0.05, float("nan"), float("inf")])
def test_invalid_truth_rejected(truth):
    target = torch.tensor([truth])
    with pytest.raises(ValueError):
        ttc_to_phase(target)
    _, valid = phase_terms(torch.tensor([0.1]), target)
    assert not bool(valid)


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), -float("inf")])
def test_invalid_phase_rejected(raw):
    phase, target = torch.tensor([raw]), torch.tensor([1.0])
    with pytest.raises(ValueError):
        phase_components(phase, ttc_to_phase(target))
    _, valid = phase_terms(phase, target)
    assert not bool(valid)
