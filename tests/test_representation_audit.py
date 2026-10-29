"""Contracts for the prospectively frozen bypass intervention."""

import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner, training_loss
from operational.representation_audit.model import RepresentationRefiner


def inputs():
    generator = torch.Generator().manual_seed(19)
    return (
        torch.randn(6, 8, 17, generator=generator),
        torch.randn(6, 8, 4, generator=generator),
        torch.ones(6, 8, dtype=torch.bool),
        torch.rand(6, 3, generator=generator) * 0.1,
        torch.randn(6, 896, generator=generator),
    )


@pytest.mark.parametrize("active", [False, True])
def test_initial_bypass_matches_native_outputs_and_native_gradients(active):
    torch.manual_seed(7)
    native = TemporalRefiner(TemporalConfig(feature_count=17, hidden=160, backbone="gru"))
    torch.manual_seed(7)
    model = RepresentationRefiner(active)
    batch = inputs()
    left = native(*batch[:4])
    right = model(*batch)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    truth = torch.full((6,), 0.15)
    mass = torch.full((6,), 1 / 6)
    training_loss(left, truth, batch[3], mass, 6).backward()
    training_loss(right, truth, batch[3], mass, 6).backward()
    for name, param in native.named_parameters():
        torch.testing.assert_close(
            param.grad, dict(model.named_parameters())[name].grad, rtol=0, atol=0
        )


def test_control_ignores_side_even_when_bypass_parameters_are_nonzero():
    model = RepresentationRefiner(False)
    last = model.bypass[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        last.weight.fill_(0.03)
        model.location.weight.fill_(0.01)
    batch = inputs()
    a = model(*batch)
    b = model(*batch[:4], batch[4] * 100)
    assert torch.equal(a["raw_location"], b["raw_location"])


def test_active_bypass_can_use_side():
    model = RepresentationRefiner(True)
    last = model.bypass[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        last.weight.normal_(std=0.1)
        model.location.weight.normal_(std=0.1)
    batch = inputs()
    a = model(*batch)
    b = model(*batch[:4], -batch[4])
    assert not torch.equal(a["raw_location"], b["raw_location"])


def test_inference_rejects_targets_and_bad_dimension():
    model = RepresentationRefiner(True)
    batch = inputs()
    with pytest.raises(TypeError):
        model(*batch, target_phase=torch.ones(6))
    with pytest.raises(ValueError):
        model(*batch[:4], batch[4][:, :20])
