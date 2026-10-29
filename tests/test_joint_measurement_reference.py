from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from e_jepa_ttc.joint_measurement.model import (
    PairMeasurement,
    bidirectional_multiscale,
    local_cosine_volume,
    pair_objective,
)

torch.set_num_threads(1)


def encoder(c=12):
    return nn.Sequential(
        nn.Conv2d(c, 32, 5, stride=2, padding=2),
        nn.GroupNorm(8, 32),
        nn.GELU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.GroupNorm(8, 64),
        nn.GELU(),
        nn.Dropout2d(0.05),
    )


def model(**kwargs):
    torch.manual_seed(7)
    return PairMeasurement(encoder(kwargs.get("input_channels", 12)), **kwargs)


def test_identical_center_cosine():
    torch.manual_seed(1)
    x = torch.randn(2, 64, 8, 8)
    v, m = local_cosine_volume(x, x)
    torch.testing.assert_close(v[:, 12], torch.ones_like(v[:, 12]), atol=1e-6, rtol=1e-6)
    assert m[:, 12].all()


def test_no_wraparound():
    x = torch.ones(1, 3, 4, 4)
    v, m = local_cosine_volume(x, x)
    assert not m[0, 0, 0, 0] and v[0, 0, 0, 0] == 0
    assert m[0, 24, 0, 0]


def test_volume_preserves_alternatives():
    x = torch.ones(1, 3, 6, 6)
    v, m = local_cosine_volume(x, x)
    assert v.shape == (1, 25, 6, 6)
    assert torch.sum(v[0, :, 3, 3] > 0.99) == 25


@pytest.mark.parametrize("radius", [-1, 5])
def test_bad_radius(radius):
    with pytest.raises(ValueError):
        local_cosine_volume(torch.rand(1, 2, 4, 4), torch.rand(1, 2, 4, 4), radius)


def test_correlations_have_gradients():
    a = torch.randn(1, 8, 6, 6, requires_grad=True)
    b = torch.randn(1, 8, 6, 6, requires_grad=True)
    v, m = local_cosine_volume(a, b)
    v.square().sum().backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0


def test_bidirectional_multiscale_shape():
    x = torch.rand(2, 64, 8, 10)
    v, m = bidirectional_multiscale(x, x)
    assert v.shape == m.shape == (2, 100, 8, 10) and m.dtype == torch.bool


@pytest.mark.parametrize("c", [3, 12])
def test_rgb_event_shapes(c):
    m = model(input_channels=c)
    out = m(torch.randn(2, 2, c, 32, 32), torch.tensor([0.1, 0.09]))
    assert out["raw_phase"].shape == (2,) and torch.isfinite(out["raw_phase"]).all()


def test_gradient_reaches_encoder():
    m = model()
    x = torch.randn(2, 2, 12, 32, 32)
    out = m(x, torch.full((2,), 0.1))
    out["raw_phase"].square().mean().backward()
    assert m.encoder[0].weight.grad.abs().sum() > 0


def test_frozen_stays_eval_and_has_no_gradients():
    m = model(freeze_encoder=True).train()
    assert not m.encoder.training
    before = copy.deepcopy(m.encoder.state_dict())
    m(torch.randn(2, 2, 12, 32, 32), torch.full((2,), 0.1))["raw_phase"].sum().backward()
    assert all(p.grad is None for p in m.encoder.parameters())
    assert all(torch.equal(v, before[k]) for k, v in m.encoder.state_dict().items())


def test_direct_and_corr_same_architecture_initialization():
    a = model(use_correlation=True)
    b = model(use_correlation=False)
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_clock(dt):
    with pytest.raises(ValueError):
        model()(torch.randn(1, 2, 12, 32, 32), torch.tensor([dt]))


def test_rejects_three_frames_not_silent_truncation():
    with pytest.raises(ValueError):
        model()(torch.randn(1, 3, 12, 32, 32), torch.tensor([0.1]))


def test_permutation_equivariance_eval_batch():
    m = model().eval()
    x = torch.randn(3, 2, 12, 32, 32)
    d = torch.tensor([0.1, 0.2, 0.3])
    p = torch.tensor([2, 0, 1])
    with torch.no_grad():
        a = m(x, d)["raw_phase"]
        b = m(x[p], d[p])["raw_phase"]
    torch.testing.assert_close(a[p], b)


def test_serialization():
    m = model().eval()
    state = copy.deepcopy(m.state_dict())
    n = model().eval()
    n.load_state_dict(state)
    x = torch.randn(2, 2, 12, 32, 32)
    d = torch.full((2,), 0.1)
    with torch.no_grad():
        torch.testing.assert_close(m(x, d)["raw_phase"], n(x, d)["raw_phase"], atol=0, rtol=0)


def test_global_mass_not_batch_normalized():
    pred = torch.tensor([0.1, 0.2])
    truth = torch.zeros(2)
    w = torch.tensor([0.01, 0.02])
    loss = pair_objective(pred, truth, w, 100, lambda x: x)
    expected = (100 * w * (1.1 * pred.abs()) / 0.03).mean()
    torch.testing.assert_close(loss, expected)


def test_raw_term_keeps_gradient_in_emission_plateau():
    raw = torch.tensor([0.0], requires_grad=True)
    loss = pair_objective(
        raw, torch.tensor([0.1]), torch.tensor([1.0]), 1, lambda x: torch.zeros_like(x) + 0.002
    )
    loss.backward()
    assert raw.grad.item() < 0


def test_accumulated_mean_gradients():
    a = torch.tensor([0.05, -0.03, 0.07, 0.09], requires_grad=True)
    b = a.detach().clone().requires_grad_()
    truth = torch.tensor([0.08, -0.05, 0.02, 0.1])
    w = torch.tensor([0.1, 0.2, 0.3, 0.4])
    pair_objective(a, truth, w, 4, lambda x: x).backward()
    for sl in (slice(0, 2), slice(2, 4)):
        (0.5 * pair_objective(b[sl], truth[sl], w[sl], 4, lambda x: x)).backward()
    torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)


def test_temporal_difference_error_is_not_absolute_height_error():
    # Constructed illustration, not a dataset finding.
    true = torch.tensor([0.0, 0.1])
    old = true + 1.0
    new = true + torch.tensor([0.1, -0.1])
    assert (new - true).abs().mean() < (old - true).abs().mean()
    assert (new.diff() - true.diff()).abs().item() > (old.diff() - true.diff()).abs().item()
