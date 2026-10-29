"""V15 sampler changes draws while preserving the native expected risk."""

import numpy as np
import torch

from e_jepa_ttc.simplex_t.model import training_loss
from operational.range_readout.qa import synthetic
from operational.range_readout.train_heads import initialize, loss_on_batch
from operational.range_transfer.cache import HeadCache


def test_sampler_expected_native_loss_and_gradient(tmp_path):
    path = tmp_path / "H.npz"
    synthetic(path)
    cache = HeadCache(path)
    model, _, _, _ = initialize(7)
    model.eval()
    ids = torch.arange(cache.population)
    for mass_id in (0, 1):
        weights = torch.from_numpy(cache.plans[mass_id].copy())
        p = weights.double() / weights.double().sum()
        output = cache.forward(model, ids)
        target = torch.from_numpy(cache.values["target_phase"].copy())
        experts = torch.from_numpy(cache.values["expert_phase"].copy())
        exact = training_loss(output, target, experts, weights, cache.population)
        individual = []
        for i in range(cache.population):
            item = {k: v[i : i + 1] for k, v in output.items()}
            individual.append(
                training_loss(
                    item,
                    target[i : i + 1],
                    experts[i : i + 1],
                    (weights[i : i + 1].double() / p[i]).float(),
                    1,
                )
            )
        expectation = (torch.stack(individual).double() * p).sum()
        torch.testing.assert_close(exact.double(), expectation, atol=2e-6, rtol=2e-6)
        a = torch.autograd.grad(exact, tuple(model.parameters()), retain_graph=True)
        b = torch.autograd.grad(expectation, tuple(model.parameters()))
        for left, right in zip(a, b, strict=True):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


def test_sampled_loss_has_no_second_mass_weight(tmp_path):
    path = tmp_path / "H.npz"
    synthetic(path)
    cache = HeadCache(path)
    model, _, _, generator = initialize(7)
    model.eval()
    p = torch.from_numpy(cache.plans[1].astype(np.float64))
    p /= p.sum()
    ids = torch.multinomial(p, 128, replacement=True, generator=generator)
    actual = loss_on_batch(model, cache, ids, 1, 0)
    expected = training_loss(
        cache.forward(model, ids),
        cache.tensor("target_phase", ids, torch.device("cpu")),
        cache.tensor("expert_phase", ids, torch.device("cpu")),
        torch.ones(128),
        1,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
