"""The partial-depth contrast must change gradients, never the initial forward."""

import copy

import pytest
import torch
from torch import nn

from operational.joint_measurement import correspondence_train as original
from operational.joint_measurement.adaptation_depth import REGISTRATION, configure, engine_context


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.2))

    def forward(self, x):
        return x + self.network(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 4),
            nn.Identity(),
            nn.Identity(),
            nn.Linear(4, 4),
            nn.Identity(),
            nn.Identity(),
            Residual(),
        )
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.encoder(x))


def test_same_forward_and_dropout_but_different_gradient_boundary():
    full = Model()
    partial = copy.deepcopy(full)
    configure(full, "FULL")
    policy = configure(partial, "LAST-BLOCK")
    x = torch.randn(3, 4)
    torch.manual_seed(7)
    a = full(x)
    torch.manual_seed(7)
    b = partial(x)
    assert torch.equal(a, b)
    a.sum().backward()
    b.sum().backward()
    assert all(p.grad is not None for p in full.parameters())
    assert all(p.grad is None for n, p in partial.named_parameters() if n in policy["frozen"])
    assert all(
        p.grad is not None for n, p in partial.named_parameters() if n in policy["trainable"]
    )
    assert partial.encoder[6].network[1].training


def test_context_restores_original_registration_even_on_error():
    previous = original.REGISTRATION
    with pytest.raises(RuntimeError):
        with engine_context():
            assert original.REGISTRATION == REGISTRATION
            raise RuntimeError("test")
    assert original.REGISTRATION == previous


def test_running_stat_normalization_is_not_an_implicit_second_factor():
    model = Model()
    model.encoder[1] = nn.BatchNorm1d(4)
    with pytest.raises(ValueError, match="Running-stat"):
        configure(model, "LAST-BLOCK")
