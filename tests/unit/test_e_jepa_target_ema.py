from __future__ import annotations

import torch
from torch import nn

from e_jepa_ttc.training.jepa import _update_ema


def test_target_encoder_ema_is_exact_and_does_not_build_a_gradient_graph() -> None:
    online = nn.Linear(3, 2)
    target = nn.Linear(3, 2)
    with torch.no_grad():
        online.weight.fill_(2.0)
        online.bias.fill_(4.0)
        target.weight.zero_()
        target.bias.zero_()

    divergence = _update_ema(target, online, momentum=0.75)

    assert torch.allclose(target.weight, torch.full_like(target.weight, 0.5))
    assert torch.allclose(target.bias, torch.full_like(target.bias, 1.0))
    assert divergence > 0.0
    assert all(parameter.grad is None for parameter in target.parameters())


def test_target_ema_rejects_invalid_momentum() -> None:
    with torch.no_grad():
        online = nn.Linear(2, 2)
        target = nn.Linear(2, 2)
    try:
        _update_ema(target, online, momentum=1.1)
    except ValueError as error:
        assert "momentum" in str(error)
    else:
        raise AssertionError("Invalid EMA momentum was accepted.")
