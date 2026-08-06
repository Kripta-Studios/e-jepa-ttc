from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_2 import ObjectEventV42Output
from e_jepa_ttc.training.object_event_v4_5 import (
    ObjectEventV45LossConfig,
    log_eta,
    object_event_v4_5_loss,
    range_balanced_log_eta_loss,
    reciprocal_log_eta_error,
    reciprocal_reverse_target,
)


def _output(forward: torch.Tensor, reverse: torch.Tensor) -> ObjectEventV42Output:
    maximum = 0.25
    raw = torch.atanh((forward / maximum).clamp(-0.999, 0.999))
    reverse_raw = torch.atanh((reverse / maximum).clamp(-0.999, 0.999))
    batch = forward.shape[0]
    return ObjectEventV42Output(
        expansion=forward,
        reverse_expansion=reverse,
        raw_score=raw,
        reverse_raw_score=reverse_raw,
        reversal_consistency_error=(forward + reverse).abs(),
        endpoint_embeddings=torch.zeros(batch, 3, 2),
        spatial_embeddings=torch.zeros(batch, 3, 2),
    )


def test_exact_reciprocal_reverse_is_inverse_in_log_eta() -> None:
    forward = torch.tensor([-0.15, -0.05, 0.0, 0.05, 0.15], dtype=torch.float64)
    reverse = reciprocal_reverse_target(forward, maximum=0.25)
    assert torch.allclose(
        log_eta(forward, maximum=0.25) + log_eta(reverse, maximum=0.25),
        torch.zeros_like(log_eta(forward, maximum=0.25)),
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    assert not torch.allclose(reverse, -forward, atol=1.0e-3, rtol=0.0)


def test_reciprocal_consistency_is_zero_for_exact_pair() -> None:
    forward = torch.tensor([-0.15, -0.03, 0.04, 0.18])
    reverse = reciprocal_reverse_target(forward, maximum=0.25)
    value = reciprocal_log_eta_error(forward, reverse, maximum=0.25)
    assert float(value) < 1.0e-6


def test_range_balanced_mid_uses_range_means_not_class_counts() -> None:
    target = torch.zeros(6)
    prediction = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.04, -0.02])
    ttc = torch.tensor([2.0, 2.0, 2.0, 2.0, 5.0, -5.0])
    actual = range_balanced_log_eta_loss(
        prediction,
        target,
        ttc,
        maximum=0.25,
        range_weights=(0.5, 0.3, 0.1, 0.1),
    )
    errors = (log_eta(prediction, maximum=0.25) - log_eta(target, maximum=0.25)).abs()
    expected = (0.5 * errors[:4].mean() + 0.3 * errors[4] + 0.1 * errors[5]) / 0.9
    assert torch.allclose(actual, expected)


def test_full_loss_is_finite_and_backpropagates_through_both_orders() -> None:
    delta_t = torch.full((8,), 0.1)
    ttc = torch.tensor([2.0, 2.5, 4.0, 5.5, 7.0, 9.0, -3.0, -7.0])
    target = (delta_t / ttc).clamp(-0.24975, 0.24975)
    forward = (target + 0.003).detach().clone().requires_grad_(True)
    reverse = (
        reciprocal_reverse_target(target, maximum=0.25) - 0.002
    ).detach().clone().requires_grad_(True)
    loss = object_event_v4_5_loss(
        _output(forward, reverse),
        delta_t,
        ttc,
        config=ObjectEventV45LossConfig(),
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert forward.grad is not None and torch.isfinite(forward.grad).all()
    assert reverse.grad is not None and torch.isfinite(reverse.grad).all()
