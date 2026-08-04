from __future__ import annotations

import torch

from e_jepa_ttc.models.object_expansion import (
    bounded_log_magnitude,
    hard_direction_sign,
    log_ratio_from_signed_inverse_ttc,
    soft_direction_sign,
)


def test_bounded_log_magnitude_stays_inside_interval() -> None:
    raw = torch.tensor([-100.0, 0.0, 100.0])
    value = bounded_log_magnitude(raw, minimum=0.01, maximum=2.0).exp()
    assert torch.all(value >= 0.01)
    assert torch.all(value <= 2.0)
    assert value[0] < value[1] < value[2]


def test_direction_helpers_use_positive_zero_and_negative_one() -> None:
    logits = torch.tensor([[4.0, -2.0], [-3.0, 5.0]])
    assert torch.equal(hard_direction_sign(logits), torch.tensor([1.0, -1.0]))
    soft = soft_direction_sign(logits)
    assert soft[0] > 0.0
    assert soft[1] < 0.0


def test_log_ratio_matches_lhr_identity() -> None:
    ttc = torch.tensor([2.0, -4.0])
    delta_t = torch.tensor([0.1, 0.1])
    signed_inverse = ttc.reciprocal()
    actual = log_ratio_from_signed_inverse_ttc(
        signed_inverse,
        delta_t,
        epsilon=1.0e-6,
    )
    expected = torch.log1p(-delta_t / ttc)
    assert torch.allclose(actual, expected)
