from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.losses.collision_clock import (
    WEIGHTED_PHASE_REDUCTION,
    normalized_weighted_absolute_phase_error,
)


def test_weighted_loss_is_normalized_by_sum_weights_in_float64() -> None:
    prediction = torch.tensor([1.0, 4.0], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([0.0, 0.0], dtype=torch.float32)
    weight = torch.tensor([1.0, 3.0], dtype=torch.float32)
    loss = normalized_weighted_absolute_phase_error(prediction, target, weight)
    assert WEIGHTED_PHASE_REDUCTION == "normalized_weighted_absolute_phase_error"
    assert loss.dtype == torch.float64
    assert float(loss.detach()) == pytest.approx(13.0 / 4.0)
    incorrect_mean = (weight * prediction.abs()).mean().detach()
    assert float(loss.detach()) != pytest.approx(float(incorrect_mean))
    assert float(loss.detach()) != pytest.approx(13.0)
    loss.backward()
    assert prediction.grad is not None


@pytest.mark.parametrize(
    ("prediction", "target", "weight"),
    [
        ([0.0], [0.0], [0.0]),
        ([0.0], [0.0], [-1.0]),
        ([float("nan")], [0.0], [1.0]),
        ([0.0], [float("inf")], [1.0]),
        ([0.0], [0.0], [float("nan")]),
    ],
)
def test_weighted_loss_fails_closed(
    prediction: list[float], target: list[float], weight: list[float]
) -> None:
    with pytest.raises(ValueError):
        normalized_weighted_absolute_phase_error(
            torch.tensor(prediction),
            torch.tensor(target),
            torch.tensor(weight),
        )
