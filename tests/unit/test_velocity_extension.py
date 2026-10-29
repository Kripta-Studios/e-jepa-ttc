from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.detection.velocity_extension import matched_velocity_l1


def test_matched_velocity_l1_masks_invalid_rows_and_gradients() -> None:
    prediction = torch.tensor([[2.0, -1.0], [9.0, 8.0]], requires_grad=True)
    target = torch.tensor([[1.0, 1.0], [float("nan"), float("nan")]])
    valid = torch.tensor([True, False])

    loss = matched_velocity_l1(prediction, target, valid)
    loss.backward()

    assert loss.item() == pytest.approx(3.0)
    assert prediction.grad is not None
    assert prediction.grad.tolist() == [[1.0, -1.0], [0.0, 0.0]]


def test_matched_velocity_l1_averages_summed_components_per_valid_object() -> None:
    prediction = torch.tensor([[2.0, 4.0], [5.0, 9.0]], requires_grad=True)
    target = torch.tensor([[1.0, 1.0], [3.0, 8.0]])
    valid = torch.tensor([True, True])

    loss = matched_velocity_l1(prediction, target, valid)

    assert loss.item() == pytest.approx((1.0 + 3.0 + 2.0 + 1.0) / 2.0)


def test_matched_velocity_l1_all_invalid_is_differentiable_zero() -> None:
    prediction = torch.tensor([[2.0, -1.0]], requires_grad=True)
    target = torch.tensor([[float("nan"), float("nan")]])

    loss = matched_velocity_l1(prediction, target, torch.tensor([False]))
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert prediction.grad.tolist() == [[0.0, 0.0]]


def test_matched_velocity_l1_accepts_empty_matches() -> None:
    prediction = torch.empty((0, 2), requires_grad=True)
    loss = matched_velocity_l1(
        prediction,
        torch.empty((0, 2)),
        torch.empty((0,), dtype=torch.bool),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert prediction.grad.shape == (0, 2)


@pytest.mark.parametrize(
    ("prediction", "target", "valid", "error"),
    (
        (torch.zeros(2), torch.zeros(2), torch.tensor([True]), ValueError),
        (torch.zeros(1, 2), torch.zeros(2, 2), torch.tensor([True]), ValueError),
        (torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([[True]]), ValueError),
        (torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.long), torch.tensor([True]), TypeError),
        (torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([1.0]), TypeError),
        (
            torch.tensor([[float("inf"), 0.0]]),
            torch.zeros(1, 2),
            torch.tensor([True]),
            ValueError,
        ),
        (
            torch.zeros(1, 2),
            torch.tensor([[float("nan"), 0.0]]),
            torch.tensor([True]),
            ValueError,
        ),
    ),
)
def test_matched_velocity_l1_rejects_invalid_contracts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        matched_velocity_l1(prediction, target, valid)
