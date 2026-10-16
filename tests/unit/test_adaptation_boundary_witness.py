"""A missing or stale scope witness must never hide weight-contract violations."""

import pytest
import torch

from operational.joint_measurement.adaptation_observed_run import differences


def test_complete_weight_differences() -> None:
    parent = {"frozen": torch.tensor([1.0]), "trainable": torch.tensor([2.0])}
    current = {"frozen": torch.tensor([1.0]), "trainable": torch.tensor([3.0])}
    assert differences(current, parent) == {"frozen": 0.0, "trainable": 1.0}
    assert parent["trainable"].item() == 2.0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_weights_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="Nonfinite"):
        differences({"a": torch.tensor([value])}, {"a": torch.tensor([0.0])})


def test_missing_weight_rejected() -> None:
    with pytest.raises(ValueError, match="keys"):
        differences({}, {"a": torch.tensor([0.0])})


@pytest.mark.parametrize("value", [torch.zeros(2), torch.zeros(1, dtype=torch.float64)])
def test_tensor_contract_rejected(value: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="contract"):
        differences({"a": value}, {"a": torch.zeros(1)})
