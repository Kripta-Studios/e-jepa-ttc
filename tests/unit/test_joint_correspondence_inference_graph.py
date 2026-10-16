"""Inference acceleration must retain original batch, clocks and precision."""

import pytest
import torch

from operational.joint_measurement.compute_guard import is_mlflow_mcp
from operational.joint_measurement.correspondence_inference_graph import validate_inputs


@pytest.mark.parametrize("batch", [4, 2])
def test_registered_full_and_tail_batches(batch: int) -> None:
    validate_inputs(torch.zeros(batch, 2, 12, 128, 128), torch.ones(batch))


@pytest.mark.parametrize("batch", [1, 3, 8])
def test_no_batch_expansion_or_padding(batch: int) -> None:
    with pytest.raises(ValueError, match="batches4"):
        validate_inputs(torch.zeros(batch, 2, 12, 128, 128), torch.ones(batch))


def test_clock_layout_and_precision_cannot_change() -> None:
    pixels = torch.zeros(4, 2, 12, 128, 128)
    with pytest.raises(ValueError, match="FP32"):
        validate_inputs(pixels, torch.ones(4, 1))
    with pytest.raises(ValueError, match="FP32"):
        validate_inputs(pixels.half(), torch.ones(4))


def test_only_known_control_plane_is_exempt() -> None:
    assert is_mlflow_mcp(["python.exe", "mlflow.exe", "mcp", "run"])
    assert not is_mlflow_mcp(["python.exe", "mlflow.exe", "run", "training"])
    assert not is_mlflow_mcp(["python.exe", "train.py", "mcp", "run"])
    assert not is_mlflow_mcp(["python.exe", "mlflow.exe", "mcp", "run", "extra"])
