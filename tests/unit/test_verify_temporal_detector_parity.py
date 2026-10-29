from __future__ import annotations

import torch

from scripts.verify_temporal_detector_parity import _compare_nested


def test_nested_comparison_covers_all_tensors_exactly() -> None:
    value = {
        "pred_logits": torch.tensor([[1.0, 2.0]]),
        "aux_outputs": [{"pred_velocity": torch.tensor([3.0, 4.0])}],
        "inter_outputs": (torch.tensor(5),),
    }
    mismatches, maximum, tensors = _compare_nested(value, value)
    assert mismatches == []
    assert maximum == 0
    assert tensors == 3


def test_nested_comparison_reports_tensor_path_and_maximum() -> None:
    left = {"aux": [{"pred_velocity": torch.tensor([1.0, 2.0])}]}
    right = {"aux": [{"pred_velocity": torch.tensor([1.0, 2.25])}]}
    mismatches, maximum, tensors = _compare_nested(left, right)
    assert mismatches == [
        {"path": "output.aux[0].pred_velocity", "reason": "value", "max": 0.25}
    ]
    assert maximum == 0.25
    assert tensors == 1
