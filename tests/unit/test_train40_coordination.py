"""Exact rectangle diagnostics and bulk scalar copies preserve the frozen computation."""

import pytest
import torch

from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps
from e_jepa_ttc.training.causal_scale_eap import _record_relational_fg_bg_diagnostic
from operational.train40_system.coordination import (
    foreground_mask,
    relational_diagnostic,
    scalar_metrics,
)


def reference_mask(boxes, height, width, feat_h, feat_w):
    result = torch.zeros((len(boxes), 2, feat_h, feat_w), dtype=torch.bool)
    for b in range(len(boxes)):
        for ep, index in enumerate([0, 1] if boxes.shape[1] == 2 else [1, 2]):
            if index >= boxes.shape[1]:
                continue
            box = boxes[b, index].float()
            x1 = int(torch.floor(box[0] * (feat_w / float(width))).clamp(0, feat_w).item())
            y1 = int(torch.floor(box[1] * (feat_h / float(height))).clamp(0, feat_h).item())
            x2 = int(torch.ceil(box[2] * (feat_w / float(width))).clamp(0, feat_w).item())
            y2 = int(torch.ceil(box[3] * (feat_h / float(height))).clamp(0, feat_h).item())
            if x2 > x1 and y2 > y1:
                result[b, ep, y1:y2, x1:x2] = True
    return result


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_rectangle_edges_and_padding_are_bit_identical(steps):
    boxes = torch.rand((32, steps, 4), generator=torch.Generator().manual_seed(9)) * 300 - 50
    boxes[0] = torch.tensor([0, 0, 128, 128])
    boxes[1] = torch.tensor([10.25, 20.75, 11.01, 22.0])
    boxes[2] = torch.tensor([float("-inf"), 0, float("inf"), 128])
    assert torch.equal(
        foreground_mask(boxes, 128, 128, 32, 32), reference_mask(boxes, 128, 128, 32, 32)
    )


@pytest.mark.parametrize("empty", [False, True])
def test_all_diagnostic_reductions_equal_original_with_empty_validity(empty):
    features = torch.randn((3, 2, 8, 8, 8), generator=torch.Generator().manual_seed(8))
    relations = local_cosine_relation_maps(features)
    teacher = relations.values + 0.03
    valid = relations.valid if not empty else torch.zeros_like(relations.valid)
    boxes = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [-5.0, 0.1, 127.0, 128.0], [60.5, 32.0, 85.25, 65.0]]] * 3
    )
    expected, actual = {}, {}
    arguments = (features, teacher, valid, boxes, 128, 128, 8, 8)
    _record_relational_fg_bg_diagnostic(*arguments, expected)
    relational_diagnostic(*arguments, actual)
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)


def test_bulk_copy_matches_each_original_scalar_and_preserves_gradients():
    value = torch.tensor(0.125, requires_grad=True)
    components = {"one": value, "two": (value * 2).to(torch.bfloat16)}
    total = value * 3
    actual = scalar_metrics(components, total)
    expected = {key: float(tensor.detach().float().cpu()) for key, tensor in components.items()}
    expected["total"] = float(total.detach().float().cpu())
    assert actual == expected
    assert list(actual) == list(expected)
    total.backward()
    assert value.grad == 3


def test_nan_boxes_remain_rejected():
    with pytest.raises(ValueError, match="NaN"):
        foreground_mask(torch.full((1, 2, 4), float("nan")), 128, 128, 32, 32)
