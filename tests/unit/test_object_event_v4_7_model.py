from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config
from e_jepa_ttc.models.object_event_v4_7 import ObjectEventTTCV47, ObjectEventV47Config


def _rectangle(height: int, *, size: int = 64) -> torch.Tensor:
    mask = torch.zeros(1, 1, size, size)
    top = (size - height) // 2
    mask[:, :, top : top + height, 16:48] = 1.0
    return mask


def test_soft_extent_increases_with_rectangle_height() -> None:
    small = _rectangle(16)
    large = _rectangle(32)
    _, _, _, small_extent = ObjectEventTTCV47.soft_vertical_extent(
        small, edge_temperature=0.08, floor=1.0e-4
    )
    _, _, _, large_extent = ObjectEventTTCV47.soft_vertical_extent(
        large, edge_temperature=0.08, floor=1.0e-4
    )
    assert float(large_extent) > float(small_extent)


def test_v47_forward_is_event_only_and_has_highres_masks() -> None:
    model = ObjectEventTTCV47(
        ObjectEventV41Config(dropout=0.0),
        ObjectEventV47Config(mask_size=64),
    )
    output = model(torch.randn(2, 3, 12, 64, 64))
    assert output.expansion.shape == (2,)
    assert output.foreground_logits.shape == (2, 3, 64, 64)
    assert output.predicted_log_heights.shape == (2, 3)
    assert torch.isfinite(output.expansion).all()
    assert torch.allclose(
        output.height_log_eta + (-output.height_log_eta),
        torch.zeros_like(output.height_log_eta),
    )
