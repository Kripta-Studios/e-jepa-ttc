from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.data.event_v4_geometry import (
    EVENT_V4_CHANNEL_COUNT,
    box_in_common_roi,
    common_square_from_boxes,
    select_active_event_channels,
    shifted_precontext_window,
)


def test_common_roi_preserves_endpoint_scale_ratio() -> None:
    boxes = [
        (40.0, 40.0, 60.0, 60.0),
        (35.0, 35.0, 65.0, 65.0),
        (25.0, 25.0, 75.0, 75.0),
    ]
    square = common_square_from_boxes(boxes, (0, 1, 2), margin_fraction=0.25)
    mapped = [box_in_common_roi(box, square, roi_size=128) for box in boxes]
    source_ratio = (boxes[2][3] - boxes[2][1]) / (boxes[0][3] - boxes[0][1])
    mapped_ratio = (mapped[2][3] - mapped[2][1]) / (mapped[0][3] - mapped[0][1])
    assert np.isclose(mapped_ratio, source_ratio)
    assert mapped[0][3] - mapped[0][1] < mapped[2][3] - mapped[2][1]


def test_active_channel_selection_drops_constant_compatibility_tail() -> None:
    value = torch.randn(21, 8, 8)
    value[12:] = 0.0
    selected = select_active_event_channels(value)
    assert selected.shape == (EVENT_V4_CHANNEL_COUNT, 8, 8)
    assert torch.equal(selected, value[:12])


def test_shifted_precontext_window_is_causal_and_preserves_duration() -> None:
    shifted = shifted_precontext_window((1_000_000, 1_050_000), shift_s=0.1)
    assert shifted == (900_000, 950_000)
    assert shifted[1] <= 1_000_000
    assert shifted[1] - shifted[0] == 50_000
