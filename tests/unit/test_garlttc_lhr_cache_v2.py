from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.data.garlttc_lhr_cache import (
    OBSERVABLE_MOTION_DIM,
    _box_features,
    observable_motion_from_boxes_torch,
)


def test_observable_motion_has_no_ttc_or_depth_dependency() -> None:
    first = (400.0, 200.0, 500.0, 400.0)
    second = (430.0, 200.0, 530.0, 410.0)
    values, metadata = _box_features(first, second, 0.1)
    assert values.shape == (OBSERVABLE_MOTION_DIM,)
    assert np.isfinite(values).all()
    assert metadata["lateral_speed_raw"] > 0.0


def test_torch_observable_motion_accepts_normalized_boxes() -> None:
    boxes = torch.tensor(
        [
            [
                [0.30, 0.30, 0.40, 0.60],
                [0.32, 0.30, 0.42, 0.61],
            ]
        ],
        dtype=torch.float32,
    )
    motion = observable_motion_from_boxes_torch(boxes, torch.tensor([0.1]))
    assert motion.shape == (1, OBSERVABLE_MOTION_DIM)
    assert torch.isfinite(motion).all()
