from __future__ import annotations

import numpy as np
import pytest

from e_jepa_ttc.data.garl_official_preprocessing import (
    official_resize_feature,
    official_square_box,
    official_timevolume_roi_np,
)


def test_official_timevolume_has_reference_shape_counts_and_determinism() -> None:
    x = np.asarray([1, 2, 8, 20], dtype=np.int32)
    y = np.asarray([1, 2, 8, 20], dtype=np.int32)
    t_us = np.asarray([0, 20_000, 50_000, 120_000], dtype=np.int64)
    first, counts = official_timevolume_roi_np((0, 0, 16, 16), x, y, t_us)
    second, repeated_counts = official_timevolume_roi_np((0, 0, 16, 16), x, y, t_us)
    assert first.shape == (20, 16, 16)
    assert int(counts.sum()) == 3
    np.testing.assert_array_equal(counts, repeated_counts)
    np.testing.assert_array_equal(first, second)


def test_resize_and_square_box_follow_explicit_bounds() -> None:
    feature = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    resized = official_resize_feature(feature, (8, 10))
    assert tuple(resized.shape) == (2, 8, 10)
    assert official_square_box([(10.0, 20.0, 30.0, 40.0), (10.0, 20.0, 50.0, 35.0)], 0) == (
        0,
        10,
        40,
        50,
    )
    with pytest.raises(ValueError, match="feature"):
        official_resize_feature(feature[0], (2, 2))
