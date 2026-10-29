"""Exact input arithmetic checks for the additive TRAIN40 encoding helpers."""

import numpy as np
import pytest
import torch

from e_jepa_ttc.efficient_context.mapped_union import encode_mapped, map_roi
from e_jepa_ttc.rgb_port.data import crop_uint8
from operational.train40_runtime.encoding import (
    crop_uint8_roi_first,
    encode_mapped_lut,
    map_roi_lut,
)


@pytest.mark.parametrize("offset", [0.0, 1.25, -2.0])
@pytest.mark.parametrize("square", [(0.0, 0.0, 1280.0, 720.0), (1.25, -5.5, 800.75, 830.0)])
def test_coordinate_lut_exact_with_sensor_bounds_and_offsets(offset, square):
    rng = np.random.default_rng(31)
    raw = {
        "x": rng.integers(-4, 1285, 1000, dtype=np.int32),
        "y": rng.integers(-4, 725, 1000, dtype=np.int32),
        "t": np.arange(1000, dtype=np.int64),
        "p": rng.integers(-1, 2, 1000, dtype=np.int8),
    }
    expected = map_roi(raw, square, 128, offset)
    actual = map_roi_lut(raw, square, 128, offset)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
        assert actual[name].dtype == expected[name].dtype


def test_coordinate_lut_rejects_rollback():
    raw = {"x": np.array([2, 3]), "y": np.array([2, 3]), "t": np.array([2, 1]), "p": np.ones(2)}
    with pytest.raises(ValueError, match="rollback"):
        map_roi_lut(raw, (0.0, 0.0, 128.0, 128.0), 128)


@pytest.mark.parametrize("count", [0, 1, 100_000])
@pytest.mark.parametrize("duration", [100_000, 99_999])
def test_temporal_lut_preserves_bincount_and_quantile(count, duration):
    rng = np.random.default_rng(19)
    start = 20_200_738
    elapsed = rng.integers(0, duration, count, dtype=np.int64)
    if count > 10:
        elapsed[:6] = [0, 1, 24999, 25000, 50000, duration - 1]
    mapped = {
        "x": rng.integers(0, 128, count, dtype=np.int32),
        "y": rng.integers(0, 128, count, dtype=np.int32),
        "t": np.sort(elapsed) + start,
        "p": rng.choice(np.array([-1, 1], dtype=np.int8), count),
    }
    expected = encode_mapped(mapped, start, start + duration, 128, "fixture")
    actual = encode_mapped_lut(mapped, start, start + duration, 128, "fixture")
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("square", [(2.1, 3.9, 40.6, 52.9), (-3.9, -10.5, 128.0, 190.0)])
def test_crop_first_retains_pil_pixels(square):
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (180, 120, 3), dtype=np.uint8)
    expected = crop_uint8(frame, square)
    actual = crop_uint8_roi_first(frame, square)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
