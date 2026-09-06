"""Basic boundary tests, not a substitute for real V4 input replay parity."""

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.query_context_voxel import encode_query_window


def encode(raw):
    return encode_query_window(
        raw,
        square_xyxy=(0.0, 0.0, 64.0, 64.0),
        start_us=0,
        end_us=100,
        sequence_id="fixture",
        roi_size=64,
        bins_per_polarity=5,
        event_pixel_diff=5.0,
    )


def test_crop_count_rate_and_empty_inputs():
    raw = {
        "x": np.array([0, 59]),
        "y": np.array([0, 0]),
        "t": np.array([0, 99]),
        "p": np.array([0, 1]),
    }
    result = encode(raw)
    assert result.shape == (12, 64, 64)
    assert np.isclose(float(result[10, 0, 0]), np.log1p(1))
    assert np.isclose(float(result[11, 0, 0]), np.log1p(10000))
    assert encode({key: value[:0] for key, value in raw.items()}).count_nonzero() == 0


def test_reject_future_and_privileged_fields():
    raw = {"x": np.array([0]), "y": np.array([0]), "t": np.array([100]), "p": np.array([1])}
    with pytest.raises(ValueError, match="dependency"):
        encode(raw)
    with pytest.raises(ValueError, match="sensor columns"):
        encode({**raw, "ttc": np.array([4.0])})
