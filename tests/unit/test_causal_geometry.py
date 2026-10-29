import numpy as np

from e_jepa_ttc.baselines.causal_geometry import (
    _causal_scale_derivative,
    _hybrid_geometry_with_fallback,
)


def test_causal_scale_derivative_ignores_future_outlier() -> None:
    timestamps = np.arange(6, dtype=np.float64)
    scale = np.array([1.0, 2.0, 3.0, 4.0, 1000.0, 2000.0], dtype=np.float64)

    baseline = _causal_scale_derivative(timestamps, scale, 3, window=4)
    scale[4:] = -1000.0
    with_changed_future = _causal_scale_derivative(timestamps, scale, 3, window=4)

    assert baseline == with_changed_future
    assert baseline == 1.0


def test_causal_scale_derivative_requires_history() -> None:
    timestamps = np.arange(3, dtype=np.float64)
    scale = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert _causal_scale_derivative(timestamps, scale, 1, window=3) is None


def test_causal_geometry_hybrid_falls_back_only_for_invalid_rows() -> None:
    geometry = np.array([1.0, np.nan, -2.0, 4.0], dtype=np.float64)
    neural = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

    hybrid, valid = _hybrid_geometry_with_fallback(geometry, neural)

    np.testing.assert_array_equal(valid, [True, False, False, True])
    np.testing.assert_allclose(hybrid, [1.0, 20.0, 30.0, 4.0])
