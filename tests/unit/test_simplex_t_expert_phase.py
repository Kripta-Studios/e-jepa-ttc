"""Extended original expert coordinate, with finite-route bit parity."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.evaluation.stage61_nested_pair_router import build_router_features, phase_from_ttc
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase
from e_jepa_ttc.models.three_expert_router import BASE8_FEATURES
from e_jepa_ttc.simplex_t.expert_phase import (
    build_expert_features,
    expert_benchmark_phase,
    expert_phase_from_ttc,
)


def test_finite_phases_are_bit_identical_to_each_historical_arithmetic():
    values = np.array([-60, -0.1, 0.10001, 0.11, 1, 60, 1e30], np.float64)
    assert expert_phase_from_ttc(values).tobytes() == phase_from_ttc(values).tobytes()
    assert expert_benchmark_phase(values).tobytes() == benchmark_phase(values).tobytes()


@pytest.mark.parametrize("function", [expert_phase_from_ttc, expert_benchmark_phase])
def test_infinite_original_points_have_signed_zero_limit(function):
    points = np.array([-np.inf, np.inf])
    phase = function(points)
    assert phase.tolist() == [0.0, 0.0]
    assert np.signbit(phase).tolist() == [True, False]
    assert np.isinf(points).all()


@pytest.mark.parametrize("value", [np.nan, 0.0, 0.05, 0.1])
@pytest.mark.parametrize("function", [expert_phase_from_ttc, expert_benchmark_phase])
def test_invalid_original_points_still_fail(function, value):
    with pytest.raises(ValueError):
        function(np.array([value, np.inf]))


def test_full_feature_assembly_is_exact_and_infinity_not_clipped():
    frame = pd.DataFrame({name: np.arange(4, dtype=float) for name in BASE8_FEATURES})
    frame["token_id"] = ["a", "b", "c", "d"]
    frame["prediction_ttc"] = [-3.0, -60.0, 5.0, 1.0]
    pair = np.array([2.0, -2.0, 4.0, 80.0])
    old = build_router_features(frame, frame, pair)[1]
    new = build_expert_features(frame, frame, pair)[1]
    assert old.to_numpy().tobytes() == new.to_numpy().tobytes()
    pair[-1] = np.inf
    new = build_expert_features(frame, frame, pair)[1]
    assert new.pair_benchmark_phase.iloc[-1] == 0
    assert np.isfinite(new.to_numpy()).all()
    assert np.isinf(pair[-1])
