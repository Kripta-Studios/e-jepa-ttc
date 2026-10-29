"""The parallel TRAIN preflight rejects leakage and illegal temporal context."""

import numpy as np
import pytest

from operational.efficient_context.garl_parallel_preflight import producer_for, validate_history


def fit(**updates):
    value = {
        "key": "outer0_inner1",
        "outer": 0,
        "inner": 1,
        "excluded_inner": ["held"],
        "train_sequences": ["train"],
    }
    return {**value, **updates}


def test_selects_the_excluded_inner_producer():
    assert producer_for("held", 0, [fit(), fit(key="outer0", inner=None)]) == "outer0_inner1"


@pytest.mark.parametrize("fits", [[], [fit(), fit()], [fit(outer=1)], [fit(inner=None)]])
def test_refuses_missing_ambiguous_or_outer_routes(fits):
    with pytest.raises(ValueError, match="exactly one"):
        producer_for("held", 0, fits)


def test_refuses_a_holdout_in_training_even_if_exclusion_is_declared():
    with pytest.raises(ValueError, match="occurs in producer training"):
        producer_for("held", 0, [fit(train_sequences=["held"])])


def test_historical_masks_and_lags_are_not_changed():
    history = np.array([[-1, -1, -1, -1, -1, -1, 0, 1]])
    before = history.copy()
    valid, lags = validate_history(history, np.array([100, 150]))
    np.testing.assert_array_equal(history, before)
    np.testing.assert_array_equal(valid, [[False] * 6 + [True, True]])
    np.testing.assert_array_equal(lags[:, -2:], [[50, 0]])


@pytest.mark.parametrize("invalid", [-2, 2, 1.5])
def test_refuses_invalid_observation_ids(invalid):
    history = np.array([[-1, -1, -1, -1, -1, -1, invalid, 1]])
    with pytest.raises(ValueError):
        validate_history(history, np.array([100, 150]))


def test_refuses_missing_current_slot():
    with pytest.raises(ValueError, match="missing current"):
        validate_history(np.full((1, 8), -1), np.array([100, 150]))


def test_refuses_future_context():
    with pytest.raises(ValueError, match="future observation"):
        validate_history(np.array([[-1, -1, -1, -1, -1, -1, 1, 0]]), np.array([100, 150]))
