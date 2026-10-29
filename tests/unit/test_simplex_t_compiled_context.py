"""Strict schema and availability checks before cached head inputs are accepted."""

import numpy as np
import pytest

from e_jepa_ttc.evaluation.stage61_nested_pair_router import phase_from_ttc
from e_jepa_ttc.simplex_t.compiled_context import store_observations, validate_block


def block():
    values = {
        "features145": np.zeros((2, 145), np.float32),
        "expert_ttc": np.ones((2, 3), np.float32),
        "known": np.ones((2, 2), bool),
        "pair_features": np.zeros((2, 133), np.float32),
        "observation_ids": np.array([1, 2], np.int64),
        "anchor_us": np.array([100, 200], np.int64),
        "available_us": np.array([210, 210], np.int64),
    }
    values["features145"][:, 8:11] = phase_from_ttc(values["expert_ttc"])
    return values


def test_complete_input_block_accepted():
    validate_block(block(), np.array([1, 2]), np.array([100, 200]), 210)


@pytest.mark.parametrize(
    "corruption", [None, "features145", "expert_ttc", "known", "anchor_us", "available_us", "sign"]
)
def test_deduplicated_rows_require_byte_exact_compiled_fields(corruption):
    arrays = block()
    fields = ("features145", "expert_ttc", "known", "anchor_us", "available_us")
    destinations = {
        name: np.zeros((4, *arrays[name].shape[1:]), arrays[name].dtype) for name in fields
    }
    consumed = np.zeros(4, bool)
    store_observations(destinations, consumed, arrays)
    before = {name: array.copy() for name, array in destinations.items()}
    arrays["observation_ids"][1] = 3  # One duplicate and one fresh row.
    if corruption == "sign":
        arrays["features145"][0, 0] = -0.0
    elif corruption == "known":
        arrays["known"][0, 0] = False
    elif corruption is not None:
        arrays[corruption][0] += 1
    if corruption is None:
        store_observations(destinations, consumed, arrays)
        assert consumed.tolist() == [False, True, True, True]
        for name in fields:
            assert np.array_equal(destinations[name][3], arrays[name][1])
    else:
        with pytest.raises(ValueError, match="content differs"):
            store_observations(destinations, consumed, arrays)
        assert consumed.tolist() == [False, True, True, False]
        for name in fields:
            assert destinations[name].tobytes() == before[name].tobytes()


def test_original_infinite_pair_requires_consistent_zero_phase():
    arrays = block()
    arrays["expert_ttc"][0, 2] = np.inf
    arrays["features145"][0, 10] = 0
    validate_block(arrays, np.array([1, 2]), np.array([100, 200]), 210)
    arrays["features145"][0, 10] = 0.01
    with pytest.raises(ValueError, match="point/phase inconsistency"):
        validate_block(arrays, np.array([1, 2]), np.array([100, 200]), 210)


def test_nonfinite_a5_remains_invalid():
    arrays = block()
    arrays["expert_ttc"][0, 0] = np.inf
    arrays["features145"][0, 8] = 0
    with pytest.raises(ValueError, match="nonfinite A5/C2F"):
        validate_block(arrays, np.array([1, 2]), np.array([100, 200]), 210)


@pytest.mark.parametrize("corruption", ["time", "identity", "nan", "dtype", "shape", "target"])
def test_corrupt_or_privileged_block_rejected(corruption):
    arrays = block()
    if corruption == "time":
        arrays["available_us"][0] = 100
    elif corruption == "identity":
        arrays["observation_ids"] = arrays["observation_ids"][::-1]
    elif corruption == "nan":
        arrays["features145"][0, 0] = np.nan
    elif corruption == "dtype":
        arrays["anchor_us"] = arrays["anchor_us"].astype(float)
    elif corruption == "shape":
        arrays["features145"] = arrays["features145"][:, :17]
    else:
        arrays["target_ttc"] = np.ones(2)
    with pytest.raises(ValueError):
        validate_block(arrays, np.array([1, 2]), np.array([100, 200]), 210)
