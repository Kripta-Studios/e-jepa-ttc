"""Strict schema and availability checks before cached head inputs are accepted."""

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.compiled_context import validate_block


def block():
    return {
        "features145": np.zeros((2, 145), np.float32),
        "expert_ttc": np.ones((2, 3), np.float32),
        "known": np.ones((2, 2), bool),
        "pair_features": np.zeros((2, 133), np.float32),
        "observation_ids": np.array([1, 2], np.int64),
        "anchor_us": np.array([100, 200], np.int64),
        "available_us": np.array([210, 210], np.int64),
    }


def test_complete_input_block_accepted():
    validate_block(block(), np.array([1, 2]), np.array([100, 200]), 210)


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
