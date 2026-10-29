"""Exact paired factorial algebra without training or scientific scores."""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.simplex_t.factorial_analysis import paired_factor_effects


def fixture(d1):
    expected = pd.DataFrame(
        dict(
            sample_token=[f"q{i:05d}" for i in range(8192)],
            sequence_id=[f"s{i % 9}" for i in range(8192)],
            track_id=[f"t{i % 25}" for i in range(8192)],
            outer_fold=np.arange(8192) % 3,
        )
    )
    frames = []
    for d, h, c in product((0, 1) if d1 else (0,), (0, 1), (0, 1)):
        frames.append(
            expected.assign(
                arm=f"TPR-D{d}-H{(1, 8)[h]}-C{(64, 160)[c]}",
                seed=7,
                loss=20 + 2 * d + 3 * h + 5 * c + 7 * h * c,
            )
        )
    return expected, pd.concat(frames).sample(frac=1, random_state=7).reset_index(drop=True)


@pytest.mark.parametrize("d1", [False, True])
def test_factorial_pairs_by_identity_not_input_order(d1):
    expected, predictions = fixture(d1)
    result = paired_factor_effects(predictions, expected, d1_available=d1)
    np.testing.assert_array_equal(result.H, np.full(8192, 6.5))
    np.testing.assert_array_equal(result.C, np.full(8192, 8.5))
    np.testing.assert_array_equal(result.HxC, np.full(8192, 7))
    assert ("D" in result) == d1
    if d1:
        np.testing.assert_array_equal(result.D, np.full(8192, 2))
        assert (result[["DxH", "DxC", "DxHxC"]] == 0).all().all()


@pytest.mark.parametrize("corruption", ["drop", "duplicate", "fold", "track", "loss"])
def test_invalid_pairing_never_drops_queries_silently(corruption):
    expected, predictions = fixture(False)
    if corruption == "drop":
        predictions = predictions.iloc[1:]
    elif corruption == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[:1]])
    elif corruption == "fold":
        predictions.loc[0, "outer_fold"] = 99
    elif corruption == "track":
        predictions.loc[0, "track_id"] = "another-track"
    else:
        predictions.loc[0, "loss"] = np.nan
    with pytest.raises(ValueError):
        paired_factor_effects(predictions, expected, d1_available=False)
