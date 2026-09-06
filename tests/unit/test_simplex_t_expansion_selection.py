"""Input-only production D1 selection must not infer eligibility from targets."""

from copy import deepcopy

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.expansion_selection import selected_expansion_rows
from e_jepa_ttc.simplex_t.pools import expansion_producer


def fixture():
    families = [
        {"outer_fold": 0, "role": f"inner{i}", "family_sha256": str(i) * 64} for i in range(3)
    ]
    family = expansion_producer("extra", ("0" * 64, "1" * 64, "2" * 64))
    index = {
        "tokens": np.array(["a", "b", "c"]),
        "sequences": np.array(["extra"] * 3),
        "producer_family": np.full((3, 3), -1, np.int64),
        "valid": np.zeros((3, 16), bool),
    }
    index["producer_family"][0, [0, 2]] = int(family[-1])
    index["valid"][:, -1] = True
    history = np.full((3, 16), -1, np.int64)
    history[[0, 2], -1] = [0, 1]
    pool = {
        "folds": {
            "0": {
                "additional_train_tokens": ["c", "a"],
                "original_train_tokens": ["old"],
                "D0_train_queries": 1,
                "D1_train_queries": 3,
                "additional_groups": ["extra"],
                "sequence_family_sha256": {"extra": family},
            }
        }
    }
    kwargs = {
        "outer": 0,
        "pool": pool,
        "families": families,
        "allowed_expansion_sequences": {"extra"},
    }
    return index, history, kwargs


def test_preserves_registered_pool_order_and_ignores_privileged_fields():
    index, history, kwargs = fixture()
    expected = selected_expansion_rows(index, history, **kwargs)
    assert expected.tolist() == [2, 0]
    altered = deepcopy(index)
    altered.update(ttc=np.array([np.nan, np.inf, -999]), velocity=np.ones(3), depth=np.zeros(3))
    np.testing.assert_array_equal(expected, selected_expansion_rows(altered, history, **kwargs))


@pytest.mark.parametrize("change", ["duplicate", "inactive", "producer", "groups", "count", "mask"])
def test_rejects_wrong_query_or_family_binding(change):
    index, history, kwargs = fixture()
    fold = kwargs["pool"]["folds"]["0"]
    if change == "duplicate":
        fold["additional_train_tokens"] = ["a", "a"]
    elif change == "inactive":
        fold["additional_train_tokens"] = ["a", "b"]
    elif change == "producer":
        index["producer_family"][0, 0] = (index["producer_family"][0, 0] + 1) % 3
    elif change == "groups":
        kwargs["allowed_expansion_sequences"] = {"protected"}
    elif change == "count":
        fold["D1_train_queries"] = 4
    else:
        history[1, -1] = 2
    with pytest.raises(ValueError):
        selected_expansion_rows(index, history, **kwargs)
