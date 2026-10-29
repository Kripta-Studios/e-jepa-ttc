"""Expanded scheduling preserves inactive roles without reading targets."""

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.expanded_replay_plan import expanded_family_queries


def families():
    return [
        {
            "outer_fold": outer,
            "role": f"inner{slot}" if slot < 3 else "outer_dev",
            "experts": {key: "a" * 64 for key in ("A5", "C2F", "PAIR")},
        }
        for outer in range(3)
        for slot in range(4)
    ]


def test_inactive_fold_query_does_not_inherit_a_producer():
    assignments = np.array([[0, 1, 2], [4, 5, 6], [-1, 9, 10]], dtype=np.int64)
    result = expanded_family_queries(assignments, families(), queries=3)
    assert sum(map(len, result.values())) == 8
    assert result[8].size == 0
    assert set(result) == {0, 1, 2, 4, 5, 6, 8, 9, 10}
    np.testing.assert_array_equal(result[9], [1])


@pytest.mark.parametrize("change", ["outer_dev", "other_fold", "dtype", "shape", "family_order"])
def test_invalid_producer_assignment_rejected(change):
    assignments = np.array([[0, 1, 2], [4, 5, 6], [8, 9, 10]], dtype=np.int64)
    descriptions = families()
    if change == "outer_dev":
        assignments[0, 0] = 3
    elif change == "other_fold":
        assignments[0, 0] = 4
    elif change == "dtype":
        assignments = assignments.astype(float)
    elif change == "shape":
        assignments = assignments[:, :2]
    else:
        descriptions[0], descriptions[1] = descriptions[1], descriptions[0]
    with pytest.raises(ValueError):
        expanded_family_queries(assignments, descriptions, queries=3)
