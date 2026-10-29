"""Replay population selection does not inspect targets or scores."""

import pytest

from e_jepa_ttc.simplex_t.replay_selection import select_replay_rows


def populations():
    return {
        f"outer{outer}/{role}": [f"token{index}" for index in range(100)]
        for outer in range(3)
        for role in ("inner0", "inner1", "inner2", "outer_dev")
    }


def test_selection_is_order_invariant_and_covers_all_families():
    data = populations()
    result = select_replay_rows(data)
    assert result == select_replay_rows({key: values[::-1] for key, values in data.items()})
    assert len(result) == 12
    assert len({token for values in result.values() for token in values}) == 64
    assert sorted(map(len, result.values())) == [5] * 8 + [6] * 4


def test_selection_rejects_missing_lineage():
    data = populations()
    data.pop("outer0/inner0")
    with pytest.raises(ValueError, match="twelve"):
        select_replay_rows(data)


def test_selection_rejects_duplicate_and_insufficient_population():
    data = populations()
    data["outer0/inner0"] = ["same", "same"]
    with pytest.raises(ValueError, match="duplicate"):
        select_replay_rows(data)
    data["outer0/inner0"] = ["one"]
    with pytest.raises(ValueError, match="insufficient"):
        select_replay_rows(data)
