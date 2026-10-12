"""Cross-fitting must separate whole groups and every normalization observation."""

import numpy as np
import pytest

from operational.multimodal_research.crossfit_data import group_folds, observation_split


def test_group_assignment_is_stable_under_row_reordering() -> None:
    groups = np.repeat([f"group{i}" for i in range(8)], 3)
    expected = group_folds(groups)
    order = np.random.default_rng(19).permutation(len(groups))
    np.testing.assert_array_equal(group_folds(groups[order]), expected[order])
    assert all(len(set(expected[groups == group])) == 1 for group in groups)
    assert all(len(set(groups[expected == fold])) == 2 for fold in range(4))


def test_shared_observation_across_groups_is_rejected() -> None:
    ids = np.array([["shared", "train"], ["shared", "heldout"]])
    with pytest.raises(ValueError, match="crosses"):
        observation_split(ids, np.ones((2, 2), bool), np.array([True, False]))


def test_padding_cannot_enter_normalizer_and_duplicates_count_once() -> None:
    ids = np.array([["padding", "train"], ["train", "train"], ["padding", "heldout"]])
    valid = np.array([[False, True], [True, True], [False, True]])
    training, heldout = observation_split(ids, valid, np.array([True, True, False]))
    assert training == ["train"] and heldout == ["heldout"]
