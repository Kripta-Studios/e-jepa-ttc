"""D1 merge preserves OLD inference inputs and excludes its observations from normalization."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.cache import CachedQueries, Normalizer, training_mass
from e_jepa_ttc.simplex_t.expanded_sources import merge_validated_sources


def fixture():
    normalizer = Normalizer(np.zeros(17), np.ones(17), "fixture")
    old = np.full((3, 17), 2, np.float32)
    old[2] = 10000
    new = np.full((2, 17), 6, np.float32)
    anchors = np.array([100, 200, 300], np.int64)
    available = anchors + 10

    def source(features, times, availability, ids, name):
        history = np.full((len(ids), 16), -1, np.int64)
        history[:, -1] = ids
        return CachedQueries(
            features,
            times,
            availability,
            history,
            np.zeros(len(ids), np.float32),
            np.full(len(ids), 1 / len(ids)),
            normalizer,
            name,
        )

    d0 = {
        "inner_oof": source(old, anchors, available, [0, 1], "old-train"),
        "outer_dev": source(old, anchors, available, [2], "old-dev"),
    }
    expansion = source(new, anchors[:2], available[:2], [0, 1], "new-train")
    return d0, expansion


def merge(d0, expansion, sequences=None):
    return merge_validated_sources(
        d0,
        expansion,
        original_train_sequences=np.array(["a", "b"]),
        expansion_train_sequences=np.array(["c", "d"]) if sequences is None else sequences,
    )


def test_shared_train_normalizer_and_unchanged_old_dev():
    d0, expansion = fixture()
    before = d0["outer_dev"].gather(torch.tensor([0]))
    combined = merge(d0, expansion)
    train, dev = combined["inner_oof"], combined["outer_dev"]
    assert train.normalizer is dev.normalizer
    np.testing.assert_array_equal(train.normalizer.mean, np.full(17, 4.0))
    np.testing.assert_array_equal(train.mass, np.full(4, 0.25))
    assert train.history[:, -1].tolist() == [0, 1, 3, 4]
    after = dev.gather(torch.tensor([0]))
    for index in range(1, 6):
        assert torch.equal(before[index], after[index])
    np.testing.assert_array_equal(d0["outer_dev"].normalizer.mean, np.zeros(17))


def test_merged_mass_matches_registered_recipe_with_unequal_group_counts():
    d0, expansion = fixture()
    old_sequences, new_sequences = np.array(["a", "a"]), np.array(["c", "d"])
    combined = merge_validated_sources(
        d0,
        expansion,
        original_train_sequences=old_sequences,
        expansion_train_sequences=new_sequences,
    )
    expected = training_mass(
        np.array([-2.0, -1.0, 1.0, 3.0]), np.concatenate((old_sequences, new_sequences))
    )
    np.testing.assert_allclose(combined["inner_oof"].mass, expected, rtol=1e-15, atol=0)


@pytest.mark.parametrize("failure", ["overlap", "control", "mass", "leak", "index"])
def test_merge_rejects_invalid_source_composition(failure):
    d0, expansion = fixture()
    sequences = np.array(["c", "d"])
    if failure == "overlap":
        sequences[0] = "a"
    elif failure == "control":
        expansion = replace(expansion, control="PAST_REVERSED")
    elif failure == "mass":
        expansion.mass[:] = [0.1, 0.9]
    elif failure == "leak":
        d0["inner_oof"].history[0, -1] = 2
    else:
        expansion.history[0, -1] = 5
    with pytest.raises(ValueError):
        merge(d0, expansion, sequences)
