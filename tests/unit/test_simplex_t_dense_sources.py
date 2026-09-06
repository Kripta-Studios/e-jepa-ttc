"""Dense TRAIN replacement preserves OLD_DEV and avoids duplicated original TRAIN."""

import numpy as np
import pytest
import torch

from e_jepa_ttc.evaluation.stage61_nested_pair_router import phase_from_ttc
from e_jepa_ttc.simplex_t.cache import CachedQueries, Normalizer, training_mass
from e_jepa_ttc.simplex_t.dense_sources import dense_source_pair


def fixture():
    original_features = np.full((3, 17), 2, np.float32)
    original_features[2] = 10000
    times = np.array([100, 200, 300], np.int64)
    availability = times + 10
    normalizer = Normalizer(np.zeros(17), np.ones(17), "fixture")

    def source(features, anchors, available, ids, name):
        history = np.full((len(ids), 16), -1, np.int64)
        history[:, -1] = ids
        return CachedQueries(
            features,
            anchors,
            available,
            history,
            phase_from_ttc(np.ones(len(ids))).astype(np.float32),
            np.full(len(ids), 1 / len(ids)),
            normalizer,
            name,
        )

    original = {
        "inner_oof": source(original_features, times, availability, [0, 1], "old-train"),
        "outer_dev": source(original_features, times, availability, [2], "old-dev"),
    }
    dense_features = np.concatenate((original_features[:2], np.full((2, 17), 6, np.float32)))
    dense_times = np.array([100, 200, 400, 500], np.int64)
    dense = source(dense_features, dense_times, dense_times + 10, [0, 1, 2, 3], "dense")
    arguments = dict(
        original_tokens=np.array(["q0", "q1"]),
        original_sequences=np.array(["a", "b"]),
        dev_sequences=np.array(["c"]),
        dense_tokens=np.array(["q0", "q1", "q2", "q3"]),
        dense_sequences=np.array(["a", "b", "a", "b"]),
        dense_target_ttc=np.ones(4),
    )
    return original, dense, arguments


def test_dense_replaces_train_and_preserves_old_dev_without_train_duplicates():
    original, dense, args = fixture()
    before = original["outer_dev"].gather(torch.tensor([0]))
    # Input masses are not trusted as the new pool's supervision weights.
    dense.mass[:] = [0.1, 0.2, 0.3, 0.4]
    result = dense_source_pair(original, dense, **args)
    train, dev = result["inner_oof"], result["outer_dev"]
    assert train.population == 4 and dev.population == 1
    assert len(train.features) == 5  # Dense four plus only OLD_DEV, not original TRAIN again.
    assert train.normalizer is dev.normalizer
    assert np.array_equal(train.normalizer.mean, np.full(17, 4))
    assert np.array_equal(
        train.mass, training_mass(args["dense_target_ttc"], args["dense_sequences"])
    )
    assert dev.history[0, -1] == 4
    after = dev.gather(torch.tensor([0]))
    for position in range(1, 6):
        assert torch.equal(before[position], after[position])
    assert np.array_equal(original["outer_dev"].normalizer.mean, np.zeros(17))


@pytest.mark.parametrize("failure", ["missing", "groups", "dev", "features", "labels", "history"])
def test_invalid_dense_replacement_rejected(failure):
    original, dense, args = fixture()
    if failure == "missing":
        args["dense_tokens"][0] = "xx"
    elif failure == "groups":
        args["dense_sequences"][0] = "d"
    elif failure == "dev":
        args["dev_sequences"][0] = "a"
    elif failure == "features":
        dense.features[0, 0] += 1
    elif failure == "labels":
        args["dense_target_ttc"][0] = 3
    else:
        dense.history[0, -1] = 999
    with pytest.raises(ValueError):
        dense_source_pair(original, dense, **args)
