"""Matched controls recompute TRAIN weighting and normalization without changing OLD."""

import json
from dataclasses import replace

import numpy as np
import pytest

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.evaluation.stage61_nested_pair_router import phase_from_ttc
from e_jepa_ttc.simplex_t.cache import CachedQueries, Normalizer, training_mass
from e_jepa_ttc.simplex_t.diverse_sources import diverse_source_view
from e_jepa_ttc.simplex_t.matched_sources import matched_source_view


def fixture():
    features = np.repeat(np.array([0, 100, 2, 1000, 10000], np.float32)[:, None], 17, axis=1)
    times = np.arange(5, dtype=np.int64)
    normalizer = Normalizer(np.zeros(17), np.ones(17), "parent")
    targets = np.array([1, 2, 3, 4], np.float64)
    sequences = np.array(["a", "a", "b", "c"])
    history = np.full((4, 16), -1, np.int64)
    history[:, -1] = np.arange(4)
    train = CachedQueries(
        features,
        times,
        times,
        history,
        phase_from_ttc(targets).astype(np.float32),
        training_mass(targets, sequences),
        normalizer,
        "train",
    )
    dev_history = np.full((1, 16), -1, np.int64)
    dev_history[:, -1] = 4
    dev = CachedQueries(
        features, times, times, dev_history, np.ones(1, np.float32), np.ones(1), normalizer, "dev"
    )
    kwargs = {
        "selected_rows": np.array([2, 0], np.int64),
        "train_target_ttc": targets,
        "train_sequences": sequences,
        "pool": "DIVERSE_MATCHED",
        "pool_manifest_sha256": "a" * 64,
    }
    return {"inner_oof": train, "outer_dev": dev}, kwargs


def test_selected_train_normalizer_excludes_unselected_and_dev_observations():
    sources, kwargs = fixture()
    result = matched_source_view(sources, **kwargs)
    train, dev = result["inner_oof"], result["outer_dev"]
    assert train.history[:, -1].tolist() == [2, 0]
    np.testing.assert_array_equal(train.normalizer.mean, np.ones(17))
    np.testing.assert_array_equal(train.mass, [0.5, 0.5])
    assert train.normalizer is dev.normalizer
    assert dev.features is sources["outer_dev"].features
    np.testing.assert_array_equal(dev.history, sources["outer_dev"].history)
    np.testing.assert_array_equal(dev.target_phase, sources["outer_dev"].target_phase)
    assert sources["inner_oof"].population == 4
    assert sources["inner_oof"].normalizer.consumed_ids_sha256 == "parent"


@pytest.mark.parametrize("failure", [None, "count", "query", "group", "producer", "pin"])
def test_diverse_registered_manifest_selection(tmp_path, failure):
    sources, arguments = fixture()
    selection = {
        "tokens": ["q2", "q0"],
        "sequences": ["a", "b"],
        "sequence_family_sha256": {"a": "a" * 64, "b": "b" * 64},
    }
    count = 2
    if failure == "count":
        count = 3
    elif failure == "query":
        selection["tokens"][0] = "foreign"
    elif failure == "group":
        selection["sequences"] = ["a", "c"]
    elif failure == "producer":
        selection["sequence_family_sha256"]["a"] = "f" * 64
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(
            {
                "folds": [
                    {
                        "outer": 0,
                        "nominal_common_count": count,
                        "pools": {
                            "DENSE_OLD": {"tokens": ["d0", "d1"]},
                            "DIVERSE_MATCHED": selection,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    kwargs = dict(
        pool_manifest=path,
        pool_sha256=compute_file_hash(str(path)),
        outer=0,
        train_tokens=np.array(["q0", "q1", "q2", "q3"]),
        train_sequences=arguments["train_sequences"],
        train_target_ttc=arguments["train_target_ttc"],
        sequence_families={"a": "a" * 64, "b": "b" * 64, "c": "c" * 64},
    )
    if failure == "pin":
        kwargs["pool_sha256"] = "0" * 64
    if failure:
        with pytest.raises(ValueError):
            diverse_source_view(sources, **kwargs)
    else:
        result = diverse_source_view(sources, **kwargs)
        assert result["inner_oof"].history[:, -1].tolist() == [2, 0]
        np.testing.assert_array_equal(result["inner_oof"].normalizer.mean, np.ones(17))
        np.testing.assert_array_equal(result["outer_dev"].history, sources["outer_dev"].history)


@pytest.mark.parametrize(
    "change", ["duplicate", "negative", "outside", "target", "dev_leak", "control"]
)
def test_invalid_matched_view_rejected(change):
    sources, kwargs = fixture()
    if change == "duplicate":
        kwargs["selected_rows"] = np.array([0, 0], np.int64)
    elif change == "negative":
        kwargs["selected_rows"] = np.array([-1], np.int64)
    elif change == "outside":
        kwargs["selected_rows"] = np.array([4], np.int64)
    elif change == "target":
        kwargs["train_target_ttc"][0] = 5
    elif change == "dev_leak":
        sources["outer_dev"].history[0, -1] = 0
    else:
        sources["inner_oof"] = replace(sources["inner_oof"], control="REPEAT_CURRENT")
    with pytest.raises(ValueError):
        matched_source_view(sources, **kwargs)
