"""Canonical refitting must preserve exclusions, identity and coverage gates."""

import copy

import numpy as np
import pytest

from scripts.train_phase_canonical_router import comparison, joined_population


def fixture_population():
    source = {
        "token": np.array(["a", "b"]),
        "group": np.array(["road", "Slider-500"]),
        "family": np.array(["road", "Slider"]),
        "target": np.array([2, 3], np.float32),
        "geometry": np.array([2, 3], np.float32),
        "quality": np.ones((2, 20), np.float32),
    }
    records = [
        {
            "token": t,
            "group": g,
            "baseline_exact": True,
            "results": {"original": {"native": [1, 2, 3]}, "canonical": {"native": [2, 3, 4]}},
        }
        for t, g in zip(source["token"], source["group"], strict=True)
    ]
    config = {
        "fit_groups": ["road", "Slider-500"],
        "excluded_road": ["benchmark"],
        "excluded_slider": ["Slider-750", "Slider-1000"],
    }
    return source, records, config


def test_join_uses_identity_not_order():
    source, records, config = fixture_population()
    records[1]["results"]["canonical"]["native"] = [6, 7, 8]
    joined = joined_population(source, records[::-1], config)
    np.testing.assert_array_equal(joined["canonical"], [[2, 3, 4], [6, 7, 8]])
    np.testing.assert_array_equal(joined["target"], source["target"])
    np.testing.assert_array_equal(joined["quality"], source["quality"])


@pytest.mark.parametrize("fault", ["duplicate", "group", "baseline", "excluded", "target"])
def test_invalid_training_admission_is_rejected(fault):
    source, records, config = copy.deepcopy(fixture_population())
    if fault == "duplicate":
        records[1]["token"] = records[0]["token"]
    elif fault == "group":
        records[1]["group"] = "different"
    elif fault == "baseline":
        records[0]["baseline_exact"] = False
    elif fault == "excluded":
        config["excluded_road"].append("road")
    else:
        source["target"][0] = np.nan
    with pytest.raises(ValueError):
        joined_population(source, records, config)


def test_improvement_on_one_domain_cannot_hide_missing_other_domain():
    source, _, _ = fixture_population()
    result = comparison(source, np.array([4, 4.0]), np.array([2, np.nan]), np.ones(2, bool))
    assert not result["passes"]
    assert result["domains"]["slider"]["newly_missing"] == 1


def test_equal_outputs_do_not_count_as_improvement():
    source, _, _ = fixture_population()
    result = comparison(source, np.array([4, 4.0]), np.array([4, 4.0]), np.ones(2, bool))
    assert not result["passes"]
