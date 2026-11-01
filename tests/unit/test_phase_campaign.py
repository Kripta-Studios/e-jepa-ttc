import numpy as np
import pytest


def test_family_folds_exclude_overlap_and_hold_all_siblings():
    from operational.phase_campaign.protocol import family_folds

    records = [
        {"sequence_id": "a1", "scenario_family": "a", "role": "development_exposed"},
        {"sequence_id": "a2", "scenario_family": "a", "role": "development_exposed"},
        {"sequence_id": "b1", "scenario_family": "b", "role": "development_exposed"},
        {"sequence_id": "b2", "scenario_family": "b", "role": "excluded"},
    ]
    folds = family_folds(records)
    assert folds[0] == {"family": "a", "train": ["b1"], "test": ["a1", "a2"]}
    assert folds[1]["train"] == ["a1", "a2"]
    assert all("b2" not in f["train"] + f["test"] for f in folds)
    with pytest.raises(ValueError, match="duplicate"):
        family_folds(records + records[:1])


def test_bbox_estimator_uses_current_endpoint_and_real_clock():
    from operational.phase_campaign.evaluation import bbox_ttc

    boxes = np.array([[0, 0, 10, 10], [0, 0, 11, 11]], float)
    assert bbox_ttc(boxes, np.array([100000, 200000])) == pytest.approx(1.0)
    assert bbox_ttc(boxes, np.array([8100000, 8200000])) == pytest.approx(1.0)
    assert np.isnan(bbox_ttc(np.ones((2, 4)), np.array([1, 2])))


def test_metrics_keep_missing_and_zero_targets_and_signed_intervals():
    from operational.phase_campaign.evaluation import metrics, phase_interval_metrics

    result = metrics(np.array([1.0, 0.0, -2.0]), np.array([2.0, 1.0, np.nan]))
    assert result["population"] == 3
    assert result["finite"] == 2
    assert result["mae_seconds"] == 1
    assert result["relative_rows"] == 1
    assert result["relative_absolute_error_pct"] == 100
    interval = phase_interval_metrics(
        np.array([-0.2, 0.2]), np.array([-0.3, -0.1]), np.array([-0.1, 0.3])
    )
    assert interval["coverage_80"] == 1
    assert interval["crosses_infinite_ttc_fraction"] == 0.5


def test_paired_bootstrap_resamples_groups_not_individual_windows():
    from operational.phase_campaign.evaluation import paired_group_bootstrap

    result = paired_group_bootstrap(
        np.array([1.0, 3.0]), np.array([0.5, 2.0]), np.array(["a", "b"]), repetitions=200, seed=7
    )
    assert result["groups"] == 2
    assert result["mean_difference"] == pytest.approx(-0.75)
    assert result["upper_95"] <= -0.5


def test_fold_indices_never_fit_heldout_or_excluded_recordings():
    from operational.phase_campaign.protocol import fold_indices

    values = {
        "domain": np.array(["eap", "evttc", "evttc", "evttc"]),
        "group_id": np.array(["eap:source", "a", "b", "excluded"]),
    }
    train, test = fold_indices(values, {"train": ["a"], "test": ["b"]})
    assert train.tolist() == [0, 1]
    assert test.tolist() == [2]
    with pytest.raises(ValueError, match="overlap"):
        fold_indices(values, {"train": ["a", "b"], "test": ["b"]})
