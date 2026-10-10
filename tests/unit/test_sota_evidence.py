"""Boundary, failure, aggregation and submission integrity regressions."""

import math

import numpy as np
import pandas as pd
import pytest

from operational.sota_evidence.metrics import bands, observations, paired_macro, summarize
from operational.sota_evidence.run import training_split_audit
from operational.sota_evidence.submission import build_submission


def test_two_papers_do_not_define_the_same_mid():
    result = observations(np.array([2.0]), np.array([4.0]))
    assert result["garl_mid"][0] == pytest.approx(abs(math.log(0.95 / 0.975)) * 1e4)
    assert result["react_logratio"][0] == pytest.approx(math.log(2))


def test_exact_predictions_have_zero_error():
    truth = np.array([-8.0, 1.0, 4.0, 8.0])
    result = summarize(truth, truth, np.array(["a", "a", "b", "b"]))
    assert result["garl_weighted_mid_strict"] == 0
    assert result["react_weighted_logratio_raw"] == 0
    assert result["macro_rte_percent"] == 0


def test_paper_band_boundaries_are_explicit():
    truth = np.array([-10, -1, 0, 3, 6, 10, 11.0])
    assert bands(truth, "garl").tolist() == ["n", "n", "outside", "c", "s", "l", "outside"]
    assert bands(truth, "garl_code").tolist()[0] == "outside"
    assert bands(truth, "react").tolist() == [
        "outside",
        "outside",
        "outside",
        "s",
        "l",
        "outside",
        "outside",
    ]


@pytest.mark.parametrize("prediction", [np.nan, np.inf, -np.inf, 0.0])
def test_invalid_predictions_are_retained(prediction):
    result = summarize(np.array([1.0, 2.0]), np.array([prediction, 2.0]), np.array(["a", "b"]))
    assert result["n"] == 2
    if not np.isfinite(prediction):
        assert result["micro_rte_percent"] is None
        assert result["macro_rte_percent"] is None
    assert result["garl_mid_strict"] is None
    assert result["react_logratio_strict"] is None


def test_reference_infinity_quirk_cannot_pass_strict_metric():
    result = observations(np.array([2.0]), np.array([np.inf]))
    assert np.isfinite(result["garl_reference_mid"][0])
    assert np.isnan(result["garl_mid"][0])
    assert result["garl_reference_failed"][0]


def test_near_zero_mid_domain_and_negative_predictions():
    result = observations(np.array([0.05, 2.0, -2.0]), np.array([0.05, 0.1, -2.0]))
    assert np.isnan(result["garl_mid"][:2]).all()
    assert result["garl_mid"][2] == 0
    assert result["react_logratio"][0] == 0
    assert np.isnan(result["react_logratio"][2])


def test_macro_weights_sequences_equally_not_windows():
    result = summarize(np.ones(4), np.array([1, 1, 1, 2.0]), np.array(["a", "a", "a", "b"]))
    assert result["micro_rte_percent"] == 25
    assert result["macro_rte_percent"] == 50


def test_missing_band_never_reweights_remaining_bands():
    result = summarize(np.array([1.0]), np.array([2.0]), np.array(["a"]))
    assert result["react_weighted_logratio_raw"] is None
    assert result["garl_weighted_mid_strict"] is None


def test_react_weight_sum_is_not_silently_normalized():
    truth = np.array([1.0, 4.0, 8.0])
    result = summarize(truth, truth * 2, np.array(["a", "b", "c"]))
    assert result["react_weighted_logratio_raw"] == pytest.approx(0.9 * math.log(2))
    assert result["react_weighted_logratio_normalized"] == pytest.approx(math.log(2))


def test_paired_bootstrap_preserves_clusters_and_direction():
    result = paired_macro(
        np.array([2.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 4.0]), np.array(["a", "b"])
    )
    assert result["delta_macro_rte_pp"] == -100
    assert result["ci_low_pp"] == -100
    assert result["ci_high_pp"] == -100


def test_paired_failure_and_single_sequence_do_not_get_ci():
    assert (
        paired_macro(np.ones(2), np.ones(2), np.ones(2), np.array(["a", "a"]))["status"]
        == "TOO_FEW_SEQUENCES"
    )
    assert (
        paired_macro(np.ones(2), np.array([1.0, np.nan]), np.ones(2), np.array(["a", "b"]))[
            "status"
        ]
        == "INCOMPLETE"
    )


@pytest.mark.parametrize("truth", [np.array([0.0]), np.array([np.nan]), np.ones((1, 1))])
def test_invalid_ground_truth_is_rejected(truth):
    with pytest.raises(ValueError):
        observations(truth, np.ones_like(truth))


def test_submission_is_exact_and_has_no_placeholders():
    frame = pd.DataFrame({"sample_token": ["b", "a"], "prediction": [-2.0, 3.0]})
    result = build_submission(["a", "b"], frame, checkpoint_sha256="a" * 64)
    assert result["results"] == {"a": {"ttc": 3.0}, "b": {"ttc": -2.0}}


@pytest.mark.parametrize(
    "tokens,values",
    [
        (["a"], [1]),
        (["a", "a"], [1, 2]),
        (["a", "c"], [1, 2]),
        (["a", "b"], [1, np.nan]),
        (["a", "b"], [1, np.inf]),
    ],
)
def test_bad_submission_is_rejected(tokens, values):
    with pytest.raises(ValueError):
        build_submission(
            ["a", "b"],
            pd.DataFrame({"sample_token": tokens, "prediction": values}),
            checkpoint_sha256="a" * 64,
        )


def test_actual_training_sequences_are_checked_against_official_test(tmp_path):
    index = tmp_path / "train.parquet"
    pd.DataFrame({"sequence_id": ["a", "a", "b"]}).to_parquet(index)
    (tmp_path / "official_train.txt").write_text("a\nb\n")
    (tmp_path / "official_test.txt").write_text("c\n")
    assert training_split_audit(index, tmp_path)["status"] == "PASSED"
    (tmp_path / "official_test.txt").write_text("b\nc\n")
    result = training_split_audit(index, tmp_path)
    assert result["status"] == "FAILED"
    assert result["train_test_overlap"] == ["b"]
