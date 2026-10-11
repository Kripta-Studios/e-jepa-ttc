"""Missing external GT is not a zero TTC and must not modify predictions."""

import numpy as np

from operational.dev32_missing_truth_score import labelled_arrays
from operational.external_missing_truth_queue import route
from operational.fcwd_rgb_score_fix import normalize_missing_truth


def test_empty_truth_is_missing_and_other_fields_unchanged() -> None:
    rows = [
        {"truth_ttc_seconds": "", "prediction": "12.0"},
        {"truth_ttc_seconds": "0", "prediction": "nan"},
        {"prediction": "3"},
        {"truth_ttc_seconds": "bad"},
    ]
    result = list(normalize_missing_truth(iter(rows)))
    assert result == [{"truth_ttc_seconds": "nan", "prediction": "12.0"}, *rows[1:]]
    assert rows[0]["truth_ttc_seconds"] == ""


def test_gt_filter_retains_failed_predictions_and_corresponding_groups() -> None:
    truth, prediction, groups = labelled_arrays(
        np.array([1.0, np.nan, 3.0]), np.array([np.nan, 99.0, 4.0]), np.array(["a", "b", "c"])
    )
    np.testing.assert_array_equal(truth, [1.0, 3.0])
    np.testing.assert_array_equal(prediction, [np.nan, 4.0])
    np.testing.assert_array_equal(groups, ["a", "c"])


def test_queue_routes_only_the_score_command() -> None:
    command = [
        "python",
        "-m",
        "operational.rgb_port.transfer",
        "score-dev32",
        "--comparator-csv",
        "c.csv",
    ]
    assert route(command) == [
        "python",
        "-m",
        "operational.dev32_missing_truth_score",
        "--comparator",
        "c.csv",
    ]
    for target in ["predict-dev32", "block-external"]:
        other = ["python", "-m", "operational.rgb_port.transfer", target]
        assert route(other) == other
