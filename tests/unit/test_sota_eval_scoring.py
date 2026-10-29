"""Hand-calculated metric, cohort and cluster-bootstrap contract checks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from operational.sota_eval.scoring import read_rows, run, score


def make_csv(tmp_path: Path, rows: list[list[object]]) -> Path:
    path = tmp_path / "input.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "sequence_id", "truth_ttc_seconds", "A", "B", "group"])
        writer.writerows(rows)
    return path


def test_signed_rte_micro_macro_hand_computed(tmp_path):
    source = make_csv(
        tmp_path,
        [
            ["a", "s1", 2, 4, 2, "g1"],
            ["b", "s1", -4, -2, -4, "g1"],
            ["c", "s2", 10, 10, 10, "g2"],
        ],
    )
    _, rows = read_rows(source, ["A", "B"])
    report, sequences, pairs = score(rows, ["A", "B"], bootstrap_draws=100, seed=17)
    metrics = report["methods"]["A"]["complete_cohort_aggregate"]
    assert metrics["micro_rte_percent"] == pytest.approx(50)
    assert metrics["macro_sequence_rte_percent"] == pytest.approx(37.5)
    assert metrics["micro_mae_seconds"] == pytest.approx(4 / 3)
    assert metrics["micro_rmse_seconds"] == pytest.approx(math.sqrt(8 / 3))
    assert metrics["micro_median_absolute_error_seconds"] == 2
    assert metrics["macro_sequence_median_absolute_error_seconds"] == 1
    assert len(sequences) == 4
    assert pairs[0]["metrics"]["micro_rte_percent"]["point_delta"] == 50
    assert pairs[0]["metrics"]["micro_rte_percent"]["percentile_ci95"] == [0, 75]


@pytest.mark.parametrize("failed_prediction", ["nan", "inf", "-inf"])
def test_gt_only_cohort_preserves_missing_zero_and_failed_predictions(tmp_path, failed_prediction):
    source = make_csv(
        tmp_path,
        [
            ["a", "s1", 2, failed_prediction, 2, "g1"],
            ["b", "s2", -4, -2, -4, "g2"],
            ["zero", "s1", 0, 999, 0, "g1"],
            ["missing", "s2", "", "inf", 2, "g2"],
        ],
    )
    report = run(source, tmp_path / "output", ["A", "B"], bootstrap_draws=30)
    assert report["cohort"]["gt_eligible"] == 2
    assert report["cohort"]["gt_nonfinite"] == 1
    assert report["cohort"]["gt_zero"] == 1
    method = report["methods"]["A"]
    assert method["complete_cohort_metrics"] is None
    assert method["complete_cohort_aggregate"] is None
    assert method["prediction_nonfinite_on_eligible_gt"] == 1
    assert method["conditional_on_scorable_predictions_metrics"]["rte_percent"] == 50
    assert report["methods"]["B"]["complete_cohort_metrics"]["rte_percent"] == 0
    assert report["paired"][0]["status"] == "UNAVAILABLE_INCOMPLETE_COHORT"
    with (tmp_path / "output" / "SCORED_ROWS.csv").open(newline="", encoding="utf-8") as handle:
        preserved = list(csv.DictReader(handle))
    assert len(preserved) == 4
    assert preserved[3]["query_id"] == "missing"
    assert preserved[3]["truth_ttc_seconds"] == ""


@pytest.mark.parametrize(
    "rows,match",
    [
        ([["x", "s1", 1, 1, 1, "g1"], ["x", "s2", 1, 1, 1, "g2"]], "Duplicate query_id"),
        ([["x", "s1", 1, 1, 1, "g1"], ["y", "s1", 1, 1, 1, "g2"]], "multiple bootstrap"),
    ],
)
def test_rejects_duplicate_queries_and_split_groups(tmp_path, rows, match):
    source = make_csv(tmp_path, rows)
    with pytest.raises(ValueError, match=match):
        read_rows(source, ["A", "B"], "group")


def test_group_bootstrap_keeps_sequences_together_and_is_paired(tmp_path):
    source = make_csv(
        tmp_path,
        [
            ["a", "s1", 1, 2, 2, "g1"],
            ["b", "s2", 1, 3, 3, "g1"],
            ["c", "s3", 1, 4, 4, "g2"],
        ],
    )
    _, rows = read_rows(source, ["A", "B"], "group")
    first, _, _ = score(rows, ["A", "B"], bootstrap_draws=80, seed=31)
    second, _, _ = score(rows, ["A", "B"], bootstrap_draws=80, seed=31)
    assert first == second
    assert first["bootstrap"]["groups"] == ["g1", "g2"]
    for metric in first["paired"][0]["metrics"].values():
        assert metric["point_delta"] == 0
        assert metric["percentile_ci95"] == [0, 0]


def test_outputs_regenerate_identically_source_unchanged_and_hashes_match(tmp_path):
    source = make_csv(tmp_path, [["a", "s1", 2, 1, 2, "g1"], ["b", "s2", 3, 2, 3, "g2"]])
    before = source.read_bytes()
    destination = tmp_path / "output"
    run(source, destination, ["A", "B"], bootstrap_draws=60)
    first = {path.name: path.read_bytes() for path in destination.iterdir()}
    run(source, destination, ["A", "B"], bootstrap_draws=60)
    assert first == {path.name: path.read_bytes() for path in destination.iterdir()}
    assert source.read_bytes() == before
    hashes = json.loads((destination / "SHA256.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == expected


def test_overflow_counts_as_failure_without_json_nan(tmp_path):
    source = make_csv(tmp_path, [["a", "s1", 1, "1e308", 2, "g1"]])
    report = run(source, tmp_path / "out", ["A", "B"], bootstrap_draws=10)
    assert report["methods"]["A"]["arithmetic_nonfinite_on_eligible_gt"] == 1
    assert report["methods"]["A"]["complete_cohort_metrics"] is None
    assert "NaN" not in (tmp_path / "out" / "REPORT.json").read_text()


def test_single_cluster_does_not_publish_spurious_interval(tmp_path):
    source = make_csv(tmp_path, [["a", "s1", 1, 2, 1, "g1"]])
    _, rows = read_rows(source, ["A", "B"])
    report, _, _ = score(rows, ["A", "B"], bootstrap_draws=10, seed=1)
    comparison = report["paired"][0]
    assert comparison["status"] == "POINT_ONLY_FEWER_THAN_TWO_GROUPS"
    assert comparison["metrics"]["micro_rte_percent"]["point_delta"] == 100
    assert comparison["metrics"]["micro_rte_percent"]["percentile_ci95"] is None


def test_no_eligible_truth_has_no_metrics_or_bootstrap(tmp_path):
    source = make_csv(tmp_path, [["a", "s1", "nan", 2, 1, "g1"], ["b", "s2", 0, 2, 1, "g2"]])
    report = run(source, tmp_path / "out", ["A", "B"], bootstrap_draws=10)
    assert report["methods"]["A"]["status"] == "NO_ELIGIBLE_GT"
    assert report["methods"]["A"]["coverage_on_eligible_gt"] is None
    assert report["bootstrap"]["groups"] == []
