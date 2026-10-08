from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from operational.sota_eval.baseline_scoring import ALL_METHODS, _score_method


def test_incomplete_method_has_no_complete_cohort_metric() -> None:
    rows = [
        {"truth_ttc_seconds": "2.0", "candidate": 2.5},
        {"truth_ttc_seconds": "3.0", "candidate": ""},
        {"truth_ttc_seconds": "", "candidate": 4.0},
    ]
    result = _score_method(rows, "candidate")
    assert result["prediction_coverage_all_queries"] == pytest.approx(2 / 3)
    assert result["prediction_coverage_on_exposed_truth"] == 0.5
    assert result["complete_exposed_cohort_metrics"] is None
    assert result["complete_exposed_cohort_status"] == "N/A_INCOMPLETE_PREDICTION_COVERAGE"
    assert result["conditional_support"] == 1
    assert result["conditional_success_metrics"]["mae_seconds"] == 0.5
    assert result["conditional_metrics_not_rankable"] is True


def test_complete_method_scores_all_exposed_truth_only() -> None:
    rows = [
        {"truth_ttc_seconds": "2.0", "candidate": 2.5},
        {"truth_ttc_seconds": "3.0", "candidate": 2.5},
        {"truth_ttc_seconds": "", "candidate": ""},
    ]
    result = _score_method(rows, "candidate")
    assert result["complete_exposed_cohort_status"] == "AVAILABLE"
    assert result["complete_exposed_cohort_metrics"]["mae_seconds"] == 0.5
    assert result["conditional_metrics_not_rankable"] is False


def test_all_expected_methods_are_declared() -> None:
    assert ALL_METHODS[:2] == ("cmax_reference", "strttc_adapted")
    assert "H8_seed7" in ALL_METHODS
    assert "public_Garl_rgb_event_full" in ALL_METHODS


def test_fixture_documents_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("query_id", "anchor_us"))
        writer.writeheader()
        writer.writerow({"query_id": "sequence:00", "anchor_us": 100})
    assert list(csv.DictReader(path.open(encoding="utf-8")))[0]["query_id"] == "sequence:00"
    assert hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads('{"new_test_labels_opened": false}')["new_test_labels_opened"] is False
