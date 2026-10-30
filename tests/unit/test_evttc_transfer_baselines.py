import json
import sys

import pytest

from operational.evttc_transfer_audit.baselines import saved_predictions
from scripts.report_evttc_transfer import metric_text


@pytest.mark.parametrize("target_value", [None, 0.0])
def test_report_retains_population_without_relative_metrics(tmp_path, monkeypatch, target_value):
    from scripts.report_evttc_transfer import main

    for name in ("phase", "garl"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "all_rows.jsonl").write_text("", encoding="utf-8")
    target = {
        "query_id": "a:0",
        "query_us": 10,
        "sequence_id": "a",
        "target_status": "missing" if target_value is None else "valid",
        "reason": "missing_target",
        "target_ttc": target_value,
        "target_contract_sha256": "fixture",
    }
    (tmp_path / "targets.json").write_text(json.dumps({"rows": [target]}), encoding="utf-8")
    (tmp_path / "policy.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report",
            "--phase-run",
            str(tmp_path / "phase"),
            "--garl-run",
            str(tmp_path / "garl"),
            "--targets",
            str(tmp_path / "targets.json"),
            "--policy",
            str(tmp_path / "policy.json"),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    main()
    result = json.loads((tmp_path / "out/report.json").read_text(encoding="utf-8"))
    assert all(row["counts"]["requested"] == 1 for row in result["scores"].values())
    assert "N/A" in (tmp_path / "out/BASELINES.md").read_text(encoding="utf-8")


def test_unavailable_metric_renders_as_na() -> None:
    assert metric_text(None) == "N/A"
    assert metric_text(0.0) == "0.000000"


def test_saved_baseline_must_match_query_clock_not_just_row_number() -> None:
    targets = [{"query_id": "a:0", "query_us": 10, "sequence_id": "a"}]
    ledger = [
        {
            "identity": {"sequence_id": "a", "row_index": 0, "anchor_us": 11},
            "prediction_ttc_seconds": 1.0,
            "variant": "full",
        }
    ]
    with pytest.raises(ValueError, match="clock"):
        saved_predictions(ledger, targets, "Garl_full")


def test_saved_baseline_keeps_absent_outputs() -> None:
    targets = [{"query_id": "a:0", "query_us": 10, "sequence_id": "a"}]
    ledger = [
        {
            "identity": {"sequence_id": "a", "row_index": 0, "anchor_us": 10},
            "prediction_ttc_seconds": None,
            "variant": "full",
        }
    ]
    assert saved_predictions(ledger, targets, "Garl_full") == [
        {"query_id": "a:0", "prediction": None}
    ]
