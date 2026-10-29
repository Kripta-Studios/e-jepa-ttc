"""Failures must close audibly without granting a scientific rescue branch."""

import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.campaign_failure import close_campaign_failure, failure_category


@pytest.mark.parametrize(
    "error,category",
    [
        (ValueError("hash differs"), "INTEGRITY_BLOCKED"),
        (FileNotFoundError("source missing"), "INPUT_MISSING"),
        (FloatingPointError("nan"), "NUMERICAL_FAILURE"),
        (TimeoutError("budget exhausted"), "RESOURCE_BLOCKED"),
        (RuntimeError("unexpected failure"), "TECHNICAL_FAILURE"),
    ],
)
def test_failure_categories_are_distinct(error: Exception, category: str) -> None:
    assert failure_category(error) == category


def test_failure_closure_preserves_previous_result_and_partial_inventory(tmp_path: Path) -> None:
    (tmp_path / "RUN_LEDGER.jsonl").write_text(
        json.dumps({"state": "stage64_seed7_running"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "checkpoint_last.pt").write_bytes(b"partial checkpoint")
    (tmp_path / "CAMPAIGN_RESULT.json").write_text('{"prior": true}', encoding="utf-8")
    result = close_campaign_failure(tmp_path, ValueError("checkpoint hash differs"))
    assert result["status"] == "INTEGRITY_BLOCKED"
    assert result["failure_phase"] == "stage64_seed7_running"
    assert not result["scientific_negative"]
    assert not result["fallback_authorized_by_failure"]
    assert not result["observed_artifacts"][0]["accepted_as_valid"]
    assert len(list(tmp_path.glob("PRIOR_CAMPAIGN_RESULT_*.json"))) == 1
    assert json.loads((tmp_path / "CAMPAIGN_RESULT.json").read_text()) == result
