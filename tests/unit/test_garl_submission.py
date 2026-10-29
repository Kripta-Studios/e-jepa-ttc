from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_garlttc_submission import validate_submission


def test_garl_submission_validator_accepts_known_token_and_rejects_nonfinite(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.json"
    submission = tmp_path / "submission.json"
    output = tmp_path / "validation.json"
    sample.write_text(
        json.dumps(
            {"meta": {"format": "garlttc_prediction_v1"}, "results": {"token": {"ttc": 1.0}}}
        ),
        encoding="utf-8",
    )
    submission.write_text(
        json.dumps(
            {"meta": {"format": "garlttc_prediction_v1"}, "results": {"token": {"ttc": 1.5}}}
        ),
        encoding="utf-8",
    )
    result = validate_submission(submission, sample, output)
    assert result["status"] == "PASS"
    assert result["prediction_count"] == 1

    submission.write_text(
        '{"meta":{"format":"garlttc_prediction_v1"},"results":{"token":{"ttc":NaN}}}',
        encoding="utf-8",
    )
    result = validate_submission(submission, sample, output)
    assert result["status"] == "FAIL"
    assert "finite numeric" in result["errors"][0]
