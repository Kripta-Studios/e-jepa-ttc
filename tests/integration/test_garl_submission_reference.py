from __future__ import annotations

import json

from tests.integration._artifact_helpers import artifact_path, read_artifact


def test_local_reference_submission_is_validated_without_external_send() -> None:
    validation = read_artifact(
        "artifacts/official/garl_release_inference_local_smoke_v1/submission_validation.json"
    )
    submission = json.loads(
        artifact_path(
            "artifacts/official/garl_release_inference_local_smoke_v1/submission.json"
        ).read_text(encoding="utf-8")
    )
    assert validation["status"] == "PASS"
    assert validation["external_submission_sent"] is False
    assert submission["meta"]["format"] == "garlttc_prediction_v1"
    assert validation["prediction_count"] == len(submission["results"])
