"""Submission ZIP integrity without private labels or remote uploads."""

import json
import zipfile

import pandas as pd
import pytest

from operational.sota_evidence.codabench import package_predictions


def test_zip_has_exactly_one_root_json_and_preserves_predictions(tmp_path):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"test-fixture-not-a-trained-model")
    output = tmp_path / "submission.zip"
    frame = pd.DataFrame({"sample_token": ["b", "a"], "prediction": [-4.0, 2.0]})
    receipt = package_predictions(["a", "b"], frame, checkpoint, output)
    assert receipt["status"] == "PACKAGED_NOT_SUBMITTED"
    assert receipt["sample_count"] == 2
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["submission.json"]
        payload = json.loads(archive.read("submission.json"))
    assert payload["results"] == {"a": {"ttc": 2.0}, "b": {"ttc": -4.0}}
    assert payload["meta"]["format"] == "garlttc_prediction_v1"
    with pytest.raises(FileExistsError):
        package_predictions(["a", "b"], frame, checkpoint, output)


def test_missing_tokens_never_create_upload_zip(tmp_path):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"test-fixture")
    output = tmp_path / "submission.zip"
    frame = pd.DataFrame({"sample_token": ["a"], "prediction": [2.0]})
    with pytest.raises(ValueError, match="population"):
        package_predictions(["a", "b"], frame, checkpoint, output)
    assert not output.exists()


def test_json_template_is_not_a_prediction_table(tmp_path):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"test-fixture")
    with pytest.raises(ValueError, match="columns"):
        package_predictions(
            ["a"],
            pd.DataFrame({"meta": ["garlttc_prediction_v1"]}),
            checkpoint,
            tmp_path / "submission.zip",
        )
