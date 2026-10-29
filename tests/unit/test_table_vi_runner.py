from __future__ import annotations

import json

import pytest

from scripts.evaluate_garl_evttc_table_vi import score


def _write(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_table_vi_score_requires_label_free_predictions(tmp_path) -> None:
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    _write(
        predictions,
        {
            "predictions": [
                {
                    "sequence_id": "s",
                    "sample_token": "t",
                    "track_id": "k",
                    "timestamp_us": 1,
                    "predicted_ttc_s": 1.0,
                    "target_ttc_s": 1.0,
                }
            ]
        },
    )
    _write(
        targets,
        {
            "targets": [
                {
                    "sequence_id": "s",
                    "sample_token": "t",
                    "track_id": "k",
                    "timestamp_us": 1,
                    "target_ttc_s": 1.0,
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="forbidden labels"):
        score(predictions, targets, tmp_path / "out.json")


def test_table_vi_score_uses_separate_targets(tmp_path) -> None:
    predictions = tmp_path / "predictions.json"
    targets = tmp_path / "targets.json"
    output = tmp_path / "out.json"
    identity = {
        "sequence_id": "s",
        "sample_token": "t",
        "track_id": "k",
        "timestamp_us": 1,
    }
    _write(predictions, {"predictions": [{**identity, "predicted_ttc_s": 1.0}]})
    _write(targets, {"targets": [{**identity, "target_ttc_s": 1.0}]})
    result = score(predictions, targets, output)
    assert result["sample_count"] == 1
    assert result["training_updates_on_target_dataset"] == 0
    assert output.is_file()
