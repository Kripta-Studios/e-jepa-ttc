from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from scripts.build_garl_validation_subset_from_predictions import build_subset


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    keys = {
        "sequence_id": ["seq-b", "seq-a", "seq-b"],
        "sample_token": ["token-0", "token-1", "token-2"],
        "track_id": ["track-0", "track-1", "track-2"],
        "public_track_id": ["track-0", "track-1", "track-2"],
        "timestamp_us": [100, 200, 300],
    }
    data = pd.DataFrame({**keys, "events_path": ["a", "b", "c"]})
    labels = pd.DataFrame({**keys, "ttc": [1.0, 4.0, -2.0]})
    predictions = pd.DataFrame(
        {
            "sample_token": ["token-2", "token-0"],
            "sequence_id": ["seq-b", "seq-b"],
            "target_ttc_s": [-2.0, 1.0],
            "prediction_ttc_s": [-1.8, 1.1],
        }
    )
    data_path = tmp_path / "data-source.parquet"
    labels_path = tmp_path / "labels-source.parquet"
    predictions_path = tmp_path / "predictions.csv"
    data.to_parquet(data_path, index=False)
    labels.to_parquet(labels_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    return predictions_path, data_path, labels_path


def test_builder_preserves_exact_prediction_tokens_and_signs_manifest(
    tmp_path: Path,
) -> None:
    predictions, data, labels = _sources(tmp_path)
    output = tmp_path / "subset"

    payload = build_subset(
        predictions_path=predictions,
        data_parquet=data,
        labels_parquet=labels,
        output_dir=output,
        expected_count=2,
    )

    assert verify_artifact_hash(payload)
    assert payload["sample_count"] == 2
    assert payload["sequence_ids"] == ["seq-b"]
    assert payload["bucket_counts"] == {
        "crucial": 1,
        "small": 0,
        "large": 0,
        "negative": 1,
    }
    assert pd.read_parquet(output / "data.parquet")["sample_token"].tolist() == [
        "token-2",
        "token-0",
    ]
    assert pd.read_parquet(output / "labels.parquet")["sample_token"].tolist() == [
        "token-2",
        "token-0",
    ]
    assert (output / "assets.txt").read_text(encoding="utf-8") == "seq-b\n"
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert verify_artifact_hash(stored)


def test_builder_rejects_target_mismatch_without_promoting_output(tmp_path: Path) -> None:
    predictions, data, labels = _sources(tmp_path)
    frame = pd.read_csv(predictions)
    frame.loc[0, "target_ttc_s"] = -3.0
    frame.to_csv(predictions, index=False)
    output = tmp_path / "subset"

    with pytest.raises(ValueError, match="TTC targets differ"):
        build_subset(
            predictions_path=predictions,
            data_parquet=data,
            labels_parquet=labels,
            output_dir=output,
            expected_count=2,
        )

    assert not output.exists()
