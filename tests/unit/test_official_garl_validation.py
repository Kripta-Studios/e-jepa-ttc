from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import scripts.evaluate_official_garl_validation as evaluator


def _release(tmp_path: Path) -> tuple[Path, Path, Path]:
    release = tmp_path / "release"
    config = release / "configs" / "garl_ttc_eventdecoder.yaml"
    checkpoint = release / "checkpoints" / "paper_ours_full.pth"
    network = release / "garl_ttc" / "models" / "ttc_network.py"
    dataset = release / "garl_ttc" / "datasets" / "ttc_dataset.py"
    for path, content in (
        (config, "model:\n  mode: height_ratio\n"),
        (checkpoint, "checkpoint"),
        (network, "# immutable network\n"),
        (dataset, "# immutable dataset\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return release, config, checkpoint


def _split(tmp_path: Path) -> tuple[Path, Path, Path]:
    keys = {
        "sequence_id": ["seq-a", "seq-b", "seq-a", "unused"],
        "sample_token": ["a-1", "b-1", "a-2", "x-1"],
        "track_id": ["ta", "tb", "ta", "tx"],
        "public_track_id": ["pa", "pb", "pa", "px"],
        "timestamp_us": [1, 2, 3, 4],
    }
    data = pd.DataFrame(keys)
    labels = pd.DataFrame({**keys, "ttc": [1.0, -2.0, 4.0, 8.0]})
    data_path = tmp_path / "validation_data.parquet"
    labels_path = tmp_path / "validation_labels.parquet"
    assets_path = tmp_path / "validation.txt"
    data.to_parquet(data_path, index=False)
    labels.to_parquet(labels_path, index=False)
    assets_path.write_text("seq-a\nseq-b\n", encoding="utf-8")
    return data_path, labels_path, assets_path


def test_evaluate_writes_strict_predictions_metrics_and_provenance(tmp_path: Path) -> None:
    release, config, checkpoint = _release(tmp_path)
    data, labels, assets = _split(tmp_path)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in release.rglob("*")
        if path.is_file()
    }

    def fake_inference(**kwargs: Any) -> pd.DataFrame:
        assert kwargs["release_root"] == release
        return pd.DataFrame(
            {
                "sample_token": ["a-2", "a-1", "b-1"],
                "target_from_loader_ttc_s": [4.0, 1.0, -2.0],
                "predicted_ttc_s": [4.0, 1.0, -2.0],
            }
        )

    output = tmp_path / "output"
    report = evaluator.evaluate(
        release_root=release,
        config_path=config,
        checkpoint=checkpoint,
        dataset_root=tmp_path,
        data_parquet=data,
        labels_parquet=labels,
        asset_list=assets,
        output_dir=output,
        inference_runner=fake_inference,
    )

    assert report["status"] == "completed"
    assert report["signed_garl_metrics"]["failure_count"] == 0
    assert report["sequence_macro_signed_metrics"]["per_sequence"].keys() == {
        "seq-a",
        "seq-b",
    }
    assert report["provenance"]["validation_only"] is True
    assert report["provenance"]["test_data_used"] is False
    assert report["provenance"]["sample_count"] == 3
    assert (output / "predictions.parquet").is_file()
    assert (output / "metrics.json").is_file()
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert set(predictions["sample_token"]) == {"a-1", "a-2", "b-1"}
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in release.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_evaluate_rejects_partial_predictions(tmp_path: Path) -> None:
    release, config, checkpoint = _release(tmp_path)
    data, labels, assets = _split(tmp_path)

    def partial_inference(**kwargs: Any) -> pd.DataFrame:
        del kwargs
        return pd.DataFrame(
            {
                "sample_token": ["a-1"],
                "target_from_loader_ttc_s": [1.0],
                "predicted_ttc_s": [1.0],
            }
        )

    with pytest.raises(ValueError, match="token set mismatch"):
        evaluator.evaluate(
            release_root=release,
            config_path=config,
            checkpoint=checkpoint,
            dataset_root=tmp_path,
            data_parquet=data,
            labels_parquet=labels,
            asset_list=assets,
            output_dir=tmp_path / "output",
            inference_runner=partial_inference,
        )


def test_main_returns_nonzero_when_validation_fails(tmp_path: Path) -> None:
    release, _, _ = _release(tmp_path)
    exit_code = evaluator.main(
        [
            "--release-root",
            str(release),
            "--dataset-root",
            str(tmp_path),
            "--data-parquet",
            str(tmp_path / "missing.parquet"),
            "--labels-parquet",
            str(tmp_path / "missing-labels.parquet"),
            "--asset-list",
            str(tmp_path / "missing-assets.txt"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert exit_code != 0
