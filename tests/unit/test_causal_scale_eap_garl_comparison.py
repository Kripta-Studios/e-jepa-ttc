from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import scripts.build_causal_scale_eap_garl_comparison as comparison
from e_jepa_ttc.artifacts.hashing import verify_artifact_hash


def _inputs(tmp_path: Path) -> dict[str, Any]:
    rows = []
    labels = []
    causal = []
    release = []
    targets = [1.0, 4.0, 8.0, -2.0]
    for sequence in ("seq-a", "seq-b"):
        for index, target in enumerate(targets):
            token = f"{sequence}-{index}"
            base = {
                "sample_token": token,
                "sequence_id": sequence,
                "track_id": f"{sequence}-track",
                "public_track_id": f"{sequence}-track",
                "timestamp_us": index,
            }
            rows.append(base)
            labels.append({**base, "ttc": target})
            causal.append(
                {
                    "sample_token": token,
                    "sequence_id": sequence,
                    "target_ttc_s": target,
                    "prediction_ttc_s": target * 1.5,
                }
            )
            release.append(
                {
                    "sample_token": token,
                    "sequence_id": sequence,
                    "target_ttc_s": target,
                    "predicted_ttc_s": target,
                }
            )
    paths = {
        "causal_predictions": tmp_path / "causal.csv",
        "causal_summary": tmp_path / "summary.json",
        "release_predictions": tmp_path / "release.parquet",
        "release_metrics": tmp_path / "release_metrics.json",
        "subset_data": tmp_path / "data.parquet",
        "subset_labels": tmp_path / "labels.parquet",
        "subset_manifest": tmp_path / "manifest.json",
        "official_train_assets": tmp_path / "train.txt",
        "official_train_labels": tmp_path / "official_train.parquet",
        "official_config": tmp_path / "event.yaml",
        "official_checkpoint": tmp_path / "event.pth",
        "output_json": tmp_path / "comparison.json",
        "outliers_csv": tmp_path / "outliers.csv",
    }
    pd.DataFrame(causal).to_csv(paths["causal_predictions"], index=False)
    pd.DataFrame(release).to_parquet(paths["release_predictions"], index=False)
    pd.DataFrame(rows).to_parquet(paths["subset_data"], index=False)
    pd.DataFrame(labels).to_parquet(paths["subset_labels"], index=False)
    pd.DataFrame(labels).to_parquet(paths["official_train_labels"], index=False)
    paths["causal_summary"].write_text(
        json.dumps(
            {
                "history": [{}, {}],
                "selection": {"best_epoch": 1},
                "elapsed_seconds": 2.0,
                "peak_vram_mb": 3.0,
                "validation_metrics": {"weak_bbox_iou": 0.4, "log_ratio_pearson": 0.0},
            }
        ),
        encoding="utf-8",
    )
    paths["release_metrics"].write_text(
        json.dumps({"artifact_type": "official_garl_validation_evaluation_v1"}),
        encoding="utf-8",
    )
    paths["subset_manifest"].write_text(
        json.dumps({"sample_tokens_sha256": "a" * 64}), encoding="utf-8"
    )
    paths["official_train_assets"].write_text("seq-a\nseq-b\n", encoding="utf-8")
    paths["official_config"].write_text("model: fixture\n", encoding="utf-8")
    paths["official_checkpoint"].write_bytes(b"checkpoint")
    return paths


def test_build_comparison_is_signed_token_exact_and_sequence_bootstrapped(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    result = comparison.build_comparison(**paths, bootstrap_iterations=100, bootstrap_seed=3)

    assert verify_artifact_hash(result)
    assert result["scope"]["exact_token_equality_verified"] is True
    assert result["release_reference"]["exposure_audit"][
        "all_validation_sequences_exposed"
    ] is True
    paired = result["release_reference"]["paired"]
    assert paired["window_level_bootstrap_used"] is False
    assert paired["causal_minus_release_sequence_bootstrap_paper_MiD"][
        "sequence_count"
    ] == 2
    assert result["matched_training"]["status"] == "pending"
    assert paths["output_json"].is_file()
    assert paths["outliers_csv"].is_file()


def test_build_comparison_rejects_nonidentical_token_sets(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    causal = pd.read_csv(paths["causal_predictions"]).iloc[:-1]
    causal.to_csv(paths["causal_predictions"], index=False)

    with pytest.raises(ValueError, match="token sets are not exactly equal"):
        comparison.build_comparison(**paths, bootstrap_iterations=100)


def test_build_comparison_adds_exact_matched_training_table(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    matched_predictions = tmp_path / "matched.parquet"
    matched_summary = tmp_path / "matched_summary.json"
    matched = pd.read_parquet(paths["release_predictions"])
    matched["predicted_ttc_s"] = matched["target_ttc_s"] * 1.2
    matched.to_parquet(matched_predictions, index=False)
    matched_summary.write_text(
        json.dumps(
            {
                "history": [{}, {}, {}],
                "protocol": {"seed": 7},
                "selection": {"best_epoch": 2},
                "timing": {"training_and_validation_elapsed_seconds": 12.0},
                "resources": {"peak_vram_mb": 4.0, "parameter_count": 10},
            }
        ),
        encoding="utf-8",
    )

    result = comparison.build_comparison(
        **paths,
        matched_predictions=matched_predictions,
        matched_summary=matched_summary,
        bootstrap_iterations=100,
        bootstrap_seed=3,
    )

    assert verify_artifact_hash(result)
    assert result["status"] == "release_reference_and_matched_training_complete"
    assert result["matched_training"]["status"] == "complete"
    assert result["matched_training"]["training_budget"]["selected_epoch"] == 2
    assert result["matched_training"]["paired"][
        "causal_minus_matched_sequence_bootstrap_paper_MiD"
    ]["sequence_count"] == 2
    outliers = pd.read_csv(paths["outliers_csv"])
    assert "matched_mid_per_sample" in outliers.columns
    assert b"\r\n" not in paths["output_json"].read_bytes()
    assert b"\r\n" not in paths["outliers_csv"].read_bytes()


def test_build_comparison_uses_explicit_candidate_label(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)

    result = comparison.build_comparison(
        **paths,
        bootstrap_iterations=100,
        candidate_label="causal_scale_a1_geometry",
    )

    assert result["scope"]["candidate_label"] == "causal_scale_a1_geometry"
    assert "causal_scale_a1_geometry" in result["release_reference"]
    assert "causal_scale_a0" not in result["release_reference"]
    assert result["diagnosis"]["candidate_label"] == "causal_scale_a1_geometry"


def test_build_comparison_rejects_unsafe_candidate_label(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)

    with pytest.raises(ValueError, match="candidate_label"):
        comparison.build_comparison(
            **paths,
            bootstrap_iterations=100,
            candidate_label="A1 geometry",
        )
