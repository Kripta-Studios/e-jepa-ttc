from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest


def _rows(prediction: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_token": ["a", "b", "c", "d"],
            "sequence_id": ["s0", "s0", "s1", "s1"],
            "track_id": ["t0", "t0", "t1", "t1"],
            "target_ttc_s": [1.0, 4.0, 7.0, -1.0],
            "prediction_ttc_s": prediction,
        }
    )


def test_grouped_pair_uses_neutral_labels_and_binds_protocol(tmp_path: Path) -> None:
    from scripts.paired_grouped_bootstrap import run

    first = tmp_path / "a8.csv"
    second = tmp_path / "a6.csv"
    metadata = tmp_path / "metadata.csv"
    protocol = tmp_path / "protocol.json"
    output = tmp_path / "paired.json"
    _rows([1.0, 4.0, 7.0, -1.0]).to_csv(first, index=False)
    _rows([1.2, 4.2, 7.2, -1.2]).to_csv(second, index=False)
    _rows([1.0, 4.0, 7.0, -1.0])[
        ["sample_token", "sequence_id", "track_id"]
    ].to_csv(metadata, index=False)
    protocol.write_text('{"artifact_type":"grouped"}\n', encoding="utf-8")

    result = run(
        first,
        second,
        output,
        first_label="a8_0",
        second_label="a6",
        fold=0,
        resamples=20,
        seed=3,
        cluster_metadata=metadata,
        protocol=protocol,
    )

    assert result["artifact_type"] == "scientific_recovery_v5_grouped_paired_bootstrap_v1"
    assert result["comparison"] == {"first": "a8_0", "second": "a6"}
    assert result["fold"] == 0
    assert result["delta_first_minus_second"]["sequence_macro_MiD"] < 0
    assert result["checks"]["private_test_opened"] is False
    assert result["sources"]["protocol"]["sha256"] == hashlib.sha256(
        protocol.read_bytes()
    ).hexdigest()


def test_grouped_pair_rejects_invalid_label(tmp_path: Path) -> None:
    from scripts.paired_grouped_bootstrap import run

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    metadata = tmp_path / "metadata.csv"
    protocol = tmp_path / "protocol.json"
    _rows([1.0, 4.0, 7.0, -1.0]).to_csv(first, index=False)
    _rows([1.1, 4.1, 7.1, -1.1]).to_csv(second, index=False)
    _rows([1.0, 4.0, 7.0, -1.0])[
        ["sample_token", "sequence_id", "track_id"]
    ].to_csv(metadata, index=False)
    protocol.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        run(
            first,
            second,
            tmp_path / "out.json",
            first_label="A8 unsafe",
            second_label="a6",
            fold=0,
            resamples=5,
            seed=1,
            cluster_metadata=metadata,
            protocol=protocol,
        )


def test_align_fold_predictions_rejects_population_or_target_mismatch() -> None:
    from scripts.aggregate_v5_fold_results import align_fold_predictions

    reference = _rows([1.0, 4.0, 7.0, -1.0])
    changed_population = reference.iloc[:-1].copy()
    with pytest.raises(ValueError, match="sample-token population"):
        align_fold_predictions({"a6": reference, "a8_0": changed_population})

    changed_target = reference.copy()
    changed_target.loc[0, "target_ttc_s"] = 2.0
    with pytest.raises(ValueError, match="targets differ"):
        align_fold_predictions({"a6": reference, "a8_0": changed_target})


def test_align_fold_predictions_preserves_nan_rows() -> None:
    from scripts.aggregate_v5_fold_results import align_fold_predictions

    a6 = _rows([1.0, float("nan"), 7.0, -1.0])
    a8 = _rows([1.0, 4.0, 7.0, -1.0])
    aligned = align_fold_predictions({"a6": a6, "a8_0": a8})
    assert len(aligned["a6"]) == 4
    assert aligned["a6"]["prediction_ttc_s"].isna().sum() == 1
