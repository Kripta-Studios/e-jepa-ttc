from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.evaluation.collision_clock_protocol import (
    canonical_records_hash,
    precheck_production_oof,
    production_sequence_macro_metrics,
)


def _frame() -> pd.DataFrame:
    sequences = [f"sequence-{index}" for index in range(9)]
    targets = (-1.0, 1.0, 4.0, 8.0)
    rows = []
    for index in range(8192):
        target = targets[(index // 9) % len(targets)]
        rows.append(
            {
                "sample_token": f"token-{index:05d}",
                "sequence_id": sequences[index % len(sequences)],
                "track_id": f"track-{index // 4:05d}",
                "outer_fold": index % 3,
                "target_ttc": target,
                "ttc_prediction_s": target * 1.01,
                "sample_weight": 1.0 / 8192.0,
            }
        )
    return pd.DataFrame(rows)


def _hashes(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "identity_sha256": canonical_records_hash(
            frame, ("sample_token", "sequence_id", "track_id")
        ),
        "target_sha256": canonical_records_hash(frame, ("sample_token", "target_ttc")),
        "fold_sha256": canonical_records_hash(frame, ("sample_token", "outer_fold")),
        "weight_sha256": canonical_records_hash(frame, ("sample_token", "sample_weight")),
    }


def test_complete_production_precheck_and_macro_are_finite() -> None:
    frame = _frame()
    sequences = sorted(frame["sequence_id"].unique())
    checked = precheck_production_oof(
        frame,
        expected_hashes=_hashes(frame),
        required_sequences=sequences,
    )
    assert len(checked) == 8192
    assert np.isfinite(checked["mid_per_row"]).all()
    metrics = production_sequence_macro_metrics(
        frame,
        expected_hashes=_hashes(frame),
        required_sequences=sequences,
    )
    assert np.isfinite(metrics["sequence_macro_paper_MiD_overall"])


def test_smoke_is_not_accepted_by_production_aggregator() -> None:
    frame = _frame().iloc[:32].copy()
    with pytest.raises(ValueError, match="exactly 8192"):
        precheck_production_oof(
            frame,
            expected_hashes=_hashes(frame),
            required_sequences=sorted(frame["sequence_id"].unique()),
        )


@pytest.mark.parametrize("defect", ["fold", "duplicate", "prediction_nan", "weight_nan"])
def test_production_precheck_rejects_integrity_and_finiteness_defects(defect: str) -> None:
    frame = _frame()
    hashes = _hashes(frame)
    if defect == "fold":
        frame["outer_fold"] = 0
    elif defect == "duplicate":
        frame.loc[1, "sample_token"] = frame.loc[0, "sample_token"]
    elif defect == "prediction_nan":
        frame.loc[0, "ttc_prediction_s"] = np.nan
    elif defect == "weight_nan":
        frame.loc[0, "sample_weight"] = np.inf
    with pytest.raises(ValueError):
        precheck_production_oof(
            frame,
            expected_hashes=hashes,
            required_sequences=sorted(frame["sequence_id"].unique()),
        )


def test_production_precheck_rejects_hash_and_required_sequence_mismatch() -> None:
    frame = _frame()
    hashes = _hashes(frame)
    hashes["target_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target_sha256 mismatch"):
        precheck_production_oof(
            frame,
            expected_hashes=hashes,
            required_sequences=sorted(frame["sequence_id"].unique()),
        )
    with pytest.raises(ValueError, match="sequence universe mismatch"):
        precheck_production_oof(
            frame,
            expected_hashes=_hashes(frame),
            required_sequences=["missing"],
        )


def test_production_precheck_rejects_missing_bucket_within_sequence() -> None:
    frame = _frame()
    sequence = "sequence-0"
    mask = (frame["sequence_id"] == sequence) & (frame["target_ttc"] == 8.0)
    frame.loc[mask, "target_ttc"] = 4.0
    frame.loc[mask, "ttc_prediction_s"] = 4.04
    with pytest.raises(ValueError, match="lacks a required TTC bucket"):
        precheck_production_oof(
            frame,
            expected_hashes=_hashes(frame),
            required_sequences=sorted(frame["sequence_id"].unique()),
        )
