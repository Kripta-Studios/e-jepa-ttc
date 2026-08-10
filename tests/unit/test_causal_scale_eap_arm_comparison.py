from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_causal_scale_eap_arm_comparison as comparison
from e_jepa_ttc.artifacts.hashing import verify_artifact_hash


def _inputs(tmp_path: Path) -> dict[str, Path]:
    reference: list[dict[str, object]] = []
    candidate: list[dict[str, object]] = []
    for sequence in ("a", "b"):
        for index, target in enumerate((1.0, 4.0, 8.0, -2.0)):
            base = {
                "sample_token": f"{sequence}-{index}",
                "sequence_id": sequence,
                "target_ttc_s": target,
            }
            reference.append({**base, "prediction_ttc_s": target * 1.5})
            candidate.append({**base, "prediction_ttc_s": target})
    paths = {
        "reference_predictions": tmp_path / "reference.csv",
        "reference_summary": tmp_path / "reference.json",
        "candidate_predictions": tmp_path / "candidate.csv",
        "candidate_summary": tmp_path / "candidate.json",
    }
    pd.DataFrame(reference).to_csv(paths["reference_predictions"], index=False)
    pd.DataFrame(candidate).to_csv(paths["candidate_predictions"], index=False)
    paths["reference_summary"].write_text(json.dumps({"arm": "reference"}))
    paths["candidate_summary"].write_text(json.dumps({"arm": "candidate"}))
    return paths


def test_arm_comparison_is_token_exact_signed_and_sequence_bootstrapped(
    tmp_path: Path,
) -> None:
    result = comparison.build_comparison(
        **_inputs(tmp_path),
        output_json=tmp_path / "comparison.json",
        reference_label="a1_event_only_pure",
        candidate_label="a3_rgb_distilled",
        bootstrap_iterations=100,
        bootstrap_seed=3,
    )
    assert verify_artifact_hash(result)
    assert result["scope"]["exact_token_equality_verified"] is True
    assert result["paired"]["bootstrap_unit"] == "complete_sequence"
    assert result["paired"]["candidate_win_rate"] == 1.0
    bootstrap = result["paired"][
        "candidate_minus_reference_sequence_bootstrap_paper_MiD"
    ]
    assert bootstrap["candidate_better"] is True


def test_arm_comparison_rejects_token_mismatch(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    frame = pd.read_csv(paths["candidate_predictions"])
    frame.loc[0, "sample_token"] = "wrong"
    frame.to_csv(paths["candidate_predictions"], index=False)
    with pytest.raises(ValueError, match="token sets differ"):
        comparison.build_comparison(
            **paths,
            output_json=tmp_path / "comparison.json",
            reference_label="reference",
            candidate_label="candidate",
            bootstrap_iterations=10,
        )
