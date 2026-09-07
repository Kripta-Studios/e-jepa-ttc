"""Guardrail measurements from synthetic paired TTC, without optimizer updates."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, scientific_ttc
from e_jepa_ttc.simplex_t.practical_comparison import paired_practical_comparison


@pytest.fixture
def frames():
    ids = np.arange(8192)
    target = np.resize(np.array([-2.0, 1.0, 4.0, 10.0]), len(ids))
    identity = pd.DataFrame(
        {
            "sample_token": [f"q{i}" for i in ids],
            "sequence_id": [f"s{i % 9}" for i in ids],
            "track_id": [f"t{i}" for i in ids],
            "outer_fold": ids % 3,
            "target_ttc": target,
        }
    )
    phase = benchmark_phase(target)
    return (
        identity.assign(prediction_ttc_s=scientific_ttc(phase + 0.0001), loss=99999.0),
        identity.assign(prediction_ttc_s=scientific_ttc(phase + 0.001), loss=0.0),
    )


def test_metrics_recomputed_from_ttc_not_supplied_losses_and_rows_are_paired(frames):
    candidate, reference = frames
    result = paired_practical_comparison(candidate, reference.iloc[::-1])
    assert result["point_delta"] == pytest.approx(-9.0)
    assert result["crucial_bucket_mid_delta"] == pytest.approx(-9.0)
    assert result["weighted_sign_error_delta"] == 0.0
    assert result["sequence_wins"] == result["sequence_count"] == 9
    assert result["candidate_finite_fraction"] == 1.0
    assert result["gate_authorized"] is False


@pytest.mark.parametrize("change", ["query", "target", "missing", "duplicate"])
def test_misaligned_cohort_cannot_supply_gate_evidence(frames, change):
    candidate, reference = frames
    if change == "query":
        reference.loc[0, "sample_token"] = "foreign"
    elif change == "target":
        reference.loc[0, "target_ttc"] = -3.0
    elif change == "missing":
        reference = reference.iloc[1:]
    else:
        reference.loc[0, "sample_token"] = reference.loc[1, "sample_token"]
    with pytest.raises(ValueError):
        paired_practical_comparison(candidate, reference)


def test_infinite_candidate_is_retained_but_cannot_claim_full_finite_coverage(frames):
    candidate, reference = frames
    candidate.loc[0, "prediction_ttc_s"] = np.inf
    result = paired_practical_comparison(candidate, reference)
    assert result["query_count"] == 8192
    assert result["candidate_finite_fraction"] == 8191 / 8192
    assert np.isfinite(result["point_delta"])
