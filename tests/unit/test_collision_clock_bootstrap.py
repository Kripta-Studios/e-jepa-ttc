from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_bootstrap import (
    paired_hierarchical_mid_bootstrap,
)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    targets = (-1.0, 1.0, 4.0, 8.0)
    for sequence_index in range(3):
        for track_index in range(2):
            for bucket_index, target in enumerate(targets):
                rows.append(
                    {
                        "sample_token": (f"token-{sequence_index}-{track_index}-{bucket_index}"),
                        "sequence_id": f"sequence-{sequence_index}",
                        "track_id": f"track-{track_index}",
                        "target_ttc_s": target,
                        "scientific_mid_per_row": float(
                            5 + sequence_index + track_index + bucket_index
                        ),
                    }
                )
    candidate = pd.DataFrame(rows)
    reference = candidate.copy()
    reference["scientific_mid_per_row"] += 2.0
    return candidate, reference


def _protocol() -> dict[str, Any]:
    return sign_artifact(
        {
            "artifact_type": "scientific_recovery_v9_eclock_protocol_v2",
            "bootstrap": {
                "method": "paired_hierarchical_sequence_then_track_cluster_bootstrap",
                "seed": 20260814,
                "draws": 100,
            },
        }
    )


def _identity(family: str) -> dict[str, str]:
    return {
        "reference_family": family,
        "path": f"artifacts/{family}.csv",
        "file_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
    }


def _run(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    return paired_hierarchical_mid_bootstrap(
        candidate,
        reference,
        protocol=_protocol(),
        candidate_identity=_identity("X0-BASE-U"),
        reference_identity=_identity("official_a5_oof"),
    )


def test_paired_bootstrap_is_deterministic_cluster_aware_and_signed() -> None:
    candidate, reference = _frames()
    first = _run(candidate, reference)
    second = _run(candidate, reference)
    assert first == second
    assert verify_artifact_hash(first)
    assert first["cluster_order"] == ["sequence_id", "track_id"]
    assert first["rows_sampled_as_complete_tracks"] is True
    assert first["paired_identical_draws"] is True
    assert first["window_level_bootstrap_used"] is False
    assert first["delta_candidate_minus_reference"]["mean"] == pytest.approx(-2.0)
    assert first["delta_candidate_minus_reference"]["probability_delta_lt_zero"] == 1.0


def test_paired_bootstrap_is_row_permutation_invariant() -> None:
    candidate, reference = _frames()
    order = np.random.default_rng(41).permutation(len(candidate))
    shuffled_candidate = candidate.iloc[order].reset_index(drop=True)
    shuffled_reference = reference.iloc[order[::-1]].reset_index(drop=True)
    assert _run(candidate, reference) == _run(shuffled_candidate, shuffled_reference)


def test_paired_bootstrap_rejects_different_token_sets() -> None:
    candidate, reference = _frames()
    reference.loc[0, "sample_token"] = "different-token"
    with pytest.raises(ValueError, match="same sample tokens"):
        _run(candidate, reference)


def test_paired_bootstrap_rejects_track_identity_mismatch() -> None:
    candidate, reference = _frames()
    reference.loc[0, "track_id"] = "foreign-track"
    with pytest.raises(ValueError, match="track_id mismatch"):
        _run(candidate, reference)
