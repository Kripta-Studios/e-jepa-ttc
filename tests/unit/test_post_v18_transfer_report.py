"""Protect timestamp matching, cohort coverage and paired group inference."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.post_v18_transfer_inputs import sha, write
from scripts.post_v18_transfer_report import contrasts, match_target, sealed, summary


def test_nearest_target_uses_fractional_seconds() -> None:
    table = np.array([[0, 9.601, 0, 0, 3], [1, 9.611, 0, 0, 2]])
    assert match_target(table, 9.609)["target_ttc"] == 2
    assert match_target(table, 9.602)["target_ttc"] == 3


def test_target_matching_limit_zero_and_tie() -> None:
    table = np.array([[0, 1.0, 0, 0, 3], [1, 1.25, 0, 0, 2]])
    assert match_target(table, 1.05)["target_ttc"] == 3
    assert match_target(table, 1.051)["target_exclusion"] == "TTC_TIMESTAMP_MISMATCH"
    assert match_target(table, 1.125)["target_timestamp_s"] == 1.0
    table[0, 4] = 0
    assert match_target(table, 1.0)["target_exclusion"] == "TTC_INVALID_OR_ZERO"


def test_unsorted_target_clock_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        match_target(np.array([[0, 2, 0, 0, 3], [1, 1, 0, 0, 2]]), 1.1)


def test_missing_prediction_cannot_improve_complete_cohort_score() -> None:
    frame = pd.DataFrame({"target_ttc": [1, 2], "m": [1, np.nan], "group_id": ["a", "b"]})
    result = summary(frame, "m")
    assert result["coverage"] == 0.5
    assert result["complete_cohort"] is None
    assert result["finite_only_conditional"]["MAE_s"] == 0


def test_bootstrap_equal_groups_not_independent_windows() -> None:
    frame = pd.DataFrame(
        {"target_ttc": [1, 1], "a": [1, 3], "b": [2, 1], "group_id": ["short", "long"]}
    )
    first = contrasts(frame, [("a", "b")])[0]
    repeated = pd.concat([frame, *[frame.iloc[[1]]] * 30], ignore_index=True)
    second = contrasts(repeated, [("a", "b")])[0]
    assert first == second
    assert first["MAE_s"]["point"] == 0.5


def test_modified_prediction_seal_rejected(tmp_path: Path) -> None:
    registration = tmp_path / "REGISTRATION.json"
    write(registration, {"optimizer_updates": 0})
    query = tmp_path / "query_0000.json"
    write(query, {"registration_sha256": sha(registration), "targets_read": False})
    write(
        tmp_path / "SEALED.json",
        {"registration_sha256": sha(registration), "queries": 1, "files": {query.name: sha(query)}},
    )
    assert len(sealed(tmp_path, registration)) == 1
    write(query, {"registration_sha256": sha(registration), "targets_read": False, "edited": True})
    with pytest.raises(ValueError, match="Prediction changed"):
        sealed(tmp_path, registration)
