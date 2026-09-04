from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.evaluation.stage63_65 import (
    benchmark_phase,
    next_protocol_action,
    paired_hierarchical_bootstrap,
    scientific_ttc,
    strict_macro_mass,
    strict_score,
)


def _frame(offset: float = 0.0) -> pd.DataFrame:
    rows = []
    targets = [1.0, 3.0, 4.0, 6.0, 8.0, 10.0, -1.0, -9.9]
    for sequence in range(9):
        for index, target in enumerate(targets):
            rows.append(
                {
                    "sample_token": f"s{sequence}-{index}",
                    "sequence_id": f"s{sequence}",
                    "track_id": f"s{sequence}-track",
                    "target_ttc_s": target,
                    "prediction_ttc_s": target * (1.0 + offset),
                }
            )
    return pd.DataFrame(rows)


def test_phase_roundtrip_and_strict_boundaries() -> None:
    target = np.array([-10.0, -0.2, 0.10001, 3.0, 6.0, 10.0])
    np.testing.assert_allclose(scientific_ttc(benchmark_phase(target)), target)
    with pytest.raises(ValueError):
        scientific_ttc(np.array([0.0]))


def test_macro_mass_has_no_batch_renormalization() -> None:
    frame = _frame()
    mass = strict_macro_mass(frame.target_ttc_s.to_numpy(), frame.sequence_id.to_numpy())
    assert mass.sum() == pytest.approx(1.0)
    error = np.arange(len(frame), dtype=float)
    assert np.mean(len(frame) * mass * error) == pytest.approx(np.dot(mass, error))
    with pytest.raises(ValueError):
        strict_score(frame.loc[frame.target_ttc_s > 0])


def test_identical_arms_bootstrap_to_zero() -> None:
    frame = _frame()
    result = paired_hierarchical_bootstrap(frame, frame, valid_draws=128, max_attempts=512)
    assert result.point_delta == 0
    assert result.ci95_low == result.ci95_high == 0


def test_state_machine_branches_are_non_rescuing() -> None:
    assert (
        next_protocol_action(
            stage63_integrity=False,
            stage63_training_ready=False,
            raw_available_or_supported=False,
            stage64_seed7=None,
        )
        == "STOP_INTEGRITY_BLOCKED"
    )
    assert (
        next_protocol_action(
            stage63_integrity=True,
            stage63_training_ready=False,
            raw_available_or_supported=False,
            stage64_seed7=None,
        )
        == "RUN_STAGE65"
    )
    assert (
        next_protocol_action(
            stage63_integrity=True,
            stage63_training_ready=True,
            raw_available_or_supported=True,
            stage64_seed7="RAW_ALL_GATES_PASSED",
            replication={13: "RAW_GATES_FAILED", 23: "RAW_ALL_GATES_PASSED"},
        )
        == "STOP_RAW_REPLICATION_NOT_CONFIRMED"
    )
