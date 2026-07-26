from __future__ import annotations

import numpy as np

from e_jepa_ttc.evaluation.bootstrap import (
    paired_sequence_bootstrap_difference,
    sequence_bootstrap_interval,
)


def test_sequence_bootstrap_is_reproducible_and_clustered() -> None:
    truth = np.zeros(12)
    prediction = np.asarray([1.0] * 4 + [2.0] * 4 + [4.0] * 4)
    sequences = np.asarray(["a"] * 4 + ["b"] * 4 + ["c"] * 4)

    first = sequence_bootstrap_interval(
        truth,
        prediction,
        sequences,
        iterations=200,
        seed=9,
    )
    second = sequence_bootstrap_interval(
        truth,
        prediction,
        sequences,
        iterations=200,
        seed=9,
    )

    assert first == second
    assert first["sequence_count"] == 3
    assert first["lower"] <= first["estimate"] <= first["upper"]


def test_single_sequence_bootstrap_is_explicitly_degenerate() -> None:
    result = sequence_bootstrap_interval(
        np.asarray([1.0, 2.0]),
        np.asarray([1.5, 2.5]),
        np.asarray(["only", "only"]),
        iterations=10,
    )

    assert result["status"] == "degenerate_single_sequence"
    assert result["lower"] == result["upper"]


def test_paired_sequence_bootstrap_detects_consistent_improvement() -> None:
    truth = np.zeros(12)
    baseline = np.asarray([2.0] * 4 + [3.0] * 4 + [4.0] * 4)
    candidate = baseline - 1.0
    sequences = np.asarray(["a"] * 4 + ["b"] * 4 + ["c"] * 4)

    result = paired_sequence_bootstrap_difference(
        truth,
        baseline,
        candidate,
        sequences,
        iterations=200,
        seed=9,
    )

    assert result["estimate"] == -1.0
    assert result["upper"] < 0.0
    assert result["candidate_better"] is True
    assert result["status"] == "paired_sequence_cluster_bootstrap"


def test_paired_single_sequence_bootstrap_is_explicitly_degenerate() -> None:
    result = paired_sequence_bootstrap_difference(
        np.asarray([1.0, 2.0]),
        np.asarray([2.0, 3.0]),
        np.asarray([1.5, 2.5]),
        np.asarray(["only", "only"]),
        iterations=10,
    )

    assert result["status"] == "degenerate_single_sequence"
    assert result["lower"] == result["upper"] == -0.5
