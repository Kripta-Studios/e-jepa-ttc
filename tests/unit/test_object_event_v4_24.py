from __future__ import annotations

from e_jepa_ttc.training.object_event_v4_24 import (
    CandidateMetrics,
    SelectionConfig,
    candidate_eligible,
    candidate_objective,
    rank_candidates,
)


def _good(**changes: float) -> CandidateMetrics:
    values = dict(
        pearson=0.82,
        positive_accuracy=0.90,
        negative_accuracy=0.72,
        balanced_sign_accuracy=0.81,
        minimum_sequence_pearson=0.65,
        minimum_sequence_negative_accuracy=0.55,
        predicted_negative_rate=0.28,
        true_negative_rate=0.27,
        geometry_pearson=0.35,
    )
    values.update(changes)
    return CandidateMetrics(**values)


def test_good_candidate_is_eligible() -> None:
    assert candidate_eligible(_good(), SelectionConfig())


def test_large_prior_shift_is_rejected() -> None:
    assert not candidate_eligible(_good(predicted_negative_rate=0.60), SelectionConfig())


def test_geometry_collapse_is_rejected() -> None:
    assert not candidate_eligible(_good(geometry_pearson=0.05), SelectionConfig())


def test_objective_rewards_worst_sequence_negative_accuracy() -> None:
    weak = _good(minimum_sequence_negative_accuracy=0.10)
    strong = _good(minimum_sequence_negative_accuracy=0.70)
    assert candidate_objective(strong) > candidate_objective(weak)


def test_objective_penalises_prior_drift() -> None:
    aligned = _good(predicted_negative_rate=0.28)
    shifted = _good(predicted_negative_rate=0.42)
    assert candidate_objective(aligned) > candidate_objective(shifted)


def test_ranking_prefers_eligible_candidate_even_over_ineligible_high_score() -> None:
    eligible = _good(pearson=0.75)
    ineligible = _good(pearson=0.95, geometry_pearson=0.05)
    rows = rank_candidates({"eligible": eligible, "ineligible": ineligible}, SelectionConfig())
    assert rows[0]["arm"] == "eligible"
    assert rows[0]["eligible"] is True
