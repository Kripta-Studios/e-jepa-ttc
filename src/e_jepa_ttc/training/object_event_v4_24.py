"""Train-only candidate ranking for Object Event TTC v4.24."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class CandidateMetrics:
    pearson: float
    positive_accuracy: float
    negative_accuracy: float
    balanced_sign_accuracy: float
    minimum_sequence_pearson: float
    minimum_sequence_negative_accuracy: float
    predicted_negative_rate: float
    true_negative_rate: float
    geometry_pearson: float


@dataclass(frozen=True)
class SelectionConfig:
    minimum_pearson: float = 0.65
    minimum_balanced_sign: float = 0.72
    minimum_positive_accuracy: float = 0.82
    minimum_negative_accuracy: float = 0.55
    minimum_sequence_pearson: float = 0.45
    minimum_geometry_pearson: float = 0.20
    maximum_negative_prior_gap: float = 0.16


def candidate_objective(metrics: CandidateMetrics) -> float:
    """Prefer cross-sequence TTC quality, sign balance and retained geometry."""
    prior_gap = abs(metrics.predicted_negative_rate - metrics.true_negative_rate)
    return float(
        metrics.pearson
        + 0.20 * metrics.balanced_sign_accuracy
        + 0.12 * metrics.minimum_sequence_pearson
        + 0.10 * metrics.minimum_sequence_negative_accuracy
        + 0.12 * metrics.geometry_pearson
        - 0.15 * prior_gap
    )


def candidate_eligible(metrics: CandidateMetrics, config: SelectionConfig) -> bool:
    return bool(
        metrics.pearson >= config.minimum_pearson
        and metrics.balanced_sign_accuracy >= config.minimum_balanced_sign
        and metrics.positive_accuracy >= config.minimum_positive_accuracy
        and metrics.negative_accuracy >= config.minimum_negative_accuracy
        and metrics.minimum_sequence_pearson >= config.minimum_sequence_pearson
        and metrics.geometry_pearson >= config.minimum_geometry_pearson
        and abs(metrics.predicted_negative_rate - metrics.true_negative_rate) <= config.maximum_negative_prior_gap
    )


def rank_candidates(
    metrics_by_name: Mapping[str, CandidateMetrics],
    config: SelectionConfig,
) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    for name, metrics in metrics_by_name.items():
        rows.append({
            "arm": str(name),
            "eligible": candidate_eligible(metrics, config),
            "objective": candidate_objective(metrics),
            "pearson": metrics.pearson,
            "balanced_sign_accuracy": metrics.balanced_sign_accuracy,
            "positive_accuracy": metrics.positive_accuracy,
            "negative_accuracy": metrics.negative_accuracy,
            "minimum_sequence_pearson": metrics.minimum_sequence_pearson,
            "minimum_sequence_negative_accuracy": metrics.minimum_sequence_negative_accuracy,
            "geometry_pearson": metrics.geometry_pearson,
            "negative_prior_gap": abs(metrics.predicted_negative_rate - metrics.true_negative_rate),
        })
    rows.sort(key=lambda row: (bool(row["eligible"]), float(row["objective"])), reverse=True)
    return rows


__all__ = [
    "CandidateMetrics",
    "SelectionConfig",
    "candidate_eligible",
    "candidate_objective",
    "rank_candidates",
]
