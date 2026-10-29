"""Descriptive signed-TTC TRAIN comparison; no validation selection or test access."""

from __future__ import annotations

import numpy as np


def evaluate(prediction: np.ndarray, truth: np.ndarray) -> dict:
    """Keep every query in coverage and bucket denominators, including invalid outputs."""
    if prediction.ndim != 1 or prediction.shape != truth.shape or not len(truth):
        raise ValueError("Aligned nonempty TTC vectors required")
    if not np.isfinite(truth).all() or not ((truth < 0) | (truth > 0.1)).all():
        raise ValueError("Signed benchmark targets must stay in their original phase domain")
    finite = np.isfinite(prediction)
    failed = ~finite | (np.abs(prediction) < 0.1)
    phase_valid = finite & ((prediction < 0) | (prediction > 0.1))
    phase_error = np.full(len(truth), np.nan)
    phase_error[phase_valid] = 10000 * np.abs(
        np.log1p(-0.1 / prediction[phase_valid]) - np.log1p(-0.1 / truth[phase_valid])
    )
    absolute = np.abs(prediction[finite] - truth[finite])
    buckets = {}
    for name, lower, upper, weight in (
        ("negative", -10, 0, 0.1),
        ("close", 0, 3, 0.5),
        ("short", 3, 6, 0.3),
        ("long", 6, 10, 0.1),
    ):
        mask = (truth > lower) & (truth <= upper)
        valid = mask & phase_valid
        count = int(mask.sum())
        buckets[name] = {
            "population": count,
            "benchmark_weight": weight,
            "failure_rate": float(failed[mask].mean()) if count else None,
            "phase_domain_coverage": float(phase_valid[mask].mean()) if count else None,
            "MiD_phase_valid": float(phase_error[valid].mean()) if valid.any() else None,
        }
    weighted = None
    if all(bucket["MiD_phase_valid"] is not None for bucket in buckets.values()):
        weighted = float(
            sum(
                bucket["benchmark_weight"] * bucket["MiD_phase_valid"]
                for bucket in buckets.values()
            )
        )
    return {
        "population": len(truth),
        "finite_coverage": float(finite.mean()),
        "failure_rate": float(failed.mean()),
        "phase_domain_coverage": float(phase_valid.mean()),
        "MAE_seconds_finite": float(absolute.mean()) if len(absolute) else None,
        "RMSE_seconds_finite": float(np.sqrt((absolute**2).mean())) if len(absolute) else None,
        "weighted_MiD_phase_valid": weighted,
        "buckets": buckets,
        "evaluation_role": "TRAIN_FIT_DIAGNOSTIC_NOT_GENERALIZATION",
        "not_a_reproduction_of_the_published_paper_evaluation": True,
    }
