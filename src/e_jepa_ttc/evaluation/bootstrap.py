"""Cluster bootstrap confidence intervals that resample complete sequences."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sequence_bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sequence_ids: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int | str]:
    """Bootstrap a scalar metric by sequence, never by autocorrelated windows."""

    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    groups = np.asarray(sequence_ids).astype(str).reshape(-1)
    if not (target.shape == prediction.shape == groups.shape):
        msg = "Bootstrap targets, predictions and sequence IDs must match."
        raise ValueError(msg)
    if iterations <= 0 or not 0.0 < confidence < 1.0:
        msg = "Bootstrap iterations must be positive and confidence lie in (0, 1)."
        raise ValueError(msg)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target, prediction, groups = target[valid], prediction[valid], groups[valid]
    unique = np.unique(groups)
    if unique.size == 0:
        msg = "Sequence bootstrap has no finite samples."
        raise ValueError(msg)
    if metric is None:

        def mean_absolute_error(truth: np.ndarray, estimate: np.ndarray) -> float:
            return float(np.mean(np.abs(truth - estimate)))

        metric = mean_absolute_error
    point = float(metric(target, prediction))
    if unique.size == 1:
        return {
            "estimate": point,
            "lower": point,
            "upper": point,
            "confidence": confidence,
            "iterations": iterations,
            "sequence_count": 1,
            "status": "degenerate_single_sequence",
        }
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        values[iteration] = metric(target[selected], prediction[selected])
    alpha = (1.0 - confidence) * 0.5
    return {
        "estimate": point,
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
        "confidence": confidence,
        "iterations": iterations,
        "sequence_count": int(unique.size),
        "status": "sequence_cluster_bootstrap",
    }


def paired_sequence_bootstrap_difference(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    sequence_ids: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int | str | bool]:
    """Bootstrap candidate-minus-baseline metric differences by complete sequence."""

    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline_prediction, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_prediction, dtype=np.float64).reshape(-1)
    groups = np.asarray(sequence_ids).astype(str).reshape(-1)
    if not (target.shape == baseline.shape == candidate.shape == groups.shape):
        raise ValueError("Paired bootstrap arrays and sequence IDs must match.")
    if iterations <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Bootstrap iterations must be positive and confidence lie in (0, 1).")
    valid = np.isfinite(target) & np.isfinite(baseline) & np.isfinite(candidate)
    target, baseline, candidate, groups = (
        target[valid],
        baseline[valid],
        candidate[valid],
        groups[valid],
    )
    unique = np.unique(groups)
    if unique.size == 0:
        raise ValueError("Paired sequence bootstrap has no finite samples.")
    if metric is None:

        def mean_absolute_error(truth: np.ndarray, estimate: np.ndarray) -> float:
            return float(np.mean(np.abs(truth - estimate)))

        metric = mean_absolute_error
    point = float(metric(target, candidate) - metric(target, baseline))
    if unique.size == 1:
        return {
            "estimate": point,
            "lower": point,
            "upper": point,
            "confidence": confidence,
            "iterations": iterations,
            "sequence_count": 1,
            "candidate_better": point < 0.0,
            "status": "degenerate_single_sequence",
        }

    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        values[iteration] = metric(target[selected], candidate[selected]) - metric(
            target[selected],
            baseline[selected],
        )
    alpha = (1.0 - confidence) * 0.5
    lower = float(np.quantile(values, alpha))
    upper = float(np.quantile(values, 1.0 - alpha))
    return {
        "estimate": point,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
        "iterations": iterations,
        "sequence_count": int(unique.size),
        "candidate_better": upper < 0.0,
        "status": "paired_sequence_cluster_bootstrap",
    }


__all__ = ["paired_sequence_bootstrap_difference", "sequence_bootstrap_interval"]
