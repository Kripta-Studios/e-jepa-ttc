"""Distinct paper metrics with strict failure accounting and sequence aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np


def bands(truth: np.ndarray, definition: str) -> np.ndarray:
    """Preserve the different interval endpoints in the papers and release code."""
    result = np.full(truth.shape, "outside", dtype="U7")
    if definition in {"garl", "garl_code"}:
        result[(truth > 0) & (truth <= 3)] = "c"
        result[(truth > 3) & (truth <= 6)] = "s"
        result[(truth > 6) & (truth <= 10)] = "l"
        negative = (truth >= -10) & (truth < 0)
        if definition == "garl_code":
            negative &= truth > -10
        result[negative] = "n"
    elif definition == "react":
        result[(truth > 0) & (truth < 3)] = "c"
        result[(truth >= 3) & (truth < 6)] = "s"
        result[(truth >= 6) & (truth < 10)] = "l"
    else:
        raise ValueError(f"unknown band definition: {definition}")
    return result


def observations(truth: np.ndarray, prediction: np.ndarray, dt: float = 0.1) -> dict:
    """Calculate both MiD definitions; never turn an invalid prediction into success.

    `garl_reference_mid` intentionally mirrors the public evaluator's infinity
    behavior for parity auditing. It is not the strict metric used for claims.
    """
    if truth.ndim != 1 or truth.shape != prediction.shape:
        raise ValueError("aligned one-dimensional arrays required")
    if not np.isfinite(truth).all() or (truth == 0).any():
        raise ValueError("apply GT-only finite/nonzero eligibility first")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("positive finite dt required")
    finite = np.isfinite(prediction)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gt_ratio = 1 - dt / truth
        pred_ratio = np.where(prediction != 0, 1 - dt / prediction, np.nan)
        reference_mid = np.where(
            (gt_ratio > 0) & (pred_ratio > 0) & np.isfinite(pred_ratio),
            np.abs(np.log(gt_ratio) - np.log(pred_ratio)) * 1e4,
            np.nan,
        )
        logratio = np.where(
            (truth > 0) & (prediction > 0) & finite,
            np.abs(np.log(prediction) - np.log(truth)),
            np.nan,
        )
        rte = np.where(finite, np.abs(prediction - truth) / np.abs(truth) * 100, np.nan)
    return {
        "rte_percent": rte,
        "garl_reference_mid": reference_mid,
        "garl_mid": np.where(finite, reference_mid, np.nan),
        "react_logratio": logratio,
        "garl_reference_failed": ~finite | (np.abs(prediction) < dt),
    }


def strict_mean(values: np.ndarray) -> float | None:
    """A full-cohort mean is unavailable if even one required observation fails."""
    return float(values.mean()) if len(values) and np.isfinite(values).all() else None


def _weighted(values: dict, weights: dict) -> float | None:
    if any(values[k] is None for k in weights):
        return None
    return float(sum(values[k] * v for k, v in weights.items()))


def summarize(truth: np.ndarray, prediction: np.ndarray, sequences: np.ndarray) -> dict:
    """Report micro/macro RTE and explicitly scoped, non-interchangeable MiD."""
    rows = observations(truth, prediction)
    if sequences.shape != truth.shape:
        raise ValueError("sequence array must align")
    seq_rte = [strict_mean(rows["rte_percent"][sequences == s]) for s in np.unique(sequences)]
    result: dict[str, Any] = {
        "n": len(truth),
        "sequences": len(seq_rte),
        "nonfinite_predictions": int((~np.isfinite(prediction)).sum()),
        "micro_rte_percent": strict_mean(rows["rte_percent"]),
        "macro_rte_percent": (
            float(np.mean([v for v in seq_rte if v is not None]))
            if seq_rte and all(v is not None for v in seq_rte)
            else None
        ),
        "garl_mid_strict": strict_mean(rows["garl_mid"]),
        "garl_mid_invalid_n": int((~np.isfinite(rows["garl_mid"])).sum()),
        "garl_reference_failure_percent": (
            float(rows["garl_reference_failed"].mean() * 100) if len(truth) else None
        ),
        "react_logratio_strict": strict_mean(rows["react_logratio"][truth > 0]),
        "react_domain_n": int((truth > 0).sum()),
    }
    for definition, metric, weights in (
        ("garl", "garl_mid", {"c": 0.5, "s": 0.3, "l": 0.1, "n": 0.1}),
        ("react", "react_logratio", {"c": 0.5, "s": 0.3, "l": 0.1}),
    ):
        assigned = bands(truth, definition)
        means = {}
        for label in weights:
            mask = assigned == label
            means[label] = strict_mean(rows[metric][mask])
            result[f"{definition}_{label}_n"] = int(mask.sum())
            result[f"{definition}_{label}_metric"] = means[label]
            result[f"{definition}_{label}_rte_percent"] = strict_mean(rows["rte_percent"][mask])
        result[f"{definition}_outside_bands_n"] = int((assigned == "outside").sum())
        weighted = _weighted(means, weights)
        if definition == "garl":
            result["garl_weighted_mid_strict"] = weighted
        else:
            # REACT does not unambiguously specify whether its 0.9 mass is normalized.
            result["react_weighted_logratio_raw"] = weighted
            result["react_weighted_logratio_normalized"] = (
                weighted / sum(weights.values()) if weighted is not None else None
            )
    return result


def paired_macro(
    truth: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sequences: np.ndarray,
    *,
    draws: int = 10000,
) -> dict:
    """Bootstrap equal-weight sequence RTE deltas, not independent windows."""
    if truth.shape != sequences.shape:
        raise ValueError("sequence array must align")
    if draws < 2:
        raise ValueError("at least two bootstrap draws required")
    a = observations(truth, left)["rte_percent"]
    b = observations(truth, right)["rte_percent"]
    names = np.unique(sequences)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {"status": "INCOMPLETE"}
    if len(names) < 2:
        return {"status": "TOO_FEW_SEQUENCES"}
    delta = np.array([(a[sequences == s] - b[sequences == s]).mean() for s in names])
    sampled = np.random.default_rng(20261010).integers(len(names), size=(draws, len(names)))
    estimates = delta[sampled].mean(1)
    return {
        "status": "COMPLETE",
        "sequences": len(names),
        "draws": draws,
        "delta_macro_rte_pp": float(delta.mean()),
        "ci_low_pp": float(np.quantile(estimates, 0.025)),
        "ci_high_pp": float(np.quantile(estimates, 0.975)),
        "grouping": "sequence; correlated scenario families may remain",
    }
