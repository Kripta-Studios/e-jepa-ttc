"""Signed Garl-TTC metrics and validation-only checkpoint selection."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

PAPER_MID_WEIGHTS = {"crucial": 0.5, "small": 0.3, "large": 0.1, "negative": 0.1}
BUCKETS = (
    ("crucial", 0.0, 3.0),
    ("small", 3.0, 6.0),
    ("large", 6.0, 10.0),
    ("negative", -10.0, 0.0),
)


def _bucket_mask(values: np.ndarray, lower: float, upper: float, name: str) -> np.ndarray:
    if name == "negative":
        # This is the exact right-closed ``pd.cut`` interval used by the
        # official release: (-10, 0].  In particular, zero is negative in
        # the signed protocol and -10 is outside the protocol domain.
        return (values > lower) & (values <= upper)
    return (values > lower) & (values <= upper)


def signed_garl_metrics(
    y_true_ttc_s: Iterable[float] | np.ndarray,
    y_pred_ttc_s: Iterable[float] | np.ndarray,
    *,
    delta_t_s: float = 0.1,
) -> dict[str, Any]:
    """Compute MiD/RTE/FR with the official signed TTC domains."""

    if delta_t_s <= 0.0:
        raise ValueError("delta_t_s must be positive.")
    target = np.asarray(y_true_ttc_s, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred_ttc_s, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Signed Garl TTC arrays must be non-empty and shape matched.")
    if not np.isfinite(target).all():
        raise ValueError("Signed Garl TTC targets must be finite.")
    bins: dict[str, dict[str, float | int]] = {}
    weighted_mid = 0.0
    weighted_rte = 0.0
    for name, lower, upper in BUCKETS:
        selected = _bucket_mask(target, lower, upper, name)
        count = int(np.count_nonzero(selected))
        if count == 0:
            bins[name] = {
                "count": 0,
                "mid": float("nan"),
                "rte_pct": float("nan"),
                "failure_count": 0,
                "failure_rate_pct": float("nan"),
            }
            continue
        truth = target[selected]
        estimate = prediction[selected]
        with np.errstate(divide="ignore", invalid="ignore"):
            truth_eta = 1.0 - delta_t_s / truth
            estimate_eta = np.where(estimate != 0.0, 1.0 - delta_t_s / estimate, np.nan)
        failed = ~np.isfinite(estimate) | (np.abs(estimate) < 0.1)
        # The release computes MiD for every row in the bucket, then Pandas'
        # default mean skips NaN values.  RTE alone explicitly filters failed
        # predictions.  Preserve both details: an invalid bucket with no
        # usable MiD remains NaN, while one failed row need not poison its
        # bucket if another row is valid.
        with np.errstate(divide="ignore", invalid="ignore"):
            mid_per_sample = np.abs(np.log(truth_eta) - np.log(estimate_eta)) * 1e4
            rte_per_sample = np.abs(estimate - truth) / np.abs(truth) * 100.0
        non_nan_mid = mid_per_sample[~np.isnan(mid_per_sample)]
        mid = float(np.mean(non_nan_mid)) if non_nan_mid.size else float("nan")
        valid_rte = ~failed
        rte = float(np.mean(rte_per_sample[valid_rte])) if np.any(valid_rte) else float("nan")
        bins[name] = {
            "count": count,
            "mid": mid,
            "rte_pct": rte,
            "failure_count": int(np.count_nonzero(failed)),
            "failure_rate_pct": float(np.mean(failed) * 100.0),
        }
        weighted_mid += PAPER_MID_WEIGHTS[name] * mid
        weighted_rte += PAPER_MID_WEIGHTS[name] * rte
    return {
        "protocol": "garl_signed_v1",
        "delta_t_s": delta_t_s,
        "num_samples": int(target.size),
        "bins": bins,
        # Do not suppress NaN here.  The official weighted sum propagates a
        # missing bucket or an invalid MiD, which makes the defect visible to
        # checkpoint selection and to readiness gates.
        "paper_MiD_overall": weighted_mid,
        "weighted_RTE_pct": weighted_rte,
        "failure_count": int(
            np.count_nonzero(~np.isfinite(prediction) | (np.abs(prediction) < 0.1))
        ),
        "failure_rate_pct": float(
            np.mean(~np.isfinite(prediction) | (np.abs(prediction) < 0.1)) * 100.0
        ),
    }


def sequence_macro_signed_metrics(
    y_true_ttc_s: np.ndarray,
    y_pred_ttc_s: np.ndarray,
    sequence_ids: Iterable[str],
    *,
    delta_t_s: float = 0.1,
) -> dict[str, Any]:
    """Aggregate the signed protocol by complete sequences, not windows."""

    target = np.asarray(y_true_ttc_s).reshape(-1)
    prediction = np.asarray(y_pred_ttc_s).reshape(-1)
    groups = np.asarray(list(sequence_ids)).astype(str).reshape(-1)
    if target.shape != prediction.shape or target.shape != groups.shape:
        raise ValueError("Sequence-macro inputs must have matching shapes.")
    per_sequence = {
        sequence: signed_garl_metrics(
            target[groups == sequence], prediction[groups == sequence], delta_t_s=delta_t_s
        )
        for sequence in sorted(np.unique(groups).tolist())
    }
    values = [float(item["paper_MiD_overall"]) for item in per_sequence.values()]
    finite = [value for value in values if np.isfinite(value)]
    return {
        "protocol": "garl_signed_v1",
        "sequence_macro_paper_MiD_overall": float(np.mean(finite)) if finite else float("nan"),
        "per_sequence": per_sequence,
    }


def select_checkpoint(
    metrics: list[dict[str, Any]],
    *,
    protocol: str = "garl_signed_v1",
) -> dict[str, Any]:
    """Select a checkpoint using validation signed metrics only."""

    if protocol != "garl_signed_v1":
        raise ValueError(f"Unsupported checkpoint-selection protocol: {protocol!r}.")
    if not metrics:
        raise ValueError("Checkpoint selection requires validation metrics.")
    usable = [item for item in metrics if item.get("protocol") == protocol]
    if len(usable) != len(metrics):
        raise ValueError("Checkpoint metrics contain a different protocol.")
    candidates = [
        (index, item)
        for index, item in enumerate(usable)
        if np.isfinite(float(item.get("paper_MiD_overall", np.nan)))
        and np.isfinite(float(item.get("failure_rate_pct", np.nan)))
    ]
    if not candidates:
        raise ValueError("Checkpoint selection has no finite validation metrics.")
    ranked = sorted(
        candidates,
        key=lambda pair: (
            float(pair[1]["paper_MiD_overall"]),
            float(pair[1]["failure_rate_pct"]),
        ),
    )
    index, selected = ranked[0]
    return {
        "protocol": protocol,
        "selected_index": index,
        "selection_rule": "validation_paper_MiD_overall_then_failure_rate",
        "excluded_non_finite_indices": sorted(set(range(len(usable))) - {i for i, _ in candidates}),
        "selected": selected,
    }


__all__ = [
    "BUCKETS",
    "PAPER_MID_WEIGHTS",
    "select_checkpoint",
    "sequence_macro_signed_metrics",
    "signed_garl_metrics",
]
