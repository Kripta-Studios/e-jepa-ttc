"""Object-level TTC, GarlTTC and risk metrics."""

from __future__ import annotations

import numpy as np

from e_jepa_ttc.evaluation.metrics import regression_metrics


def ttc_selection_components(
    y_true_ttc_s: np.ndarray,
    y_pred_ttc_s: np.ndarray,
    *,
    low_ttc_max_s: float = 2.0,
    high_ttc_min_s: float = 6.0,
) -> dict[str, float | int]:
    """Return the dimensionless validation score frozen by PLAN_v6.

    Empty low/high TTC bins fall back to the overall relative error. This keeps
    every grouped sequence in the macro average without making its score depend
    on another sequence's target distribution.
    """

    target = np.asarray(y_true_ttc_s, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred_ttc_s, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("TTC target and prediction must be non-empty and shape matched.")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise ValueError("Checkpoint selection rejects non-finite TTC values.")
    if np.any(target <= 0.0):
        raise ValueError("Checkpoint selection requires positive TTC ground truth.")
    if low_ttc_max_s <= 0.0 or high_ttc_min_s <= low_ttc_max_s:
        raise ValueError("TTC selection bin limits are invalid.")

    absolute_error = np.abs(prediction - target)
    relative_error = absolute_error / np.maximum(target, 1e-6)
    mean_relative_error = float(np.mean(relative_error))
    normalized_rmse = float(
        np.sqrt(np.mean(np.square(prediction - target)))
        / np.maximum(np.mean(np.abs(target)), 1e-6)
    )
    low_mask = target <= low_ttc_max_s
    high_mask = target >= high_ttc_min_s
    low_relative_error = (
        float(np.mean(relative_error[low_mask]))
        if np.any(low_mask)
        else mean_relative_error
    )
    high_relative_error = (
        float(np.mean(relative_error[high_mask]))
        if np.any(high_mask)
        else mean_relative_error
    )
    selection_score = (
        mean_relative_error
        + 0.25 * normalized_rmse
        + 0.25 * high_relative_error
        + 0.25 * low_relative_error
    )
    return {
        "selection_score": selection_score,
        "mean_relative_error": mean_relative_error,
        "normalized_rmse": normalized_rmse,
        "low_ttc_relative_error": low_relative_error,
        "high_ttc_relative_error": high_relative_error,
        "low_ttc_support": int(np.count_nonzero(low_mask)),
        "high_ttc_support": int(np.count_nonzero(high_mask)),
        "low_ttc_max_s": low_ttc_max_s,
        "high_ttc_min_s": high_ttc_min_s,
    }


def grouped_ttc_selection_components(
    y_true_ttc_s: np.ndarray,
    y_pred_ttc_s: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    """Compute the validation score with equal weight for every sequence."""

    target = np.asarray(y_true_ttc_s, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred_ttc_s, dtype=np.float64).reshape(-1)
    group_ids = np.asarray(groups).astype(str).reshape(-1)
    if target.shape != prediction.shape or target.shape != group_ids.shape:
        raise ValueError("Targets, predictions and sequence groups must be shape matched.")
    unique_groups = np.unique(group_ids)
    if unique_groups.size == 0:
        raise ValueError("Grouped checkpoint selection requires at least one sequence.")

    per_group: dict[str, dict[str, float | int]] = {}
    for group in unique_groups:
        selected = group_ids == group
        per_group[str(group)] = ttc_selection_components(
            target[selected],
            prediction[selected],
        )
    scores = np.asarray(
        [float(values["selection_score"]) for values in per_group.values()],
        dtype=np.float64,
    )
    macro_fields = (
        "mean_relative_error",
        "normalized_rmse",
        "low_ttc_relative_error",
        "high_ttc_relative_error",
    )
    return {
        "sequence_macro_selection_score": float(np.mean(scores)),
        "worst_sequence_selection_score": float(np.max(scores)),
        **{
            f"sequence_macro_{field}": float(
                np.mean([float(values[field]) for values in per_group.values()])
            )
            for field in macro_fields
        },
        "per_sequence_checkpoint_selection": per_group,
    }


def garl_ttc_metrics(
    y_true_ttc_s: np.ndarray,
    y_pred_ttc_s: np.ndarray,
    *,
    delta_t_s: float = 0.1,
) -> dict[str, object]:
    """Reproduce the public GarlTTC MiD/RTE bin definitions and weights."""

    if delta_t_s <= 0:
        msg = "delta_t_s must be positive."
        raise ValueError(msg)
    target = np.asarray(y_true_ttc_s, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred_ttc_s, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape:
        msg = "GarlTTC target and prediction shapes must match."
        raise ValueError(msg)
    bin_specs = (
        ("negative", -10.0, 0.0, 0.1),
        ("crucial", 0.0, 3.0, 0.5),
        ("small", 3.0, 6.0, 0.3),
        ("large", 6.0, 10.0, 0.1),
    )
    bins: dict[str, dict[str, float | int]] = {}
    weighted_mid = 0.0
    weighted_rte = 0.0
    all_weighted_bins_available = True
    for name, lower, upper, weight in bin_specs:
        in_bin = (target > lower) & (target <= upper)
        count = int(np.count_nonzero(in_bin))
        if count == 0:
            bins[name] = {
                "count": 0,
                "mid": float("nan"),
                "rte_pct": float("nan"),
                "failure_ratio": float("nan"),
                "failure_rate_pct": float("nan"),
            }
            all_weighted_bins_available = False
            continue
        truth = target[in_bin]
        estimate = prediction[in_bin]
        truth_ratio = 1.0 - delta_t_s / truth
        estimate_ratio = np.where(
            estimate != 0.0,
            1.0 - delta_t_s / estimate,
            np.nan,
        )
        # Match the public evaluator exactly: non-positive apparent height ratios
        # yield NaN MiD, but only NaN/Inf or |TTC| < 0.1 count as failures.
        failed = ~np.isfinite(estimate) | (np.abs(estimate) < 0.1)
        valid_mid = np.isfinite(estimate_ratio) & (estimate_ratio > 0) & (truth_ratio > 0)
        mid_residual = np.abs(np.log(truth_ratio[valid_mid]) - np.log(estimate_ratio[valid_mid]))
        mid = float(np.mean(mid_residual) * 1e4) if np.any(valid_mid) else float("nan")
        valid_rte = (~failed) & (truth != 0.0)
        relative_error = np.abs(estimate[valid_rte] - truth[valid_rte]) / np.abs(truth[valid_rte])
        rte = float(np.mean(relative_error) * 100.0) if np.any(valid_rte) else float("nan")
        bins[name] = {
            "count": count,
            "mid": mid,
            "rte_pct": rte,
            "failure_ratio": float(np.mean(failed)),
            "failure_rate_pct": float(np.mean(failed) * 100.0),
        }
        if np.isfinite(mid) and np.isfinite(rte):
            weighted_mid += weight * mid
            weighted_rte += weight * rte
        else:
            all_weighted_bins_available = False
    return {
        "delta_t_s": delta_t_s,
        "weighted_mid": weighted_mid if all_weighted_bins_available else float("nan"),
        "weighted_rte_pct": (weighted_rte if all_weighted_bins_available else float("nan")),
        "bins": bins,
    }


def binary_risk_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_positive_support: int = 25,
    min_negative_support: int = 25,
) -> dict[str, object]:
    """Compute threshold-free, thresholded and calibration metrics with strict support limits."""

    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if target.shape != probability.shape:
        msg = "Risk labels and probabilities must have matching shapes."
        raise ValueError(msg)
    valid = np.isfinite(probability) & ((target == 0) | (target == 1))
    target = target[valid]
    probability = np.clip(probability[valid], 0.0, 1.0)

    positive = target == 1
    negative = ~positive
    pos_count = int(np.count_nonzero(positive))
    neg_count = int(np.count_nonzero(negative))

    is_supported = (pos_count >= min_positive_support) and (neg_count >= min_negative_support)
    support_status = "supported" if is_supported else "unsupported"

    base_payload = {
        "positive_count": pos_count,
        "negative_count": neg_count,
        "minimum_positive_support": min_positive_support,
        "minimum_negative_support": min_negative_support,
        "support_status": support_status,
        "calibration_fitted": False,
        "reportable_metrics": [],
    }

    if not is_supported:
        # Fill with NaNs for unsupported metrics
        return {
            **base_payload,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "precision_at_0_5": float("nan"),
            "recall_at_0_5": float("nan"),
            "f1_at_0_5": float("nan"),
            "false_negative_rate_at_0_5": float("nan"),
            "brier": float("nan"),
            "ece_10": float("nan"),
            "nll": float("nan"),
        }

    predicted = probability >= 0.5
    true_positive = int(np.count_nonzero(predicted & positive))
    false_positive = int(np.count_nonzero(predicted & negative))
    false_negative = int(np.count_nonzero((~predicted) & positive))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)

    return {
        **base_payload,
        "calibration_fitted": True,
        "reportable_metrics": [
            "auroc",
            "auprc",
            "precision_at_0_5",
            "recall_at_0_5",
            "f1_at_0_5",
            "false_negative_rate_at_0_5",
            "brier",
            "ece_10",
            "nll",
        ],
        "auroc": _binary_auroc(target, probability),
        "auprc": _average_precision(target, probability),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
        "false_negative_rate_at_0_5": float(false_negative / max(pos_count, 1)),
        "brier": float(np.mean((probability - target) ** 2)),
        "ece_10": _expected_calibration_error(target, probability, bins=10),
        "nll": float(
            -np.mean(
                target * np.log(np.maximum(probability, 1e-12))
                + (1 - target) * np.log(np.maximum(1.0 - probability, 1e-12))
            )
        ),
    }


def object_ttc_metrics(
    y_true_ttc_s: np.ndarray,
    y_pred_ttc_s: np.ndarray,
    risk_probabilities: np.ndarray | None = None,
    *,
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    min_positive_support: int = 25,
    min_negative_support: int = 25,
) -> dict[str, object]:
    """Combine conventional regression, GarlTTC and per-threshold risk metrics."""

    target = np.asarray(y_true_ttc_s, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred_ttc_s, dtype=np.float64).reshape(-1)
    payload: dict[str, object] = {
        "regression": regression_metrics(target, prediction),
        "garl_ttc": garl_ttc_metrics(target, prediction),
    }
    if np.all(target > 0.0):
        payload["checkpoint_selection"] = ttc_selection_components(target, prediction)
    if risk_probabilities is not None:
        probability = np.asarray(risk_probabilities, dtype=np.float64)
        if probability.shape != (target.shape[0], len(risk_thresholds_s)):
            msg = "risk_probabilities has an incompatible shape."
            raise ValueError(msg)

        risk_payload = {}
        for index, threshold in enumerate(risk_thresholds_s):
            threshold_metrics = binary_risk_metrics(
                ((target > 0.0) & (target <= threshold)).astype(np.int64),
                probability[:, index],
                min_positive_support=min_positive_support,
                min_negative_support=min_negative_support,
            )
            # Inject the threshold directly to match protocol
            threshold_metrics["threshold_s"] = threshold
            risk_payload[str(threshold)] = threshold_metrics
        payload["risk"] = risk_payload
    return payload


def _binary_auroc(target: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(np.count_nonzero(target == 1))
    negative_count = int(np.count_nonzero(target == 0))
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    start = 0
    while start < scores.shape[0]:
        stop = start + 1
        while stop < scores.shape[0] and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) * 0.5
        start = stop
    positive_rank_sum = float(np.sum(ranks[target == 1]))
    return (positive_rank_sum - positive_count * (positive_count + 1) * 0.5) / (
        positive_count * negative_count
    )


def _average_precision(target: np.ndarray, scores: np.ndarray) -> float:
    positive_count = int(np.count_nonzero(target == 1))
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_target = target[order]
    cumulative_positive = np.cumsum(sorted_target == 1)
    precision = cumulative_positive / np.arange(1, target.shape[0] + 1)
    return float(np.sum(precision[sorted_target == 1]) / positive_count)


def _expected_calibration_error(
    target: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = target.shape[0]
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if np.any(mask):
            error += (
                np.count_nonzero(mask)
                / total
                * abs(float(np.mean(probabilities[mask])) - float(np.mean(target[mask])))
            )
    return float(error)


__all__ = [
    "binary_risk_metrics",
    "garl_ttc_metrics",
    "grouped_ttc_selection_components",
    "object_ttc_metrics",
    "ttc_selection_components",
]
