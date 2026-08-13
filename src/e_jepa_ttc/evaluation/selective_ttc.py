"""Point and abstention metrics for Scientific Recovery V7."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from e_jepa_ttc.evaluation.garl_ttc_protocol import (
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)

V7_COVERAGE_LEVELS: tuple[float, ...] = (1.0, 0.99, 0.975, 0.95, 0.90, 0.80, 0.70, 0.50)


def guard_margin(
    log_height_ratio: Iterable[float] | np.ndarray,
    sensor_support: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return the frozen V7 confidence margin for each prediction."""

    ratio = np.asarray(log_height_ratio, dtype=np.float64)
    support = np.asarray(sensor_support, dtype=np.float64)
    if ratio.shape != support.shape:
        raise ValueError("log_height_ratio and sensor_support must share shape")
    return np.minimum(np.abs(ratio) / 0.002, support / 0.0001)


def selective_predictions(
    point_prediction_ttc_s: Iterable[float] | np.ndarray,
    known_mask: Iterable[bool] | np.ndarray,
) -> np.ndarray:
    """Apply abstention without changing the finite point prediction."""

    point = np.asarray(point_prediction_ttc_s, dtype=np.float64).reshape(-1)
    known = np.asarray(known_mask, dtype=bool).reshape(-1)
    if point.shape != known.shape:
        raise ValueError("point predictions and known_mask must share shape")
    if not np.isfinite(point).all():
        raise ValueError("point_prediction_ttc_s must be finite before abstention")
    return np.where(known, point, np.nan)


def risk_coverage_curve(
    target_ttc_s: Iterable[float] | np.ndarray,
    point_prediction_ttc_s: Iterable[float] | np.ndarray,
    confidence: Iterable[float] | np.ndarray,
    sequence_ids: Iterable[str],
    *,
    coverages: tuple[float, ...] = V7_COVERAGE_LEVELS,
) -> list[dict[str, Any]]:
    """Evaluate sequence-macro MiD after retaining the highest-confidence rows."""

    target = np.asarray(target_ttc_s, dtype=np.float64).reshape(-1)
    prediction = np.asarray(point_prediction_ttc_s, dtype=np.float64).reshape(-1)
    score = np.asarray(confidence, dtype=np.float64).reshape(-1)
    sequences = np.asarray(list(sequence_ids), dtype=str).reshape(-1)
    if not (target.shape == prediction.shape == score.shape == sequences.shape):
        raise ValueError("risk-coverage inputs must share shape")
    if target.size == 0 or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("risk-coverage requires non-empty finite target and point predictions")
    score = np.nan_to_num(score, nan=-np.inf)
    order = np.argsort(-score, kind="stable")
    result: list[dict[str, Any]] = []
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage levels must lie in (0,1]")
        count = max(1, int(np.ceil(coverage * target.size)))
        keep = order[:count]
        signed = signed_garl_metrics(target[keep], prediction[keep])
        macro = sequence_macro_signed_metrics(target[keep], prediction[keep], sequences[keep])
        result.append(
            {
                "requested_coverage": coverage,
                "realized_coverage": count / target.size,
                "rows": count,
                "sequence_count": int(np.unique(sequences[keep]).size),
                "sequence_macro_MiD": float(macro["sequence_macro_paper_MiD_overall"]),
                "sample_weighted_MiD": float(signed["paper_MiD_overall"]),
                "failure_pct": float(signed["failure_rate_pct"]),
            }
        )
    return result


__all__ = [
    "V7_COVERAGE_LEVELS",
    "guard_margin",
    "risk_coverage_curve",
    "selective_predictions",
]
