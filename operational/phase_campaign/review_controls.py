"""Explicit comparator denominators and geometric absence diagnostics."""

from __future__ import annotations

import numpy as np

from operational.phase_campaign.evaluation import metrics


def shared_scores(target: np.ndarray, predictions: dict[str, np.ndarray]) -> dict:
    """Compare methods only on their exact intersection, beside full coverage tables."""
    if not predictions or any(p.shape != target.shape for p in predictions.values()):
        raise ValueError("Shared comparison requires matched nonempty prediction maps")
    mask = np.isfinite(target)
    for pred in predictions.values():
        mask &= np.isfinite(pred)
    return {name: metrics(target[mask], pred[mask]) for name, pred in predictions.items()}


def endpoint_result(inverse_ttc: float) -> tuple[float, str]:
    """Reject undefined/receding collision estimates with an explicit textual cause."""
    if not np.isfinite(inverse_ttc) or inverse_ttc <= 0:
        return float("nan"), "nonpositive_or_nonfinite_endpoint_inverse_ttc"
    return float(1 / inverse_ttc), "ok"
