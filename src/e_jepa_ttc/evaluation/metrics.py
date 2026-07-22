"""Basic regression metrics."""

from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute finite TTC regression metrics."""

    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        msg = f"Shape mismatch: y_true {true.shape}, y_pred {pred.shape}."
        raise ValueError(msg)
    mask = np.isfinite(true) & np.isfinite(pred)
    if not np.any(mask):
        msg = "No finite samples for regression metrics."
        raise ValueError(msg)
    err = pred[mask] - true[mask]
    abs_err = np.abs(err)
    abs_relative_error_pct = abs_err / np.maximum(np.abs(true[mask]), 1e-6) * 100.0
    signed_log_true = np.sign(true[mask]) * np.log1p(np.abs(true[mask]))
    signed_log_pred = np.sign(pred[mask]) * np.log1p(np.abs(pred[mask]))
    signed_log_mae = float(np.mean(np.abs(signed_log_pred - signed_log_true)))
    return {
        "mae_s": float(np.mean(abs_err)),
        "median_abs_error_s": float(np.median(abs_err)),
        "rmse_s": float(np.sqrt(np.mean(err**2))),
        "mean_abs_relative_error_pct": float(np.mean(abs_relative_error_pct)),
        "median_abs_relative_error_pct": float(np.median(abs_relative_error_pct)),
        "log_mae": signed_log_mae,
        "signed_log1p_mae": signed_log_mae,
    }
