"""Independent NumPy recalc of durable prediction tables, without production scorer calls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from operational.range_transfer.common import read, sha, utc, write

from .common import RUN


def recalculate(frame: pd.DataFrame) -> dict:
    """Use the pinned signed domain and explicit per-group arithmetic."""
    truth = frame.target_ttc.to_numpy(dtype=np.float64)
    pred = frame.prediction_ttc.to_numpy(dtype=np.float64)
    groups = frame.group_id.to_numpy()
    finite = np.isfinite(pred)
    failed = ~finite | (np.abs(pred) < 0.1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mid = 10000 * np.abs(np.log1p(-0.1 / truth) - np.log1p(-0.1 / pred))
    valid_phase = finite & ((pred < 0) | (pred > 0.1)) & np.isfinite(mid)
    result = {
        "num_rows": len(frame),
        "num_failed_predictions": int(failed.sum()),
        "coverage": float(finite.mean()),
        "complete_prediction_and_mid_coverage": bool(valid_phase.all()),
        "mean_MiD": float(mid.mean()) if valid_phase.all() else None,
    }
    error = np.abs(pred[finite] - truth[finite])
    result["diagnostics.mae_s"] = float(error.mean()) if len(error) else None
    result["diagnostics.median_ae_s"] = float(np.median(error)) if len(error) else None
    result["diagnostics.rmse_s"] = float(np.sqrt((error**2).mean())) if len(error) else None
    result["diagnostics.sign_error_rate"] = (
        float((np.signbit(pred[finite]) != np.signbit(truth[finite])).mean())
        if finite.any()
        else None
    )
    bands = [
        ("crucial", 0.0, 3.0, 0.5),
        ("small", 3.0, 6.0, 0.3),
        ("large", 6.0, 10.0, 0.1),
        ("negative", -10.0, 0.0, 0.1),
    ]
    domain = np.zeros(len(frame), dtype=bool)
    overall = 0.0
    all_bands = True
    for name, low, high, weight in bands:
        mask = (truth > low) & (truth <= high)
        domain |= mask
        result[f"bins.{name}.count"] = int(mask.sum())
        result[f"bins.{name}.failure_count"] = int(failed[mask].sum())
        result[f"bins.{name}.phase_invalid_count"] = int((~np.isfinite(mid[mask])).sum())
        values = mid[mask]
        result[f"bins.{name}.pooled_diagnostic_mid"] = (
            float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None
        )
        allowed = mask.any() and not failed[mask].any() and np.isfinite(mid[mask]).all()
        if allowed:
            group_values = [
                mid[mask & (groups == group)].mean() for group in np.unique(groups[mask])
            ]
            group_mean = float(np.mean(group_values))
            overall += weight * group_mean
        else:
            group_mean = None
            all_bands = False
        result[f"bins.{name}.group_macro_mid"] = group_mean
    result["num_targets_outside_protocol"] = int((~domain).sum())
    result["overall_MiD"] = overall if all_bands and valid_phase.all() else None
    return result


def verify(folder: Path) -> dict:
    """Require complete token coverage and compare explicit scalar fields."""
    receipt = read(folder / "COMPLETE.json")
    path = folder / "predictions.parquet"
    if receipt["status"] != "COMPLETE" or sha(path) != receipt["sha256"]:
        raise ValueError("Prediction bytes differ from the complete receipt")
    frame = pd.read_parquet(path)
    if frame.sample_token.duplicated().any() or len(frame) != receipt["rows"]:
        raise ValueError("Missing or duplicated query identities")
    metrics = read(folder / "METRICS.json")
    expected = recalculate(frame)
    comparisons = []
    for name, value in expected.items():
        original: object = metrics
        for component in name.split("."):
            if not isinstance(original, dict):
                raise ValueError(f"Invalid metric field structure: {name}")
            original = original[component]
        if original is not None and not isinstance(original, (bool, int, float)):
            raise ValueError(f"Expected a scalar metric field: {name}")
        if value is None or original is None:
            matches = value is None and original is None
            difference = None
        elif isinstance(value, (bool, int)):
            matches = value == original
            difference = abs(float(value) - float(original))
        else:
            difference = abs(float(value) - float(original))
            matches = bool(np.isclose(value, original, atol=1e-9, rtol=1e-10))
        comparisons.append({"field": name, "pass": matches, "absolute_difference": difference})
    result = {
        "status": "PASS" if all(r["pass"] for r in comparisons) else "FAIL",
        "rows": len(frame),
        "prediction_sha256": receipt["sha256"],
        "metrics_sha256": sha(folder / "METRICS.json"),
        "comparisons": comparisons,
        "scope": (
            "Listed aggregate/band/coverage scalar fields; no producer replay or raw event audit"
        ),
        "utc": utc(),
    }
    destination = RUN / "verification/predictions" / folder.parent.name / folder.name
    write(destination / "NUMPY_RECALC.json", result)
    if result["status"] != "PASS":
        raise AssertionError(f"Independent metric discrepancy: {destination}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(verify(parser.parse_args().folder))
