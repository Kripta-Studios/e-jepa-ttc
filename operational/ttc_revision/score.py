"""Auditable native and common-bound metrics; failures never improve a mean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from operational.efficient_context.common import digest
from operational.train40_system.durable_io import atomic_json


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | str]:
    """Retain complete-cohort semantics and explicitly report coverage failures."""
    if truth.ndim != 1 or truth.shape != prediction.shape:
        raise ValueError("one-dimensional aligned truth and prediction required")
    if not np.isfinite(truth).all() or (truth == 0).any():
        raise ValueError("GT-only eligibility must be applied first")
    finite = np.isfinite(prediction)
    result: dict[str, float | int | str] = {
        "n": len(truth),
        "finite_predictions": int(finite.sum()),
        "coverage": float(finite.mean()) if len(truth) else 0.0,
        "status": "COMPLETE" if len(truth) and finite.all() else "INCOMPLETE",
    }
    if not len(truth) or not finite.all():
        return result
    residual = prediction - truth
    ae = np.abs(residual)
    result.update(
        mae=float(ae.mean()),
        median_ae=float(np.median(ae)),
        rmse=float(np.sqrt((residual**2).mean())),
        rte_percent=float((ae / np.abs(truth)).mean() * 100),
        bias=float(residual.mean()),
        p95_ae=float(np.quantile(ae, 0.95)),
        p99_ae=float(np.quantile(ae, 0.99)),
        sign_error_fraction=float((np.sign(prediction) != np.sign(truth)).mean()),
        overestimate_fraction=float((residual > 0).mean()),
    )
    urgent = (truth > 0) & (truth <= 1)
    result["urgent_n"] = int(urgent.sum())
    if urgent.any():
        result["urgent_miss_fraction"] = float(
            ((prediction[urgent] <= 0) | (prediction[urgent] > 1)).mean()
        )
    if (~urgent).any():
        result["urgent_false_alarm_fraction"] = float(
            ((prediction[~urgent] > 0) & (prediction[~urgent] <= 1)).mean()
        )
    return result


def paired_interval(
    truth: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int = 10000,
) -> dict[str, float | int | str]:
    """Cluster bootstrap of paired MAE differences; resample whole groups."""
    if not (truth.shape == left.shape == right.shape == groups.shape):
        raise ValueError("unaligned paired arrays")
    if not len(truth) or not (np.isfinite(left).all() and np.isfinite(right).all()):
        return {"status": "UNAVAILABLE_INCOMPLETE_COHORT"}
    names = np.unique(groups)
    if len(names) < 2:
        return {"status": "UNAVAILABLE_FEWER_THAN_TWO_GROUPS", "groups": len(names)}
    delta = np.abs(left - truth) - np.abs(right - truth)
    sums = np.array([delta[groups == name].sum() for name in names])
    counts = np.array([(groups == name).sum() for name in names])
    sampled = np.random.default_rng(20261009).integers(len(names), size=(draws, len(names)))
    differences = sums[sampled].sum(1) / counts[sampled].sum(1)
    return {
        "status": "COMPLETE",
        "groups": len(names),
        "draws": draws,
        "difference_mae": float(delta.mean()),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }


def score(path: Path, manifest_path: Path, output: Path, *, methods: list[str]) -> None:
    """Emit every population row, signed buckets, scenario errors and paired CIs."""
    table = pd.read_csv(path)
    if (
        not methods
        or len(methods) != len(set(methods))
        or {"query_id", "sequence_id", "truth_ttc_seconds", "anchor_us"}.intersection(methods)
        or not set(methods).issubset(table.columns)
    ):
        raise ValueError("distinct existing prediction columns required, excluding IDs and GT")
    if table.query_id.duplicated().any():
        raise ValueError("duplicate prediction query IDs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = pd.DataFrame(manifest["rows"])
    if metadata.query_id.duplicated().any() or set(metadata.query_id) != set(table.query_id):
        raise ValueError("prediction and frozen population IDs differ")
    if "scenario_family" not in metadata:
        metadata["scenario_family"] = metadata.sequence_id
    groups = ["scenario_family", *(c for c in ("speed_bucket", "target_type") if c in metadata)]
    table = table.merge(
        metadata[["query_id", *groups]], on="query_id", validate="one_to_one", how="left"
    )
    truth = table.truth_ttc_seconds.to_numpy(float)
    eligible = np.isfinite(truth) & (truth != 0)
    table["gt_eligible"] = eligible
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "ALL_PREDICTIONS.csv", index=False)
    selected = table[eligible]
    truth = truth[eligible]
    results, paired = [], []
    bins = [0, 0.5, 1, 2, 4, 8, np.inf]
    cohorts = {"all": np.ones(len(truth), bool), "negative": truth < 0}
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        cohorts[f"positive_{low}_{high}"] = (truth > 0) & (truth >= low) & (truth < high)
    for seq in selected.sequence_id.unique():
        cohorts[f"sequence:{seq}"] = selected.sequence_id.to_numpy() == seq
    for family in selected.scenario_family.unique():
        cohorts[f"family:{family}"] = selected.scenario_family.to_numpy() == family
    for grouping in groups[1:]:
        for value in pd.unique(np.asarray(selected[grouping])):
            cohorts[f"{grouping}:{value}"] = np.asarray(selected[grouping]) == value
    if "roi_log_event_count" in selected:
        counts = np.asarray(selected["roi_log_event_count"])
        bounds = [0, 100, 1000, 10000, 100000, np.inf]
        for low, high in zip(bounds[:-1], bounds[1:], strict=True):
            cohorts[f"roi_events:{low}_{high}"] = (counts >= np.float32(np.log1p(low))) & (
                counts < np.float32(np.log1p(high))
            )
    for policy in ("native", "common_bound_60"):
        predictions = {}
        for method in methods:
            values = np.asarray(selected[method], dtype=float)
            if policy == "common_bound_60":
                values = np.where(np.isfinite(values), np.clip(values, -60, 60), values)
            predictions[method] = values
            for name, mask in cohorts.items():
                results.append(
                    {
                        "policy": policy,
                        "method": method,
                        "cohort": name,
                        **metrics(truth[mask], values[mask]),
                    }
                )
        for left in methods:
            if left.startswith("public_Garl"):
                continue
            comparators = [m for m in methods if m.startswith("public_Garl")]
            original = left.replace("Direct_", "H8_", 1)
            if left.startswith("Direct_") and original in methods:
                comparators.append(original)
            for right in comparators:
                for grouping in ("sequence_id", "scenario_family"):
                    paired.append(
                        {
                            "policy": policy,
                            "left": left,
                            "right": right,
                            "grouping": grouping,
                            **paired_interval(
                                truth,
                                predictions[left],
                                predictions[right],
                                np.asarray(selected[grouping]),
                            ),
                        }
                    )
    pd.DataFrame(results).to_csv(output / "METRICS.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "PAIRED_BOOTSTRAP.csv", index=False)
    worst = []
    for method in methods:
        diagnosed = pd.DataFrame(selected.copy())
        diagnosed["method"] = method
        diagnosed["absolute_error"] = np.abs(np.asarray(diagnosed[method], float) - truth)
        diagnosed["signed_error"] = np.asarray(diagnosed[method], float) - truth
        finite = np.isfinite(np.asarray(diagnosed["absolute_error"], float))
        worst.append(
            pd.DataFrame(diagnosed[finite]).sort_values("absolute_error", ascending=False).head(20)
        )
    pd.concat(worst, ignore_index=True).to_csv(output / "TOP_ERRORS_FINITE.csv", index=False)
    atomic_json(
        output / "SCORING.json",
        {
            "source_sha256": digest(path),
            "manifest_sha256": digest(manifest_path),
            "scorer_sha256": digest(Path(__file__)),
            "population": len(table),
            "eligible": int(eligible.sum()),
            "methods": methods,
            "common_bound_is_secondary": True,
            "missing_predictions_never_dropped": True,
            "blind_test": False,
            "grouping_limit": (
                "scenario family is a conservative proxy; session independence not certified"
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()
    score(args.predictions, args.manifest, args.output, methods=args.methods)
