"""Paired acquisition-level comparisons retaining all coverage and tail denominators."""

from __future__ import annotations

import numpy as np

from .physics import Array


def compare(
    target: Array,
    baseline: Array,
    candidate: Array,
    groups: np.ndarray,
    *,
    seed: int = 7,
    resamples: int = 2000,
) -> dict:
    """Describe exposed development; never infer native Overall from positive-only cohorts."""
    if (
        target.ndim != 1
        or len(target) == 0
        or any(a.shape != target.shape for a in (baseline, candidate, groups))
    ):
        raise ValueError("Expected nonempty aligned query arrays")
    if not np.isfinite(target).all() or (target == 0).any() or resamples < 100:
        raise ValueError("Finite nonzero targets and at least 100 bootstrap replicates required")
    old, new = np.isfinite(baseline), np.isfinite(candidate)
    paired = old & new
    recordings = {}
    for group in sorted(set(groups.tolist())):
        members = groups == group
        mask = members & paired
        row: dict = {"required": int(members.sum()), "paired": int(mask.sum())}
        for name, values in (("baseline", baseline), ("candidate", candidate)):
            error = np.abs(values[mask] - target[mask])
            row[name] = {
                "mae": float(error.mean()) if len(error) else None,
                "relative": float((error / np.abs(target[mask])).mean()) if len(error) else None,
            }
        recordings[str(group)] = row
    report: dict = {
        "required": len(target),
        "paired": int(paired.sum()),
        "baseline_available": int(old.sum()),
        "candidate_available": int(new.sum()),
        "newly_missing": int((old & ~new).sum()),
        "sign_changes": int((paired & (np.sign(baseline) != np.sign(candidate))).sum()),
        "by_acquisition": recordings,
        "native_overall": None,
        "scope": "paired exposed development; aliases must be collapsed before calling",
    }
    for name, values in (("baseline", baseline), ("candidate", candidate)):
        error = np.abs(values[paired] - target[paired])
        report[name] = {
            "micro_mae": float(error.mean()) if len(error) else None,
            "median": float(np.median(error)) if len(error) else None,
            "p95": float(np.quantile(error, 0.95)) if len(error) else None,
            "maximum": float(error.max()) if len(error) else None,
            "tail_counts": {str(t): int((error > t).sum()) for t in (1, 3, 5, 10)},
            "sign_errors": int((np.sign(values[paired]) != np.sign(target[paired])).sum()),
            "urgency": {
                str(t): {
                    "positive_targets": int(((target > 0) & (target <= t)).sum()),
                    "false_negative": int(
                        (
                            paired & (target > 0) & (target <= t) & ~((values > 0) & (values <= t))
                        ).sum()
                    ),
                    "false_positive": int(
                        (
                            paired & ~((target > 0) & (target <= t)) & (values > 0) & (values <= t)
                        ).sum()
                    ),
                }
                for t in (0.5, 1.0, 2.0, 4.0)
            },
        }
        for metric in ("mae", "relative"):
            data = [row[name][metric] for row in recordings.values()]
            report[name]["macro_" + metric] = (
                float(np.mean(data)) if all(v is not None for v in data) else None
            )
    differences = [
        row["candidate"]["mae"] - row["baseline"]["mae"]
        for row in recordings.values()
        if row["paired"]
    ]
    if len(differences) == len(recordings) and len(differences) >= 2:
        delta = np.asarray(differences)
        indices = np.random.default_rng(seed).integers(0, len(delta), (resamples, len(delta)))
        report["paired_macro_delta_ci95"] = np.quantile(
            delta[indices].mean(1), [0.025, 0.975]
        ).tolist()
    else:
        report["paired_macro_delta_ci95"] = None
    report["bootstrap"] = {"unit": "acquisition", "seed": seed, "resamples": resamples}
    return report
