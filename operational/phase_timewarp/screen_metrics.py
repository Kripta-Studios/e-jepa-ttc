"""Coverage-aware paired screen summaries; no fitting or model selection."""

import numpy as np


def compare_screen(
    target: np.ndarray, original: np.ndarray, canonical: np.ndarray, groups: np.ndarray
) -> dict:
    """Preserve full denominators and macro-average recordings on the paired cohort."""
    if any(a.shape != target.shape for a in (original, canonical, groups)) or target.ndim != 1:
        raise ValueError("Expected aligned one-dimensional vectors")
    supported = np.isfinite(target)
    old, new = np.isfinite(original), np.isfinite(canonical)
    paired = supported & old & new
    by_recording = {}
    for group in sorted(set(groups)):
        members = groups == group
        mask = members & paired
        row: dict = {
            "population": int(members.sum()),
            "paired": int(mask.sum()),
            "original_available": int((members & old).sum()),
            "canonical_available": int((members & new).sum()),
        }
        for name, values in (("original", original), ("canonical", canonical)):
            error = np.abs(values[mask].astype(np.float64) - target[mask])
            row[name] = {
                "mae_seconds": float(error.mean()) if len(error) else None,
                "relative_error_pct": float(
                    (error / np.maximum(np.abs(target[mask]), 1e-6)).mean() * 100
                )
                if len(error)
                else None,
            }
        by_recording[str(group)] = row
    summary = {
        "population": len(target),
        "targets": int(supported.sum()),
        "paired": int(paired.sum()),
        "original_available": int(old.sum()),
        "canonical_available": int(new.sum()),
        "newly_missing": int((old & ~new).sum()),
        "newly_available": int((~old & new).sum()),
        "sign_changes": int((old & new & (np.sign(original) != np.sign(canonical))).sum()),
        "by_recording": by_recording,
    }
    complete_groups = all(row["paired"] > 0 for row in by_recording.values())
    for name in ("original", "canonical"):
        summary[name] = {
            key: float(np.mean([row[name][key] for row in by_recording.values()]))
            if complete_groups
            else None
            for key in ("mae_seconds", "relative_error_pct")
        }
    summary["nonworsening"] = bool(
        supported.all()
        and complete_groups
        and summary["newly_missing"] == 0
        and all(
            summary["canonical"][k] <= summary["original"][k]
            for k in ("mae_seconds", "relative_error_pct")
        )
    )
    summary["any_strict_improvement"] = bool(
        complete_groups
        and any(
            summary["canonical"][k] < summary["original"][k]
            for k in ("mae_seconds", "relative_error_pct")
        )
    )
    return summary
