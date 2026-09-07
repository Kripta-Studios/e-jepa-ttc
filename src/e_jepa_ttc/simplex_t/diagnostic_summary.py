"""Full-cohort and availability-stratified diagnostics without changing OLD mass."""

from __future__ import annotations

import numpy as np
import pandas as pd

from e_jepa_ttc.evaluation.stage63_65 import BUCKETS, strict_macro_mass


def summarize_diagnostics(frame: pd.DataFrame, *, history_length: int) -> pd.DataFrame:
    """Aggregate verified publication fields; never infer an object trajectory.

    The caller must validate the publication, frozen sources and independent OLD
    targets first. Exact observed history spans and ROI ages define strata, not
    thresholds selected using errors. Every stratum retains both raw averages and
    its contribution under the original full-cohort evaluation mass. Conditional
    means are labelled separately; no subset substitutes for the primary metric.
    """
    numeric = ["loss", "signed_phase_error", "escape_gain", "escape_harm", "phase_interval_width"]
    boolean = [
        "wrong_sign",
        "phase_interval_covers",
        "phase_support_saturation",
        "target_outside_support",
        "finite_ttc_cap",
    ]
    required = {
        "sample_token",
        "sequence_id",
        "outer_fold",
        "target_ttc",
        "arm",
        "seed",
        "history_count",
        "history_span_us",
        "roi_age_us",
        "cold_start",
        "hull_position",
        *numeric,
        *boolean,
    }
    if (
        history_length not in {1, 4, 8, 16}
        or not required <= set(frame)
        or len(frame) != 8192
        or frame.sample_token.duplicated().any()
        or frame.sequence_id.nunique() != 9
        or set(frame.outer_fold) != {0, 1, 2}
        or frame.arm.nunique() != 1
        or frame.seed.nunique() != 1
        or frame.loc[:, sorted(required)].isna().to_numpy().any()
    ):
        raise ValueError("one complete verified OLD arm/seed and registered history required")
    for field in [*numeric, "history_count", "history_span_us", "roi_age_us"]:
        if not np.isfinite(frame[field].to_numpy(np.float64)).all():
            raise ValueError("nonfinite diagnostic field")
    counts = frame.history_count.to_numpy()
    spans = frame.history_span_us.to_numpy()
    if (
        not np.equal(counts, np.floor(counts)).all()
        or np.any((counts < 1) | (counts > history_length))
        or np.any(spans < 0)
        or not np.equal(frame.cold_start.to_numpy(), counts == 1).all()
        or np.any(spans[counts == 1] != 0)
        or not set(frame.hull_position) <= {"below", "inside", "above"}
        or any(
            (frame[field] < 0).any()
            for field in ("loss", "escape_gain", "escape_harm", "phase_interval_width")
        )
    ):
        raise ValueError("inconsistent availability or hull diagnostics")
    for field in boolean:
        if not pd.api.types.is_bool_dtype(frame[field].dtype):
            raise ValueError("explicit Boolean diagnostic flags required")
    mass = strict_macro_mass(frame.target_ttc.to_numpy(), frame.sequence_id.to_numpy())
    groups = [
        ("all", "all", np.ones(len(frame), dtype=bool)),
        ("availability", "cold_start", counts == 1),
        ("availability", "full_history", counts == history_length),
        ("availability", "partial_history", (counts > 1) & (counts < history_length)),
    ]
    for field in ("sequence_id", "hull_position", "history_count", "history_span_us", "roi_age_us"):
        for value in sorted(frame[field].unique()):
            groups.append((field, str(value), frame[field].to_numpy() == value))
    target = frame.target_ttc.to_numpy()
    for name, lower, upper, _ in BUCKETS:
        groups.append(("ttc_bucket", name, (target > lower) & (target <= upper)))
    records = []
    for axis, name, mask in groups:
        total = float(mass[mask].sum())
        row = {
            "arm": frame.arm.iloc[0],
            "seed": int(frame.seed.iloc[0]),
            "axis": axis,
            "stratum": name,
            "queries": int(mask.sum()),
            "global_mass": total,
            "empty": not mask.any(),
        }
        for field in [*numeric, *boolean]:
            values = frame[field].to_numpy(np.float64)[mask]
            contribution = float(mass[mask] @ values)
            row[f"{field}_global_contribution"] = contribution
            row[f"{field}_conditional_weighted_mean"] = contribution / total if total else None
            row[f"{field}_raw_mean"] = float(values.mean()) if len(values) else None
        records.append(row)
    return pd.DataFrame(records)
