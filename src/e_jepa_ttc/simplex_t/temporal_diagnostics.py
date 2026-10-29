"""Analysis-only changes between sampled queries, never an object history source."""

from __future__ import annotations

import numpy as np
import pandas as pd

from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass

# Fixed before any new scientific evaluation; these are not fitted thresholds.
MAX_GAP_US = 750_000  # Registered H16 span; do not bridge longer unobserved intervals.
RAPID_PHASE_PER_SECOND = 1.0


def sampled_temporal_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return query/link diagnostics and strata retaining full OLD metric mass.

    Input must come from the sealed publication reader, including its source-bound
    anchor timestamps. Adjacency depends only on identity/time, never targets.
    Target phase rates and sign transitions define analysis subsets only. A sign
    change is observed at two sampled endpoints, not localized in between them.
    No delay-to-transition or complete object trajectory is inferred.
    """
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "anchor_us"]
    required = [*identity, "arm", "seed", "target_ttc", "prediction_ttc_s", "loss"]
    if (
        not set(required) <= set(frame)
        or len(frame) != 8192
        or frame[required].isna().to_numpy().any()
        or frame.sample_token.duplicated().any()
        or frame.sequence_id.nunique() != 9
        or set(frame.outer_fold) != {0, 1, 2}
        or frame.arm.nunique() != 1
        or frame.seed.nunique() != 1
        or not frame.groupby("sequence_id").outer_fold.nunique().eq(1).all()
    ):
        raise ValueError("complete sealed OLD arm with source-bound timestamps required")
    if (
        not pd.api.types.is_integer_dtype(frame.anchor_us.dtype)
        or (frame.anchor_us < 0).any()
        or not np.isfinite(frame.target_ttc.to_numpy(np.float64)).all()
        or not np.isfinite(frame.loss.to_numpy(np.float64)).all()
        or (frame.loss < 0).any()
        or frame.duplicated(["sequence_id", "track_id", "anchor_us"]).any()
    ):
        raise ValueError("invalid or ambiguous temporal diagnostic inputs")
    rows = (
        frame.loc[:, required]
        .sort_values(["sequence_id", "track_id", "anchor_us"], kind="mergesort")
        .reset_index(drop=True)
    )
    previous = rows.groupby(["sequence_id", "track_id"], sort=False).shift(1)
    gap = rows.anchor_us - previous.anchor_us
    eligible = previous.sample_token.notna() & gap.gt(0) & gap.le(MAX_GAP_US)
    rows["previous_sample_token"] = previous.sample_token
    rows["sample_gap_us"] = gap
    rows["temporal_eligible"] = eligible
    rows["temporal_ineligible_reason"] = np.where(
        previous.sample_token.isna(),
        "NO_PREVIOUS_SAMPLED_QUERY",
        np.where(eligible, "", "GAP_EXCEEDS_FIXED_SPAN"),
    )
    phase = benchmark_phase(rows.target_ttc.to_numpy(np.float64))
    previous_phase = pd.Series(phase).groupby([rows.sequence_id, rows.track_id]).shift(1)
    rate = (phase - previous_phase) / (gap / 1_000_000)
    rows["target_phase_rate_per_second"] = rate.where(eligible)
    rows["rapid_change"] = eligible & rate.abs().ge(RAPID_PHASE_PER_SECOND)
    rows["sign_transition"] = eligible & (np.sign(rows.target_ttc) != np.sign(previous.target_ttc))
    rows["current_sign_correct"] = np.sign(rows.prediction_ttc_s) == np.sign(rows.target_ttc)
    rows["retains_previous_target_sign"] = eligible & (
        np.sign(rows.prediction_ttc_s) == np.sign(previous.target_ttc)
    )
    mass = strict_macro_mass(rows.target_ttc.to_numpy(), rows.sequence_id.to_numpy())
    strata = {
        "all_queries": np.ones(len(rows), dtype=bool),
        "ineligible": ~eligible,
        "eligible": eligible,
        "rapid_change": rows.rapid_change,
        "nonrapid_change": eligible & ~rows.rapid_change,
        "sign_transition": rows.sign_transition,
        "no_sign_transition": eligible & ~rows.sign_transition,
        "rapid_sign_transition": rows.rapid_change & rows.sign_transition,
    }
    summaries = []
    for name, mask in strata.items():
        total = float(mass[mask].sum())
        summary = {
            "arm": rows.arm.iloc[0],
            "seed": int(rows.seed.iloc[0]),
            "stratum": name,
            "queries": int(mask.sum()),
            "global_mass": total,
            "empty": not mask.any(),
            "max_gap_us": MAX_GAP_US,
            "rapid_phase_per_second": RAPID_PHASE_PER_SECOND,
        }
        for field in ("loss", "current_sign_correct", "retains_previous_target_sign"):
            values = rows.loc[mask, field].to_numpy(np.float64)
            contribution = float(mass[mask] @ values)
            summary[f"{field}_global_contribution"] = contribution
            summary[f"{field}_conditional_weighted_mean"] = contribution / total if total else None
            summary[f"{field}_raw_mean"] = float(values.mean()) if len(values) else None
        summaries.append(summary)
    return rows, pd.DataFrame(summaries)
