"""Describe frozen external-stream errors without selecting policies or fitting models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from operational.efficient_context.common import digest
from operational.sota_evidence.metrics import observations
from operational.train40_system.durable_io import atomic_json


def run(root: Path, prior: Path) -> None:
    """Join complete predictions and preserve invalid errors in the accounting."""
    result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
    if result["status"] not in ("COMPLETE", "BASELINE_PARITY_FAILED"):
        raise ValueError("complete external predictions required before error analysis")
    if digest(prior) != result["target_table_sha256"]:
        raise ValueError("target table differs from the scored population")
    predictions = pd.read_csv(root / "PREDICTIONS.csv")
    targets = pd.read_csv(prior)
    if set(targets.query_id) != set(predictions.query_id):
        raise ValueError("population mismatch")
    merged = predictions.merge(
        targets[["query_id", "truth_ttc_seconds", "public_Garl_event_lhr"]],
        on="query_id",
        validate="many_to_one",
    )
    garl = merged.loc[merged.variant == "H8_reference"].copy()
    garl["variant"] = "Garl_event"
    garl["ttc"] = garl.public_Garl_event_lhr
    for column in ("reused", "computed", "approximate_voxel_hits", "min_source_roi_iou"):
        garl[column] = np.nan
    merged = pd.concat([merged, garl], ignore_index=True)
    eligible = np.isfinite(merged.truth_ttc_seconds) & (merged.truth_ttc_seconds != 0)
    excluded = merged.loc[~eligible, ["query_id", "variant", "truth_ttc_seconds"]]
    excluded.to_csv(root / "INELIGIBLE_TARGETS.csv", index=False)
    frame = merged.loc[eligible].copy()
    truth = np.asarray(frame.truth_ttc_seconds, float)
    prediction = np.asarray(frame.ttc, float)
    frame["MiD"] = observations(truth, prediction)["garl_mid"]
    frame["signed_error_s"] = prediction - truth
    frame["absolute_error_s"] = np.abs(prediction - truth)
    frame["band"] = np.select(
        [
            (truth > 0) & (truth <= 3),
            (truth > 3) & (truth <= 6),
            (truth > 6) & (truth <= 10),
            (truth > -10) & (truth < 0),
        ],
        ["c", "s", "l", "n"],
        default="outside",
    )
    rows = []
    for (variant, band), group in frame.groupby(["variant", "band"]):
        errors = np.asarray(group.signed_error_s, float)
        mid = np.asarray(group.MiD, float)
        finite = np.isfinite(mid)
        rows.append(
            {
                "variant": variant,
                "band": band,
                "n": len(group),
                "MAE_s": float(np.mean(np.abs(errors))),
                "signed_bias_s": float(errors.mean()),
                "underestimate_fraction": float(np.mean(errors < 0)),
                "overestimate_fraction": float(np.mean(errors > 0)),
                "absolute_error_p95_s": float(np.quantile(np.abs(errors), 0.95)),
                "invalid_MiD_n": int((~finite).sum()),
                "strict_mean_MiD": float(mid.mean()) if finite.all() else None,
                "finite_MiD_p95": float(np.quantile(mid[finite], 0.95)) if finite.any() else None,
            }
        )
    pd.DataFrame(rows).to_csv(root / "ERROR_BANDS.csv", index=False)
    columns = [
        "query_id",
        "sequence_id",
        "variant",
        "truth_ttc_seconds",
        "ttc",
        "MiD",
        "signed_error_s",
        "absolute_error_s",
        "band",
        "reused",
        "computed",
        "approximate_voxel_hits",
        "min_source_roi_iou",
    ]
    frame[columns].to_csv(root / "SCORED_PREDICTIONS.csv", index=False)
    frame.sort_values("MiD", ascending=False, na_position="first").groupby(
        "variant", sort=False
    ).head(20)[columns].to_csv(root / "WORST_MID.csv", index=False)
    reuse = predictions.groupby("variant").agg(
        n=("query_id", "count"),
        reused_mean=("reused", "mean"),
        computed_mean=("computed", "mean"),
        approximate_voxel_hits_mean=("approximate_voxel_hits", "mean"),
        min_source_roi_iou=("min_source_roi_iou", "min"),
        max_time_approximation_us=("max_time_approximation_us", "max"),
    )
    reuse.to_csv(root / "REUSE.csv")
    atomic_json(
        root / "DIAGNOSTICS.json",
        {
            "status": "COMPLETE_DESCRIPTIVE_ANALYSIS",
            "source_sha256": digest(Path(__file__)),
            "result_sha256": digest(root / "RESULT.json"),
            "predictions_sha256": digest(root / "PREDICTIONS.csv"),
            "targets_sha256": digest(prior),
            "policy_selection": False,
            "optimizer_updates": 0,
            "finite_percentiles_exclude_invalid_MiD_explicitly_counted": True,
            "outputs": {
                name: digest(root / name)
                for name in (
                    "ERROR_BANDS.csv",
                    "SCORED_PREDICTIONS.csv",
                    "WORST_MID.csv",
                    "REUSE.csv",
                    "INELIGIBLE_TARGETS.csv",
                )
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.prior)
