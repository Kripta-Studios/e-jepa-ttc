#!/usr/bin/env python
"""Diagnose the clean V6.1/A5 OOF TTC gap to Garl without opening new data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    BUCKETS,
    PAPER_MID_WEIGHTS,
    signed_garl_metrics,
)
from scripts.analyze_v5_a8_oof_failure_modes import (  # noqa: E402
    FAMILY_FEATURES,
    feature_association,
    raw_mid_per_sample,
)

ARMS = ("a5_causal", "v6_1", "a8_0", "a6", "garl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError(f"invalid signed artifact: {path}")
    return payload


def _read_prediction(path: Path, arm: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_token", "sequence_id", "track_id", "target_ttc_s", "prediction_ttc_s"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{arm} predictions lack required columns")
    if frame["sample_token"].astype(str).duplicated().any():
        raise ValueError(f"{arm} predictions contain duplicate tokens")
    return frame[list(required)].rename(columns={"prediction_ttc_s": f"{arm}_prediction"})


def align_inputs(results_dir: Path, d0_rows_path: Path) -> pd.DataFrame:
    """Align all clean OOF predictions to the signed D0 row population."""

    d0 = pd.read_csv(d0_rows_path)
    if d0["sample_token"].astype(str).duplicated().any():
        raise ValueError("D0 rows contain duplicate tokens")
    base = d0.copy()
    for arm in ("a5_causal", "v6_1"):
        current = _read_prediction(results_dir / f"{arm}_outer_dev_predictions.csv", arm)
        base = base.merge(
            current,
            on=["sample_token", "sequence_id", "track_id", "target_ttc_s"],
            how="left",
            validate="one_to_one",
        )
    for arm in ("a8_0", "a6", "garl"):
        expected = f"{arm}_prediction"
        if expected not in base.columns:
            raise ValueError(f"D0 rows lack {expected}")
    if base[["a5_causal_prediction", "v6_1_prediction"]].isna().all(axis=0).any():
        raise ValueError("a clean V6 arm failed to align")
    return base


def _failure(prediction: np.ndarray) -> np.ndarray:
    return ~np.isfinite(prediction) | (np.abs(prediction) < 0.1)


def _arm_columns(frame: pd.DataFrame) -> pd.DataFrame:
    target = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    result = frame.copy()
    for arm in ARMS:
        prediction = result[f"{arm}_prediction"].to_numpy(dtype=np.float64)
        result[f"{arm}_raw_mid"] = raw_mid_per_sample(target, prediction)
        result[f"{arm}_failure"] = _failure(prediction)
    for arm in ARMS:
        if arm != "garl":
            result[f"{arm}_minus_garl_raw_mid"] = result[f"{arm}_raw_mid"] - result["garl_raw_mid"]
    return result


def _official_bucket_decomposition(frame: pd.DataFrame, arm: str) -> dict[str, Any]:
    by_sequence = []
    for _, group in frame.groupby("sequence_id", sort=True):
        by_sequence.append(
            signed_garl_metrics(
                group["target_ttc_s"].to_numpy(dtype=np.float64),
                group[f"{arm}_prediction"].to_numpy(dtype=np.float64),
            )["bins"]
        )
    buckets: dict[str, Any] = {}
    for name, _, _ in BUCKETS:
        mids = np.asarray([float(item[name]["mid"]) for item in by_sequence])
        failures = np.asarray([float(item[name]["failure_rate_pct"]) for item in by_sequence])
        macro_mid = float(np.mean(mids))
        buckets[name] = {
            "sequence_macro_bucket_MiD": macro_mid,
            "weighted_contribution_to_sequence_macro_MiD": PAPER_MID_WEIGHTS[name] * macro_mid,
            "sequence_macro_failure_pct": float(np.mean(failures)),
            "paper_weight": PAPER_MID_WEIGHTS[name],
        }
    return buckets


def _stratum(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value, group in frame.groupby(column, observed=True, sort=True):
        item: dict[str, Any] = {"rows": len(group)}
        for arm in ARMS:
            raw = group[f"{arm}_raw_mid"].to_numpy(dtype=np.float64)
            item[arm] = {
                "mean_raw_MiD_valid": float(np.nanmean(raw)),
                "failure_pct": float(group[f"{arm}_failure"].mean() * 100.0),
            }
        result[str(value)] = item
    return result


def analyze(
    *,
    aggregate_path: Path,
    d0_path: Path,
    d0_rows_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build a signed, development-only explanation of the remaining gap."""

    aggregate = _signed(aggregate_path)
    d0 = _signed(d0_path)
    if aggregate.get("status") != "completed_development_gate_evaluation":
        raise ValueError("V6 aggregate is incomplete")
    if d0.get("rows", {}).get("sha256") != _sha256(d0_rows_path):
        raise ValueError("D0 diagnostic rows differ from their signed artifact")
    contracts = aggregate.get("contracts", {})
    if contracts.get("public_validation_used_for_selection") is not False:
        raise ValueError("aggregate used public validation")
    if contracts.get("private_test_opened") is not False:
        raise ValueError("aggregate opened private/test")

    frame = _arm_columns(align_inputs(aggregate_path.parent, d0_rows_path))
    if len(frame) != 8192:
        raise ValueError("gap analysis requires the exact 8192-row OOF population")
    bucket_decomposition = {arm: _official_bucket_decomposition(frame, arm) for arm in ARMS}
    gap_contributions: dict[str, Any] = {}
    for arm in ("a5_causal", "v6_1", "a8_0", "a6"):
        values = {
            name: bucket_decomposition[arm][name]["weighted_contribution_to_sequence_macro_MiD"]
            - bucket_decomposition["garl"][name]["weighted_contribution_to_sequence_macro_MiD"]
            for name, _, _ in BUCKETS
        }
        gap_contributions[arm] = {
            "by_target_bucket": values,
            "largest_absolute_contributor": max(values, key=lambda key: abs(values[key])),
            "sum": float(sum(values.values())),
        }

    feature_names = sorted({feature for family in FAMILY_FEATURES.values() for feature in family})
    associations: dict[str, Any] = {}
    for arm in ("a5_causal", "v6_1"):
        outcome = f"{arm}_minus_garl_raw_mid"
        current = {
            feature: feature_association(frame, feature, outcome)
            for feature in feature_names
            if feature in frame.columns
        }
        ranked = sorted(
            current,
            key=lambda feature: (
                -float(current[feature]["median_absolute_sequence_spearman"])
                if np.isfinite(current[feature]["median_absolute_sequence_spearman"])
                else float("inf"),
                feature,
            ),
        )
        associations[arm] = {"features": current, "ranked_features": ranked}

    for feature in (
        "target_log_ratio_abs",
        "a8_transport_foreground_flow_magnitude",
        "event_density_current",
        "bbox_area_fraction_current",
        "a8_transport_entropy",
    ):
        frame[f"{feature}_quartile"] = pd.qcut(
            frame[feature], 4, labels=("Q1", "Q2", "Q3", "Q4"), duplicates="drop"
        )
    strata = {
        "sequence": _stratum(frame, "sequence_id"),
        "abs_ttc_bucket": _stratum(frame, "abs_ttc_bucket"),
        "regime": _stratum(frame, "regime"),
        "fold": _stratum(frame, "fold"),
        "motion_quartile": _stratum(frame, "target_log_ratio_abs_quartile"),
        "transport_magnitude_quartile": _stratum(
            frame, "a8_transport_foreground_flow_magnitude_quartile"
        ),
        "event_density_quartile": _stratum(frame, "event_density_current_quartile"),
        "roi_area_quartile": _stratum(frame, "bbox_area_fraction_current_quartile"),
        "transport_entropy_quartile": _stratum(frame, "a8_transport_entropy_quartile"),
    }
    winner_counts: dict[str, int] = {}
    raw_matrix = frame[[f"{arm}_raw_mid" for arm in ARMS]].to_numpy(dtype=np.float64)
    raw_matrix[~np.isfinite(raw_matrix)] = np.inf
    winner_index = np.argmin(raw_matrix, axis=1)
    for index, arm in enumerate(ARMS):
        winner_counts[arm] = int(np.count_nonzero(winner_index == index))

    report: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v6_oof_garl_gap_diagnostic_v1",
        "status": "completed_train_only_grouped_development_diagnostic",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "population": {
            "rows": len(frame),
            "sequences": int(frame["sequence_id"].nunique()),
            "tracks": int(frame[["sequence_id", "track_id"]].drop_duplicates().shape[0]),
        },
        "official_bucket_metrics": bucket_decomposition,
        "gap_contributions_vs_garl": gap_contributions,
        "feature_associations_model_minus_garl_raw_MiD": associations,
        "strata_raw_MiD_diagnostic": strata,
        "per_sample_lowest_raw_MiD_counts": winner_counts,
        "interpretation_contract": {
            "association_is_not_causation": True,
            "raw_sample_MiD_strata_are_diagnostic_not_the_primary_macro_metric": True,
            "A5_is_geometry_unconstrained": True,
            "public_validation_opened": False,
            "private_test_opened": False,
            "promotion_authorized": False,
        },
        "sources": {
            "aggregate": {
                "path": str(aggregate_path.resolve()),
                "sha256": _sha256(aggregate_path),
                "artifact_sha256": aggregate["artifact_sha256"],
            },
            "d0": {
                "path": str(d0_path.resolve()),
                "sha256": _sha256(d0_path),
                "artifact_sha256": d0["artifact_sha256"],
            },
            "d0_rows": {"path": str(d0_rows_path.resolve()), "sha256": _sha256(d0_rows_path)},
        },
    }
    sign_artifact(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--d0", type=Path, required=True)
    parser.add_argument("--d0-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(
            aggregate_path=args.aggregate.resolve(strict=True),
            d0_path=args.d0.resolve(strict=True),
            d0_rows_path=args.d0_rows.resolve(strict=True),
            output_path=args.output.resolve(),
        )
    except Exception as error:
        parser.exit(2, f"V6 gap analysis failed: {type(error).__name__}: {error}\n")
    print(
        json.dumps(
            {"output": str(args.output), "gap_contributions": report["gap_contributions_vs_garl"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
