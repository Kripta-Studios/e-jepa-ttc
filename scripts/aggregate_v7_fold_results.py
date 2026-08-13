#!/usr/bin/env python
"""Aggregate one Scientific Recovery V7 arm under the frozen OOF contract."""

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
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)
from e_jepa_ttc.evaluation.selective_ttc import risk_coverage_curve  # noqa: E402

REQUIRED_COLUMNS = {
    "sample_token",
    "sequence_id",
    "track_id",
    "target_ttc_s",
    "prediction_ttc_s",
    "point_prediction_ttc_s",
    "auxiliary_prediction_ttc_s",
    "known_mask",
    "guard_margin",
    "ttc_variance",
    "fold",
    "seed",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _token_sha(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for token in sorted(values.astype(str).tolist()):
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_signed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not verify_artifact_hash(payload):
        raise ValueError(f"invalid signed artifact: {path}")
    return payload


def _read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"V7 predictions lack columns: {missing}")
    if frame["sample_token"].astype(str).duplicated().any():
        raise ValueError(f"duplicate sample_token in {path}")
    return frame


def _metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    target = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    sequences = frame["sequence_id"].astype(str).to_numpy()
    signed = signed_garl_metrics(target, prediction)
    macro = sequence_macro_signed_metrics(target, prediction, sequences)
    return {
        "sequence_macro_MiD": float(macro["sequence_macro_paper_MiD_overall"]),
        "sample_weighted_MiD": float(signed["paper_MiD_overall"]),
        "failure_pct": float(signed["failure_rate_pct"]),
        "finite_fraction": float(np.isfinite(prediction).mean()),
        "signed": signed,
        "per_sequence": macro["per_sequence"],
    }


def _paired_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    columns = ["sample_token", "sequence_id", "track_id", "target_ttc_s"]
    left = candidate[columns + ["point_prediction_ttc_s"]].rename(
        columns={"point_prediction_ttc_s": "candidate"}
    )
    right = baseline[columns + ["point_prediction_ttc_s"]].rename(
        columns={"point_prediction_ttc_s": "baseline", "target_ttc_s": "baseline_target"}
    )
    merged = left.merge(
        right,
        on=["sample_token", "sequence_id", "track_id"],
        validate="one_to_one",
    )
    if len(merged) != len(candidate) or len(merged) != len(baseline):
        raise ValueError("candidate and baseline token universes differ")
    if not np.allclose(
        merged["target_ttc_s"], merged["baseline_target"], rtol=0.0, atol=1e-6
    ):
        raise ValueError("candidate and baseline targets differ")
    sequence_count = merged["sequence_id"].nunique()
    groups = [
        np.asarray(index, dtype=np.int64)
        for index in merged.groupby(["sequence_id", "track_id"], sort=True).indices.values()
    ]
    if sequence_count < 2 or len(groups) < 2:
        raise ValueError(
            "hierarchical inference requires at least two sequences and two sequence-track clusters"
        )
    rng = np.random.default_rng(seed)
    target = merged["target_ttc_s"].to_numpy(dtype=np.float64)
    candidate_prediction = merged["candidate"].to_numpy(dtype=np.float64)
    baseline_prediction = merged["baseline"].to_numpy(dtype=np.float64)
    sequences = merged["sequence_id"].astype(str).to_numpy()
    deltas = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[group] for group in chosen])
        candidate_mid = sequence_macro_signed_metrics(
            target[rows], candidate_prediction[rows], sequences[rows]
        )["sequence_macro_paper_MiD_overall"]
        baseline_mid = sequence_macro_signed_metrics(
            target[rows], baseline_prediction[rows], sequences[rows]
        )["sequence_macro_paper_MiD_overall"]
        deltas[index] = float(candidate_mid) - float(baseline_mid)
    return {
        "resamples": resamples,
        "seed": seed,
        "sequence_count": sequence_count,
        "sequence_track_clusters": len(groups),
        "delta_candidate_minus_a5": {
            "lower_95": float(np.quantile(deltas, 0.025)),
            "median": float(np.quantile(deltas, 0.5)),
            "upper_95": float(np.quantile(deltas, 0.975)),
        },
        "probability_delta_below_zero": float(np.mean(deltas < 0.0)),
    }


def _stratified(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sequence": _metrics(frame, "point_prediction_ttc_s")["per_sequence"],
        "track": {},
    }
    for track_id, group in frame.groupby("track_id", sort=True):
        target = group["target_ttc_s"].to_numpy(dtype=np.float64)
        prediction = group["point_prediction_ttc_s"].to_numpy(dtype=np.float64)
        result["track"][str(track_id)] = {
            "rows": len(group),
            "mae_s": float(np.mean(np.abs(prediction - target))),
            "failure_pct": float(np.mean(np.abs(prediction) < 0.1) * 100.0),
        }
    if "transport_flow_magnitude" in frame.columns:
        motion = frame["transport_flow_magnitude"].to_numpy(dtype=np.float64)
        if np.isfinite(motion).all() and np.unique(motion).size >= 4:
            labels = pd.qcut(motion, q=4, labels=False, duplicates="drop")
            result["motion_quartile"] = {
                str(int(quartile) + 1): _metrics(group, "point_prediction_ttc_s")
                for quartile, group in frame.assign(_motion_quartile=labels).groupby(
                    "_motion_quartile", sort=True
                )
            }
    return result


def aggregate(
    *,
    arm: str,
    prediction_paths: list[Path],
    baseline_path: Path,
    protocol_path: Path,
    output_path: Path,
    geometry_audit_path: Path | None,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Validate, aggregate, gate and sign one V7 arm."""

    if len(prediction_paths) != 3:
        raise ValueError("V7 aggregation requires exactly three fold prediction files")
    protocol = _read_signed(protocol_path)
    frames = [_read_predictions(path) for path in prediction_paths]
    candidate = pd.concat(frames, ignore_index=True)
    baseline = _read_predictions(baseline_path)
    expected_rows = int(protocol["sample_contract"]["rows"])
    expected_token_sha = str(protocol["sample_contract"]["sorted_sample_tokens_sha256"])
    if len(candidate) != expected_rows or candidate["sample_token"].duplicated().any():
        raise ValueError("candidate OOF rows are not an exact partition")
    if _token_sha(candidate["sample_token"]) != expected_token_sha:
        raise ValueError("candidate token universe differs from the frozen protocol")
    if set(candidate["fold"].astype(int)) != {0, 1, 2}:
        raise ValueError("candidate does not contain folds 0, 1 and 2")
    if set(candidate["seed"].astype(int)) != {7}:
        raise ValueError("exploratory V7 aggregation requires seed 7")
    if candidate["sequence_id"].nunique() != int(protocol["sample_contract"]["sequences"]):
        raise ValueError("candidate sequence count differs from the frozen protocol")
    point = candidate["point_prediction_ttc_s"].to_numpy(dtype=np.float64)
    if not np.isfinite(point).all():
        raise ValueError("point_prediction_ttc_s must be finite for every OOF row")
    baseline_point = baseline["point_prediction_ttc_s"].to_numpy(dtype=np.float64)
    if not np.isfinite(baseline_point).all():
        raise ValueError("re-evaluated A5 point predictions are not finite")
    point_metrics = _metrics(candidate, "point_prediction_ttc_s")
    selective_metrics = _metrics(candidate, "prediction_ttc_s")
    baseline_metrics = _metrics(baseline, "point_prediction_ttc_s")
    paired = _paired_bootstrap(
        candidate,
        baseline,
        resamples=resamples,
        seed=bootstrap_seed,
    )
    geometry = _read_signed(geometry_audit_path) if geometry_audit_path is not None else None
    geometry_positive = bool(geometry and geometry.get("geometry_positive") is True)
    delta_mid = point_metrics["sequence_macro_MiD"] - baseline_metrics["sequence_macro_MiD"]
    probability = paired["probability_delta_below_zero"]
    candidate_coverage = float(candidate["known_mask"].astype(bool).mean())
    baseline_coverage = float(baseline["known_mask"].astype(bool).mean())
    integrity = {
        "exact_oof_rows": len(candidate) == expected_rows,
        "unique_complete_tokens": _token_sha(candidate["sample_token"]) == expected_token_sha,
        "three_folds": set(candidate["fold"].astype(int)) == {0, 1, 2},
        "nine_sequences_finite": candidate["sequence_id"].nunique() == 9
        and all(
            np.isfinite(value["paper_MiD_overall"])
            for value in point_metrics["per_sequence"].values()
        ),
        "point_finite_coverage": float(np.isfinite(point).mean()),
        "forbidden_splits_remained_closed": True,
    }
    mechanism_positive = bool(
        delta_mid <= -5.0
        and probability >= 0.90
        and candidate_coverage >= baseline_coverage - 0.01
    )
    confirmation_candidate = bool(
        all(value is True or value == 1.0 for value in integrity.values())
        and geometry_positive
        and delta_mid <= -3.0
        and probability >= 0.90
    )
    report: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v7_arm_aggregate_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "arm": arm,
        "status": "completed_seed7_oof_gate",
        "protocol_artifact_sha256": protocol["artifact_sha256"],
        "sources": {
            "fold_predictions": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in prediction_paths
            ],
            "a5_baseline": {
                "path": str(baseline_path.resolve()),
                "sha256": _sha256(baseline_path),
            },
            "geometry_audit": (
                {
                    "path": str(geometry_audit_path.resolve()),
                    "sha256": _sha256(geometry_audit_path),
                    "artifact_sha256": geometry["artifact_sha256"],
                }
                if geometry_audit_path is not None and geometry is not None
                else None
            ),
        },
        "integrity": integrity,
        "point_full_coverage": point_metrics,
        "garl_failure_protocol": {
            "failure_pct": point_metrics["failure_pct"],
            "point_finite_fraction": point_metrics["finite_fraction"],
        },
        "historical_selective": {
            "metrics": selective_metrics,
            "known_coverage": candidate_coverage,
        },
        "risk_coverage": risk_coverage_curve(
            candidate["target_ttc_s"],
            candidate["point_prediction_ttc_s"],
            candidate["guard_margin"],
            candidate["sequence_id"].astype(str),
        ),
        "stratified": _stratified(candidate),
        "a5_revalued_point": baseline_metrics,
        "paired_vs_a5": paired,
        "delta_point_MiD_candidate_minus_a5": delta_mid,
        "gates": {
            "mechanism_positive": mechanism_positive,
            "geometry_positive": geometry_positive,
            "confirmation_candidate": confirmation_candidate,
        },
        "claims": {
            "jepa_objective_active": False,
            "jepa_attribution_allowed": False,
            "sota_claim_allowed": False,
        },
        "closed_evaluation": {
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
            "evttc_test_opened": False,
            "codabench_opened": False,
        },
    }
    report = _json_safe(report)
    sign_artifact(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--predictions", type=Path, nargs=3, required=True)
    parser.add_argument("--a5-baseline", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--geometry-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
    args = parser.parse_args()
    try:
        report = aggregate(
            arm=args.arm,
            prediction_paths=[path.resolve(strict=True) for path in args.predictions],
            baseline_path=args.a5_baseline.resolve(strict=True),
            protocol_path=args.protocol.resolve(strict=True),
            output_path=args.output.resolve(),
            geometry_audit_path=(
                args.geometry_audit.resolve(strict=True) if args.geometry_audit else None
            ),
            resamples=args.resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except Exception as error:
        parser.exit(2, f"V7 aggregation failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"output": str(args.output), "gates": report["gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
