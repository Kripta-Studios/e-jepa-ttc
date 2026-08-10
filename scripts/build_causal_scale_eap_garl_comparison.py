"""Build a signed, token-exact CausalScale/Garl event-only comparison artifact.

Only public validation predictions are accepted.  Confidence intervals resample
complete sequences; window-level bootstrap is intentionally unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.evaluation.bootstrap import (  # noqa: E402
    paired_sequence_bootstrap_difference,
)
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    BUCKETS,
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)

DELTA_T_S = 0.1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _bucket_names(target: np.ndarray) -> np.ndarray:
    output = np.full(target.shape, "outside", dtype=object)
    for name, lower, upper in BUCKETS:
        output[(target > lower) & (target <= upper)] = name
    return output.astype(str)


def _mid_per_sample(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        target_eta = 1.0 - DELTA_T_S / target
        prediction_eta = 1.0 - DELTA_T_S / prediction
        return np.abs(np.log(target_eta) - np.log(prediction_eta)) * 1e4


def _ratio_diagnostics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        target_ratio = np.log1p(DELTA_T_S / target)
        prediction_ratio = np.log1p(DELTA_T_S / prediction)
    known = np.isfinite(target_ratio) & np.isfinite(prediction_ratio)
    x = target_ratio[known]
    y = prediction_ratio[known]
    pearson = float(np.corrcoef(x, y)[0, 1]) if x.size > 1 else float("nan")
    slope = (
        float(np.dot(x - x.mean(), y - y.mean()) / np.dot(x - x.mean(), x - x.mean()))
        if x.size > 1 and float(np.var(x)) > 0.0
        else float("nan")
    )
    sign_valid = np.isfinite(target) & np.isfinite(prediction) & (target != 0.0)
    return {
        "known_count": int(known.sum()),
        "known_coverage": float(known.mean()),
        "log_ratio_mae": float(np.mean(np.abs(y - x))) if x.size else float("nan"),
        "log_ratio_pearson": pearson,
        "log_ratio_slope": slope,
        "ttc_sign_accuracy": (
            float(np.mean(np.sign(target[sign_valid]) == np.sign(prediction[sign_valid])))
            if np.any(sign_valid)
            else float("nan")
        ),
    }


def _distribution(prediction: np.ndarray, *, clip_abs_s: float = 60.0) -> dict[str, Any]:
    finite = prediction[np.isfinite(prediction)]
    return {
        "count": int(prediction.size),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(prediction).sum()),
        "infinite_count": int(np.isinf(prediction).sum()),
        "mean_s": float(np.mean(finite)) if finite.size else float("nan"),
        "std_s": float(np.std(finite)) if finite.size else float("nan"),
        "minimum_s": float(np.min(finite)) if finite.size else float("nan"),
        "maximum_s": float(np.max(finite)) if finite.size else float("nan"),
        "abs_clip_s": clip_abs_s,
        "saturation_count": int(np.count_nonzero(np.abs(finite) >= clip_abs_s * 0.999)),
        "saturation_rate_pct": (
            float(np.mean(np.abs(finite) >= clip_abs_s * 0.999) * 100.0)
            if finite.size
            else float("nan")
        ),
    }


def _arm_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    target = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    sequence = frame["sequence_id"].astype(str).to_numpy()
    return {
        "signed": signed_garl_metrics(target, prediction),
        "sequence_macro": sequence_macro_signed_metrics(target, prediction, sequence),
        "ratio_diagnostics": _ratio_diagnostics(target, prediction),
        "prediction_distribution": _distribution(prediction),
    }


def _paired_tail(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    causal_error = _mid_per_sample(
        frame["target_ttc_s"].to_numpy(dtype=np.float64),
        frame["causal_prediction_ttc_s"].to_numpy(dtype=np.float64),
    )
    release_error = _mid_per_sample(
        frame["target_ttc_s"].to_numpy(dtype=np.float64),
        frame["release_prediction_ttc_s"].to_numpy(dtype=np.float64),
    )
    frame = frame.copy()
    frame["causal_mid_per_sample"] = causal_error
    frame["release_mid_per_sample"] = release_error
    frame["causal_minus_release_mid"] = causal_error - release_error
    finite_pair = np.isfinite(causal_error) & np.isfinite(release_error)
    causal_wins = causal_error[finite_pair] < release_error[finite_pair]
    finite_causal = np.flatnonzero(np.isfinite(causal_error))
    tail_count = max(1, int(np.ceil(finite_causal.size * 0.10)))
    ordered = finite_causal[np.argsort(causal_error[finite_causal])[::-1]]
    tail = frame.iloc[ordered[:tail_count]].copy()
    tail_summary = {
        "finite_paired_count": int(finite_pair.sum()),
        "causal_win_count": int(causal_wins.sum()),
        "causal_win_rate": float(causal_wins.mean()) if causal_wins.size else float("nan"),
        "tie_count": int(np.count_nonzero(causal_error[finite_pair] == release_error[finite_pair])),
        "causal_top_10pct_count": tail_count,
        "causal_top_10pct_mean_mid": float(tail["causal_mid_per_sample"].mean()),
        "causal_top_10pct_sequence_counts": {
            str(key): int(value)
            for key, value in tail["sequence_id"].value_counts().sort_index().items()
        },
        "causal_top_10pct_bucket_counts": {
            str(key): int(value)
            for key, value in tail["bucket"].value_counts().sort_index().items()
        },
        "window_level_bootstrap_used": False,
    }
    columns = [
        "sample_token",
        "sequence_id",
        "track_id",
        "public_track_id",
        "timestamp_us",
        "bucket",
        "target_ttc_s",
        "causal_prediction_ttc_s",
        "release_prediction_ttc_s",
        "causal_mid_per_sample",
        "release_mid_per_sample",
        "causal_minus_release_mid",
    ]
    return tail_summary, tail.sort_values("causal_mid_per_sample", ascending=False)[columns]


def _exposure_audit(
    validation_sequences: set[str],
    official_train_assets: Path,
    official_train_labels: Path,
) -> dict[str, Any]:
    train_sequences = {
        line.strip()
        for line in official_train_assets.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    labels = pd.read_parquet(official_train_labels, columns=["sequence_id"])
    label_sequences = labels["sequence_id"].astype(str)
    exposed = sorted(validation_sequences & train_sequences)
    exposed_rows = int(np.count_nonzero(label_sequences.isin(exposed).to_numpy(dtype=bool)))
    return {
        "official_train_asset_sequence_count": len(train_sequences),
        "public_train_parquet_row_count": len(labels),
        "public_train_parquet_sequence_count": len(set(label_sequences.tolist())),
        "validation_sequences": sorted(validation_sequences),
        "validation_sequences_in_official_train_assets": exposed,
        "all_validation_sequences_exposed": set(exposed) == validation_sequences,
        "exposed_public_train_rows": exposed_rows,
        "official_train_assets_sha256": _sha256(official_train_assets),
        "official_train_labels_sha256": _sha256(official_train_labels),
    }


def build_comparison(
    *,
    causal_predictions: Path,
    causal_summary: Path,
    release_predictions: Path,
    release_metrics: Path,
    subset_data: Path,
    subset_labels: Path,
    subset_manifest: Path,
    official_train_assets: Path,
    official_train_labels: Path,
    official_config: Path,
    official_checkpoint: Path,
    output_json: Path,
    outliers_csv: Path,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 7,
) -> dict[str, Any]:
    """Validate exact rows and write the release/matched comparison tables."""

    inputs = (
        causal_predictions,
        causal_summary,
        release_predictions,
        release_metrics,
        subset_data,
        subset_labels,
        subset_manifest,
        official_train_assets,
        official_train_labels,
        official_config,
        official_checkpoint,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required comparison input not found: {path}")
    if bootstrap_iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100.")

    causal = pd.read_csv(causal_predictions)
    release = pd.read_parquet(release_predictions)
    data = pd.read_parquet(subset_data)
    labels = pd.read_parquet(subset_labels)
    _require_columns(
        causal,
        {"sample_token", "sequence_id", "target_ttc_s", "prediction_ttc_s"},
        "causal predictions",
    )
    _require_columns(
        release,
        {"sample_token", "sequence_id", "target_ttc_s", "predicted_ttc_s"},
        "release predictions",
    )
    metadata_columns = (
        "sample_token",
        "sequence_id",
        "track_id",
        "public_track_id",
        "timestamp_us",
    )
    _require_columns(data, set(metadata_columns), "subset data")
    _require_columns(labels, set(metadata_columns) | {"ttc"}, "subset labels")
    named_frames = (("causal", causal), ("release", release), ("data", data), ("labels", labels))
    for label, frame in named_frames:
        if frame["sample_token"].astype(str).duplicated().any():
            raise ValueError(f"{label} contains duplicate sample_token values.")
    token_sets = {
        "causal": set(causal["sample_token"].astype(str)),
        "release": set(release["sample_token"].astype(str)),
        "data": set(data["sample_token"].astype(str)),
        "labels": set(labels["sample_token"].astype(str)),
    }
    if len({frozenset(value) for value in token_sets.values()}) != 1:
        counts = {key: len(value) for key, value in token_sets.items()}
        raise ValueError(f"Comparison token sets are not exactly equal: {counts}")

    aligned = data[list(metadata_columns)].merge(
        labels[["sample_token", "ttc"]], on="sample_token", validate="one_to_one"
    )
    causal_columns = ["sample_token", "sequence_id", "target_ttc_s", "prediction_ttc_s"]
    causal_aligned = causal[causal_columns].copy()
    causal_aligned.columns = [
        "sample_token",
        "causal_sequence_id",
        "causal_target_ttc_s",
        "causal_prediction_ttc_s",
    ]
    aligned = aligned.merge(
        causal_aligned,
        on="sample_token",
        validate="one_to_one",
    )
    release_columns = ["sample_token", "sequence_id", "target_ttc_s", "predicted_ttc_s"]
    release_aligned = release[release_columns].copy()
    release_aligned.columns = [
        "sample_token",
        "release_sequence_id",
        "release_target_ttc_s",
        "release_prediction_ttc_s",
    ]
    aligned = aligned.merge(
        release_aligned,
        on="sample_token",
        validate="one_to_one",
    )
    aligned["target_ttc_s"] = aligned["ttc"].astype(float)
    if not (
        (aligned["sequence_id"].astype(str) == aligned["causal_sequence_id"].astype(str)).all()
        and (aligned["sequence_id"].astype(str) == aligned["release_sequence_id"].astype(str)).all()
    ):
        raise ValueError("Sequence IDs disagree after exact-token alignment.")
    target = aligned["target_ttc_s"].to_numpy(dtype=np.float64)
    for column in ("causal_target_ttc_s", "release_target_ttc_s"):
        if not np.allclose(target, aligned[column].to_numpy(dtype=np.float64), rtol=0.0, atol=1e-5):
            raise ValueError(f"Targets disagree after exact-token alignment: {column}")
    aligned["bucket"] = _bucket_names(target)

    causal_arm = _arm_metrics(aligned, "causal_prediction_ttc_s")
    release_arm = _arm_metrics(aligned, "release_prediction_ttc_s")
    def paper_mid(truth: np.ndarray, estimate: np.ndarray) -> float:
        return float(signed_garl_metrics(truth, estimate)["paper_MiD_overall"])

    bootstrap = paired_sequence_bootstrap_difference(
        target,
        aligned["release_prediction_ttc_s"].to_numpy(dtype=np.float64),
        aligned["causal_prediction_ttc_s"].to_numpy(dtype=np.float64),
        aligned["sequence_id"].astype(str).to_numpy(),
        metric=paper_mid,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    tail, outliers = _paired_tail(aligned)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    outliers_csv.parent.mkdir(parents=True, exist_ok=True)
    outliers.to_csv(outliers_csv, index=False)

    summary = _read_json(causal_summary)
    release_report = _read_json(release_metrics)
    manifest = _read_json(subset_manifest)
    exposure = _exposure_audit(
        set(aligned["sequence_id"].astype(str)), official_train_assets, official_train_labels
    )
    result: dict[str, Any] = {
        "artifact_type": "causal_scale_eap_garl_event_only_comparison_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "release_reference_complete_matched_training_pending",
        "scope": {
            "public_validation_only": True,
            "private_test_opened": False,
            "codabench_opened": False,
            "evttc_test_opened": False,
            "sample_count": len(aligned),
            "sequence_count": len(set(aligned["sequence_id"].astype(str).tolist())),
            "exact_token_equality_verified": True,
            "target_equality_verified": True,
            "sample_tokens_sha256": manifest.get("sample_tokens_sha256"),
        },
        "release_reference": {
            "label": "official release reference; unequal training budget and sequence exposure",
            "causal_scale_a0": causal_arm,
            "official_garl_event_only": release_arm,
            "paired": {
                **tail,
                "causal_minus_release_sequence_bootstrap_paper_MiD": bootstrap,
                "bootstrap_unit": "complete_sequence",
            },
            "exposure_audit": exposure,
            "training_budget": {
                "causal_scale_a0": {
                    "train_rows": 2048,
                    "train_sequences": 9,
                    "epochs_completed": len(summary.get("history", [])),
                    "selected_epoch": summary.get("selection", {}).get("best_epoch"),
                    "elapsed_seconds": summary.get("elapsed_seconds"),
                    "peak_vram_mb": summary.get("peak_vram_mb"),
                },
                "official_release": {
                    "public_train_rows": exposure["public_train_parquet_row_count"],
                    "configured_epochs": 50,
                    "validation_sequences_exposed": exposure["all_validation_sequences_exposed"],
                    "checkpoint_size_bytes": official_checkpoint.stat().st_size,
                },
            },
            "release_report_artifact_type": release_report.get("artifact_type"),
        },
        "matched_training": {
            "status": "pending",
            "reason": (
                "No official Garl run has yet been trained on the exact 2048/2048 "
                "sequence-disjoint screen rows."
            ),
            "required_protocol": {
                "same_train_rows": 2048,
                "same_validation_rows": 2048,
                "same_train_sequences": 9,
                "same_validation_sequences": 3,
                "validation_rows_used_for_training": False,
                "seed": 7,
                "selection_source": "validation_only",
            },
        },
        "diagnosis": {
            "a0_is_negative": True,
            "foreground_localization_signal": "weak",
            "temporal_scale_signal": "near_zero",
            "weak_bbox_iou": summary.get("validation_metrics", {}).get("weak_bbox_iou"),
            "reported_log_ratio_pearson": summary.get("validation_metrics", {}).get(
                "log_ratio_pearson"
            ),
            "classification": (
                "weak-box IoU is low and temporal log-ratio correlation is near zero; "
                "the primary failures are foreground localization and scale dynamics."
            ),
            "next_single_hypothesis": (
                "A1 bbox geometry-only supervision, holding rows, seed, model, optimizer, "
                "and budget fixed."
            ),
            "promotion_authorized": False,
            "sota_claim_authorized": False,
        },
        "artifacts": {
            "top_10pct_outliers": {
                "path": str(outliers_csv.resolve()),
                "sha256": _sha256(outliers_csv),
                "rows": len(outliers),
            }
        },
        "inputs": {
            path.name + f"#{index}": {"path": str(path.resolve()), "sha256": _sha256(path)}
            for index, path in enumerate(inputs)
        },
        "official_release": {
            "config_sha256": _sha256(official_config),
            "checkpoint_sha256": _sha256(official_checkpoint),
        },
    }
    sign_artifact(result)
    output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-predictions", type=Path, required=True)
    parser.add_argument("--causal-summary", type=Path, required=True)
    parser.add_argument("--release-predictions", type=Path, required=True)
    parser.add_argument("--release-metrics", type=Path, required=True)
    parser.add_argument("--subset-data", type=Path, required=True)
    parser.add_argument("--subset-labels", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--official-train-assets", type=Path, required=True)
    parser.add_argument("--official-train-labels", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--outliers-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_comparison(
            causal_predictions=args.causal_predictions,
            causal_summary=args.causal_summary,
            release_predictions=args.release_predictions,
            release_metrics=args.release_metrics,
            subset_data=args.subset_data,
            subset_labels=args.subset_labels,
            subset_manifest=args.subset_manifest,
            official_train_assets=args.official_train_assets,
            official_train_labels=args.official_train_labels,
            official_config=args.official_config,
            official_checkpoint=args.official_checkpoint,
            output_json=args.output_json,
            outliers_csv=args.outliers_csv,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
    except Exception as error:
        print(f"comparison build failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
