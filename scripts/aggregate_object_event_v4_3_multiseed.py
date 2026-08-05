#!/usr/bin/env python3
"""Aggregate Object Event v4.2 seed repeats into a fail-closed v4.3 decision.

The fixed-split v4.2 screen established genuine event dependence for seed 7, but
one held-out sequence had very weak negative-sign recall.  This aggregator does
not allow a strong global mean to hide that failure.  It aligns predictions by
sample identity, evaluates each seed, an equal-weight ensemble, track-cluster
bootstrap uncertainty, pairwise seed agreement, and per-sequence sign metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class V43AggregateConfig:
    seeds: tuple[int, ...] = (7, 13, 23)
    require_all_seed_screens: bool = True
    mean_pearson_gate: float = 0.45
    worst_pearson_gate: float = 0.35
    pearson_std_gate: float = 0.10
    mean_balanced_sign_gate: float = 0.68
    worst_balanced_sign_gate: float = 0.60
    mean_negative_accuracy_gate: float = 0.55
    ensemble_pearson_gate: float = 0.55
    ensemble_track_bootstrap_lower_gate: float = 0.45
    ensemble_sequence_macro_pearson_gate: float = 0.45
    ensemble_min_sequence_pearson_gate: float = 0.30
    ensemble_balanced_sign_gate: float = 0.70
    ensemble_negative_accuracy_gate: float = 0.60
    ensemble_expansion_mae_gate: float = 0.022
    ensemble_saturation_gate: float = 0.08
    per_sequence_negative_min_count: int = 20
    ensemble_min_sequence_negative_accuracy_gate: float = 0.20
    pairwise_prediction_pearson_gate: float = 0.75
    mean_sample_prediction_std_gate: float = 0.018
    track_bootstrap_repeats: int = 3000

    def __post_init__(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("v4.3 requires at least three unique seeds")
        if self.per_sequence_negative_min_count < 1:
            raise ValueError("per_sequence_negative_min_count must be positive")
        if self.track_bootstrap_repeats < 100:
            raise ValueError("track_bootstrap_repeats must be at least 100")


def _load_config(path: Path) -> V43AggregateConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.3 aggregate config must be a mapping")
    allowed = {field.name for field in fields(V43AggregateConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown v4.3 aggregate fields: {unknown}")
    values = dict(raw)
    values["seeds"] = tuple(int(seed) for seed in values.get("seeds", (7, 13, 23)))
    return V43AggregateConfig(**values)


def _finite_float(value: object, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError(f"Pearson shape mismatch: {target.shape} != {prediction.shape}")
    finite = np.isfinite(target) & np.isfinite(prediction)
    target = target[finite]
    prediction = prediction[finite]
    if target.size < 2 or np.std(target) <= 1.0e-12 or np.std(prediction) <= 1.0e-12:
        return 0.0
    value = float(np.corrcoef(target, prediction)[0, 1])
    return value if math.isfinite(value) else 0.0


def _branch_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    positive = target >= 0.0
    negative = target < 0.0
    positive_accuracy = float(np.mean(prediction[positive] >= 0.0)) if positive.any() else 0.0
    negative_accuracy = float(np.mean(prediction[negative] < 0.0)) if negative.any() else 0.0
    delta_t = frame["delta_t_s"].to_numpy(dtype=np.float64)
    sign = np.where(prediction < 0.0, -1.0, 1.0)
    denominator = sign * np.maximum(np.abs(prediction), 1.0e-4)
    ttc = np.clip(delta_t / denominator, -60.0, 60.0)
    return {
        "count": int(len(frame)),
        "negative_count": int(negative.sum()),
        "positive_count": int(positive.sum()),
        "pearson": _pearson(target, prediction),
        "expansion_mae": float(np.mean(np.abs(target - prediction))),
        "prediction_std": float(np.std(prediction)),
        "target_std": float(np.std(target)),
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "balanced_sign_accuracy": 0.5 * (positive_accuracy + negative_accuracy),
        "ttc_saturation_rate": float(np.mean(np.abs(ttc) >= 60.0 * (1.0 - 1.0e-6))),
    }


def _track_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    cluster_key = frame["sequence_id"].astype(str) + "|" + frame["track_id"].astype(str)
    clusters = {key: np.flatnonzero(cluster_key.to_numpy() == key) for key in sorted(cluster_key.unique())}
    keys = np.asarray(list(clusters), dtype=object)
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        row_indices = np.concatenate([clusters[str(key)] for key in sampled])
        values[index] = _pearson(target[row_indices], prediction[row_indices])
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        "cluster_count": int(len(keys)),
        "repeats": int(repeats),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
    }


def _load_run(run_root: Path, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = run_root / f"seed-{seed}"
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "validation_predictions.csv"
    if not summary_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(f"Missing v4.2 outputs for seed {seed}: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    required = {
        "sequence_id", "sample_token", "track_id", "delta_t_s",
        "target_ttc_s", "target_expansion", "prediction_expansion",
        "zero_events_expansion", "shuffled_mean_expansion",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Seed {seed} predictions missing columns: {missing}")
    predictions = predictions.sort_values(
        ["sequence_id", "sample_token", "track_id"], kind="stable"
    ).reset_index(drop=True)
    return summary, predictions


def _align_predictions(seed_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    seeds = sorted(seed_frames)
    base = seed_frames[seeds[0]][[
        "sequence_id", "sample_token", "track_id", "delta_t_s",
        "target_ttc_s", "target_expansion",
    ]].copy()
    keys = ["sequence_id", "sample_token", "track_id"]
    for seed in seeds:
        frame = seed_frames[seed]
        if not base[keys].equals(frame[keys]):
            raise ValueError(f"Prediction identities do not align for seed {seed}")
        if not np.allclose(base["target_expansion"], frame["target_expansion"], atol=1.0e-8):
            raise ValueError(f"Targets do not align for seed {seed}")
        base[f"prediction_seed_{seed}"] = frame["prediction_expansion"].to_numpy()
        base[f"zero_seed_{seed}"] = frame["zero_events_expansion"].to_numpy()
        base[f"shuffled_seed_{seed}"] = frame["shuffled_mean_expansion"].to_numpy()
    prediction_columns = [f"prediction_seed_{seed}" for seed in seeds]
    zero_columns = [f"zero_seed_{seed}" for seed in seeds]
    shuffled_columns = [f"shuffled_seed_{seed}" for seed in seeds]
    base["ensemble_expansion"] = base[prediction_columns].mean(axis=1)
    base["ensemble_zero_events_expansion"] = base[zero_columns].mean(axis=1)
    base["ensemble_shuffled_expansion"] = base[shuffled_columns].mean(axis=1)
    base["seed_prediction_std"] = base[prediction_columns].std(axis=1, ddof=0)
    return base


def _pairwise(seed_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seeds = sorted(seed_frames)
    for index, seed_a in enumerate(seeds):
        prediction_a = seed_frames[seed_a]["prediction_expansion"].to_numpy(dtype=np.float64)
        for seed_b in seeds[index + 1 :]:
            prediction_b = seed_frames[seed_b]["prediction_expansion"].to_numpy(dtype=np.float64)
            rows.append({
                "seed_a": seed_a,
                "seed_b": seed_b,
                "prediction_pearson": _pearson(prediction_a, prediction_b),
                "mean_abs_difference": float(np.mean(np.abs(prediction_a - prediction_b))),
                "sign_agreement": float(np.mean((prediction_a < 0.0) == (prediction_b < 0.0))),
            })
    return pd.DataFrame.from_records(rows)


def aggregate(
    *,
    run_root: Path,
    config: V43AggregateConfig,
    output_dir: Path,
) -> dict[str, Any]:
    summaries: dict[int, dict[str, Any]] = {}
    seed_frames: dict[int, pd.DataFrame] = {}
    per_seed_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        summary, frame = _load_run(run_root, seed)
        summaries[seed] = summary
        seed_frames[seed] = frame
        metrics = cast(dict[str, Any], summary["validation_metrics"])
        event = cast(dict[str, Any], metrics["event"])
        sequence = cast(dict[str, Any], metrics["per_sequence"])
        dependence = cast(dict[str, Any], metrics["event_dependence"])
        per_seed_rows.append({
            "seed": seed,
            "screen_passed": bool(summary.get("screen_passed", False)),
            "best_epoch": int(summary.get("best_epoch", 0)),
            "pearson": _finite_float(event.get("pearson")),
            "balanced_sign_accuracy": _finite_float(event.get("balanced_sign_accuracy")),
            "negative_accuracy": _finite_float(event.get("negative_accuracy")),
            "expansion_mae": _finite_float(event.get("expansion_mae"), default=float("inf")),
            "saturation": _finite_float(event.get("ttc_saturation_rate"), default=float("inf")),
            "sequence_macro_pearson": _finite_float(sequence.get("macro_pearson")),
            "minimum_sequence_pearson": _finite_float(sequence.get("minimum_pearson")),
            "zero_event_pearson_drop": _finite_float(dependence.get("zero_event_pearson_drop")),
            "shuffled_event_pearson_drop": _finite_float(dependence.get("shuffled_event_pearson_drop")),
        })
    per_seed = pd.DataFrame.from_records(per_seed_rows).sort_values("seed").reset_index(drop=True)
    aligned = _align_predictions(seed_frames)
    pairwise = _pairwise(seed_frames)

    ensemble_metrics = _branch_metrics(aligned, "ensemble_expansion")
    zero_metrics = _branch_metrics(aligned, "ensemble_zero_events_expansion")
    shuffled_metrics = _branch_metrics(aligned, "ensemble_shuffled_expansion")
    ensemble_metrics["zero_event_pearson_drop"] = ensemble_metrics["pearson"] - zero_metrics["pearson"]
    ensemble_metrics["shuffled_event_pearson_drop"] = ensemble_metrics["pearson"] - shuffled_metrics["pearson"]
    ensemble_metrics["zero_event_mean_abs_change"] = float(
        np.mean(np.abs(aligned["ensemble_expansion"] - aligned["ensemble_zero_events_expansion"]))
    )
    ensemble_metrics["shuffled_event_mean_abs_change"] = float(
        np.mean(np.abs(aligned["ensemble_expansion"] - aligned["ensemble_shuffled_expansion"]))
    )
    ensemble_metrics["mean_sample_prediction_std"] = float(aligned["seed_prediction_std"].mean())
    ensemble_metrics["p95_sample_prediction_std"] = float(aligned["seed_prediction_std"].quantile(0.95))

    sequence_rows: list[dict[str, Any]] = []
    for sequence_id, frame in aligned.groupby("sequence_id", sort=True):
        row = {"sequence_id": sequence_id, **_branch_metrics(frame, "ensemble_expansion")}
        sequence_rows.append(row)
    per_sequence = pd.DataFrame.from_records(sequence_rows)
    ensemble_metrics["sequence_macro_pearson"] = float(per_sequence["pearson"].mean())
    ensemble_metrics["minimum_sequence_pearson"] = float(per_sequence["pearson"].min())
    eligible_negative_sequences = per_sequence[
        per_sequence["negative_count"] >= config.per_sequence_negative_min_count
    ]
    minimum_sequence_negative_accuracy = (
        float(eligible_negative_sequences["negative_accuracy"].min())
        if not eligible_negative_sequences.empty else 0.0
    )
    ensemble_metrics["minimum_sequence_negative_accuracy"] = minimum_sequence_negative_accuracy
    ensemble_metrics["negative_sequence_count"] = int(len(eligible_negative_sequences))
    track_bootstrap = _track_cluster_bootstrap(
        aligned,
        prediction_column="ensemble_expansion",
        repeats=config.track_bootstrap_repeats,
        seed=43007,
    )

    pearsons = per_seed["pearson"].to_numpy(dtype=np.float64)
    balanced = per_seed["balanced_sign_accuracy"].to_numpy(dtype=np.float64)
    negative_accuracy = per_seed["negative_accuracy"].to_numpy(dtype=np.float64)
    pairwise_min = float(pairwise["prediction_pearson"].min()) if not pairwise.empty else 0.0
    gates = {
        "all_seed_screens": bool(per_seed["screen_passed"].all())
        if config.require_all_seed_screens else True,
        "mean_pearson": float(pearsons.mean()) >= config.mean_pearson_gate,
        "worst_pearson": float(pearsons.min()) >= config.worst_pearson_gate,
        "pearson_std": float(pearsons.std(ddof=0)) <= config.pearson_std_gate,
        "mean_balanced_sign": float(balanced.mean()) >= config.mean_balanced_sign_gate,
        "worst_balanced_sign": float(balanced.min()) >= config.worst_balanced_sign_gate,
        "mean_negative_accuracy": float(negative_accuracy.mean()) >= config.mean_negative_accuracy_gate,
        "ensemble_pearson": ensemble_metrics["pearson"] >= config.ensemble_pearson_gate,
        "ensemble_track_bootstrap_lower": track_bootstrap["lower_95"]
        >= config.ensemble_track_bootstrap_lower_gate,
        "ensemble_sequence_macro_pearson": ensemble_metrics["sequence_macro_pearson"]
        >= config.ensemble_sequence_macro_pearson_gate,
        "ensemble_min_sequence_pearson": ensemble_metrics["minimum_sequence_pearson"]
        >= config.ensemble_min_sequence_pearson_gate,
        "ensemble_balanced_sign": ensemble_metrics["balanced_sign_accuracy"]
        >= config.ensemble_balanced_sign_gate,
        "ensemble_negative_accuracy": ensemble_metrics["negative_accuracy"]
        >= config.ensemble_negative_accuracy_gate,
        "ensemble_expansion_mae": ensemble_metrics["expansion_mae"]
        <= config.ensemble_expansion_mae_gate,
        "ensemble_saturation": ensemble_metrics["ttc_saturation_rate"]
        <= config.ensemble_saturation_gate,
        "ensemble_min_sequence_negative_accuracy": minimum_sequence_negative_accuracy
        >= config.ensemble_min_sequence_negative_accuracy_gate,
        "pairwise_prediction_pearson": pairwise_min >= config.pairwise_prediction_pearson_gate,
        "mean_sample_prediction_std": ensemble_metrics["mean_sample_prediction_std"]
        <= config.mean_sample_prediction_std_gate,
    }
    robust_passed = all(gates.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_summary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_seed_metrics.csv", index=False)
    per_sequence.to_csv(output_dir / "ensemble_per_sequence.csv", index=False)
    aligned.to_csv(output_dir / "ensemble_validation_predictions.csv", index=False)
    result = {
        "artifact_type": "object_event_v4_3_multiseed_robustness",
        "status": "robust_passed" if robust_passed else "robust_failed",
        "created_at": datetime.now(UTC).isoformat(),
        "run_root": run_root.resolve().as_posix(),
        "config": asdict(config),
        "seeds": list(config.seeds),
        "per_seed": per_seed.to_dict(orient="records"),
        "seed_statistics": {
            "pearson_mean": float(pearsons.mean()),
            "pearson_std": float(pearsons.std(ddof=0)),
            "pearson_min": float(pearsons.min()),
            "balanced_sign_mean": float(balanced.mean()),
            "balanced_sign_min": float(balanced.min()),
            "negative_accuracy_mean": float(negative_accuracy.mean()),
        },
        "ensemble_metrics": ensemble_metrics,
        "track_cluster_bootstrap_pearson": track_bootstrap,
        "pairwise_min_prediction_pearson": pairwise_min,
        "fragile_sequences": per_sequence[
            (per_sequence["negative_count"] >= config.per_sequence_negative_min_count)
            & (per_sequence["negative_accuracy"] < config.ensemble_min_sequence_negative_accuracy_gate)
        ].to_dict(orient="records"),
        "gates": gates,
        "robust_passed": robust_passed,
        "scientific_contract": {
            "same_fixed_sequence_split_all_seeds": True,
            "same_v4_2_configuration_all_seeds": True,
            "event_only": True,
            "prediction_alignment_by_identity": True,
            "track_cluster_bootstrap": True,
            "per_sequence_sign_gate": True,
            "advance_to_grouped_cv": robust_passed,
            "advance_to_motion_or_level": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = _load_config(args.config.resolve())
        result = aggregate(
            run_root=args.run_root.resolve(),
            config=config,
            output_dir=args.output_dir.resolve(),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if bool(result["robust_passed"]) else 2
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_3_operational_failure",
            "status": "operational_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
