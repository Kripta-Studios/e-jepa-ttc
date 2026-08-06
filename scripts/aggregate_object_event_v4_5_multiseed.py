#!/usr/bin/env python3
"""Aggregate v4.5 paired-reversal fine-tunes and compare them with v4.4.

The decision is fail-closed: a lower global MiD cannot hide a collapse in event
dependence, seed stability, or negative-sign recall on an eligible sequence.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.object_event_v4_4 import branch_metrics, official_eap_metrics, pearson  # noqa: E402

IDENTITY_COLUMNS = ["sequence_id", "sample_token", "track_id"]
REFERENCE_COLUMNS = [
    *IDENTITY_COLUMNS,
    "delta_t_s",
    "target_ttc_s",
    "target_expansion",
]


@dataclass(frozen=True)
class V45AggregateConfig:
    seeds: tuple[int, ...] = (7, 13, 23)
    require_all_seed_screens: bool = False
    baseline_relative_mid_improvement_gate: float = 0.06
    ensemble_pearson_gate: float = 0.58
    ensemble_balanced_sign_gate: float = 0.74
    ensemble_negative_accuracy_gate: float = 0.62
    ensemble_expansion_mae_gate: float = 0.0175
    ensemble_saturation_gate: float = 0.06
    per_sequence_negative_min_count: int = 20
    ensemble_min_sequence_negative_accuracy_gate: float = 0.20
    pairwise_prediction_pearson_gate: float = 0.78
    mean_sample_prediction_std_gate: float = 0.018
    track_bootstrap_repeats: int = 3000
    track_bootstrap_lower_gate: float = 0.50
    zero_event_pearson_drop_gate: float = 0.40
    shuffled_event_pearson_drop_gate: float = 0.40

    def __post_init__(self) -> None:
        if len(self.seeds) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("v4.5 requires at least three unique seeds")
        if self.per_sequence_negative_min_count < 1:
            raise ValueError("per_sequence_negative_min_count must be positive")
        if self.track_bootstrap_repeats < 100:
            raise ValueError("track_bootstrap_repeats must be at least 100")
        if not 0.0 <= self.baseline_relative_mid_improvement_gate < 1.0:
            raise ValueError("baseline_relative_mid_improvement_gate must lie in [0,1)")


def _load_config(path: Path) -> V45AggregateConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.5 aggregate config must be a mapping")
    allowed = {field.name for field in fields(V45AggregateConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown v4.5 aggregate fields: {unknown}")
    values = dict(raw)
    values["seeds"] = tuple(int(seed) for seed in values.get("seeds", (7, 13, 23)))
    return V45AggregateConfig(**values)


def _finite(value: object, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _load_seed(run_root: Path, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = run_root / f"seed-{seed}"
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "validation_predictions.csv"
    if not summary_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(f"Missing v4.5 outputs for seed {seed}: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(predictions_path)
    required = {
        *REFERENCE_COLUMNS,
        "prediction_expansion",
        "zero_events_expansion",
        "shuffled_mean_expansion",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns for seed {seed}: {missing}")
    if frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError(f"Duplicate prediction identities for seed {seed}")
    return cast(dict[str, Any], summary), frame.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)


def _align(seed_frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    aligned: pd.DataFrame | None = None
    for seed in sorted(seed_frames):
        frame = seed_frames[seed]
        current = frame[
            REFERENCE_COLUMNS
            + ["prediction_expansion", "zero_events_expansion", "shuffled_mean_expansion"]
        ].rename(
            columns={
                "prediction_expansion": f"prediction_seed_{seed}",
                "zero_events_expansion": f"zero_seed_{seed}",
                "shuffled_mean_expansion": f"shuffled_seed_{seed}",
            }
        )
        if aligned is None:
            aligned = current
            continue
        aligned = aligned.merge(
            current,
            on=IDENTITY_COLUMNS,
            how="inner",
            validate="one_to_one",
            suffixes=("", "_check"),
        )
        for column in ("delta_t_s", "target_ttc_s", "target_expansion"):
            check = f"{column}_check"
            if not np.allclose(
                aligned[column].to_numpy(dtype=np.float64),
                aligned[check].to_numpy(dtype=np.float64),
                rtol=1.0e-6,
                atol=1.0e-8,
            ):
                raise ValueError(f"Seed alignment changed {column}")
            aligned = aligned.drop(columns=check)
    if aligned is None:
        raise ValueError("no seed frames")
    seeds = sorted(seed_frames)
    aligned["ensemble_expansion"] = aligned[
        [f"prediction_seed_{seed}" for seed in seeds]
    ].mean(axis=1)
    aligned["ensemble_zero_events_expansion"] = aligned[
        [f"zero_seed_{seed}" for seed in seeds]
    ].mean(axis=1)
    aligned["ensemble_shuffled_expansion"] = aligned[
        [f"shuffled_seed_{seed}" for seed in seeds]
    ].mean(axis=1)
    aligned["seed_prediction_std"] = aligned[
        [f"prediction_seed_{seed}" for seed in seeds]
    ].std(axis=1, ddof=0)
    return aligned.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)


def _prediction_metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    prediction = frame[column].to_numpy(dtype=np.float64)
    delta_t = frame["delta_t_s"].to_numpy(dtype=np.float64)
    target_ttc = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    return {
        "expansion": branch_metrics(target, prediction, delta_t),
        "official_eap": official_eap_metrics(target, prediction, delta_t, target_ttc),
    }


def _pairwise(aligned: pd.DataFrame, seeds: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, seed_a in enumerate(seeds):
        a = aligned[f"prediction_seed_{seed_a}"].to_numpy(dtype=np.float64)
        for seed_b in seeds[index + 1 :]:
            b = aligned[f"prediction_seed_{seed_b}"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "prediction_pearson": pearson(a, b),
                    "mean_abs_difference": float(np.mean(np.abs(a - b))),
                    "sign_agreement": float(np.mean((a < 0.0) == (b < 0.0))),
                }
            )
    return pd.DataFrame.from_records(rows)


def _track_bootstrap(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    cluster_key = frame["sequence_id"].astype(str) + "|" + frame["track_id"].astype(str)
    clusters = {
        key: np.flatnonzero(cluster_key.to_numpy() == key)
        for key in sorted(cluster_key.unique())
    }
    keys = np.asarray(list(clusters), dtype=object)
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([clusters[str(key)] for key in sampled])
        values[index] = pearson(target[rows], prediction[rows])
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        "cluster_count": int(len(keys)),
        "repeats": int(repeats),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
    }


def aggregate(
    *,
    run_root: Path,
    baseline_summary_path: Path,
    config: V45AggregateConfig,
    output_dir: Path,
) -> dict[str, Any]:
    baseline = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    baseline_metrics = cast(Mapping[str, Any], baseline["validation_metrics"])
    baseline_neural = cast(Mapping[str, Any], baseline_metrics["neural_ensemble"])
    baseline_hybrid = cast(Mapping[str, Any], baseline_metrics["hybrid"])
    baseline_neural_mid = _finite(
        cast(Mapping[str, Any], baseline_neural["official_eap"]).get("weighted_mid"),
        default=float("inf"),
    )
    baseline_hybrid_mid = _finite(
        cast(Mapping[str, Any], baseline_hybrid["official_eap"]).get("weighted_mid"),
        default=float("inf"),
    )

    summaries: dict[int, dict[str, Any]] = {}
    frames: dict[int, pd.DataFrame] = {}
    per_seed_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        summary, frame = _load_seed(run_root, seed)
        summaries[seed] = summary
        frames[seed] = frame
        validation = cast(Mapping[str, Any], summary["validation_metrics"])
        event = cast(Mapping[str, Any], validation["event"])
        official = cast(Mapping[str, Any], validation["official_eap"])
        sequence = cast(Mapping[str, Any], validation["per_sequence"])
        per_seed_rows.append(
            {
                "seed": seed,
                "screen_passed": bool(summary.get("screen_passed", False)),
                "best_epoch": int(summary.get("best_epoch", 0)),
                "weighted_mid": _finite(official.get("weighted_mid"), default=float("inf")),
                "pearson": _finite(event.get("pearson")),
                "balanced_sign_accuracy": _finite(event.get("balanced_sign_accuracy")),
                "negative_accuracy": _finite(event.get("negative_accuracy")),
                "expansion_mae": _finite(event.get("expansion_mae"), default=float("inf")),
                "saturation": _finite(event.get("ttc_saturation_rate"), default=float("inf")),
                "minimum_sequence_negative_accuracy": _finite(
                    sequence.get("minimum_eligible_negative_accuracy")
                ),
            }
        )
    per_seed = pd.DataFrame.from_records(per_seed_rows).sort_values("seed").reset_index(drop=True)
    aligned = _align(frames)
    pairwise = _pairwise(aligned, config.seeds)
    ensemble = _prediction_metrics(aligned, "ensemble_expansion")
    zero = _prediction_metrics(aligned, "ensemble_zero_events_expansion")
    shuffled = _prediction_metrics(aligned, "ensemble_shuffled_expansion")
    expansion = cast(dict[str, Any], ensemble["expansion"])
    expansion["zero_event_pearson_drop"] = _finite(expansion["pearson"]) - _finite(
        cast(Mapping[str, Any], zero["expansion"])["pearson"]
    )
    expansion["shuffled_event_pearson_drop"] = _finite(expansion["pearson"]) - _finite(
        cast(Mapping[str, Any], shuffled["expansion"])["pearson"]
    )
    expansion["mean_sample_prediction_std"] = float(aligned["seed_prediction_std"].mean())
    expansion["p95_sample_prediction_std"] = float(aligned["seed_prediction_std"].quantile(0.95))

    sequence_rows: list[dict[str, Any]] = []
    for sequence_id, group in aligned.groupby("sequence_id", sort=True):
        metrics = _prediction_metrics(group, "ensemble_expansion")
        row = {
            "sequence_id": sequence_id,
            **cast(dict[str, Any], metrics["expansion"]),
            "weighted_mid": cast(Mapping[str, Any], metrics["official_eap"]).get("weighted_mid"),
        }
        sequence_rows.append(row)
    per_sequence = pd.DataFrame.from_records(sequence_rows)
    eligible = per_sequence[
        per_sequence["negative_count"] >= config.per_sequence_negative_min_count
    ]
    minimum_sequence_negative = (
        float(eligible["negative_accuracy"].min()) if not eligible.empty else 0.0
    )
    expansion["sequence_macro_pearson"] = float(per_sequence["pearson"].mean())
    expansion["minimum_sequence_pearson"] = float(per_sequence["pearson"].min())
    expansion["minimum_sequence_negative_accuracy"] = minimum_sequence_negative
    expansion["eligible_negative_sequence_count"] = int(len(eligible))

    bootstrap = _track_bootstrap(
        aligned,
        prediction_column="ensemble_expansion",
        repeats=config.track_bootstrap_repeats,
        seed=45007,
    )
    pairwise_min = float(pairwise["prediction_pearson"].min()) if not pairwise.empty else 0.0
    ensemble_mid = _finite(
        cast(Mapping[str, Any], ensemble["official_eap"]).get("weighted_mid"),
        default=float("inf"),
    )
    relative_mid_improvement = (
        (baseline_hybrid_mid - ensemble_mid) / baseline_hybrid_mid
        if baseline_hybrid_mid > 0.0 and math.isfinite(baseline_hybrid_mid)
        else -float("inf")
    )
    gates = {
        "all_seed_screens": bool(per_seed["screen_passed"].all())
        if config.require_all_seed_screens
        else True,
        "improves_v4_4_hybrid_mid": relative_mid_improvement
        >= config.baseline_relative_mid_improvement_gate,
        "ensemble_pearson": _finite(expansion["pearson"]) >= config.ensemble_pearson_gate,
        "ensemble_balanced_sign": _finite(expansion["balanced_sign_accuracy"])
        >= config.ensemble_balanced_sign_gate,
        "ensemble_negative_accuracy": _finite(expansion["negative_accuracy"])
        >= config.ensemble_negative_accuracy_gate,
        "ensemble_expansion_mae": _finite(expansion["expansion_mae"], default=float("inf"))
        <= config.ensemble_expansion_mae_gate,
        "ensemble_saturation": _finite(
            expansion["ttc_saturation_rate"], default=float("inf")
        )
        <= config.ensemble_saturation_gate,
        "ensemble_min_sequence_negative_accuracy": minimum_sequence_negative
        >= config.ensemble_min_sequence_negative_accuracy_gate,
        "pairwise_prediction_pearson": pairwise_min
        >= config.pairwise_prediction_pearson_gate,
        "mean_sample_prediction_std": _finite(
            expansion["mean_sample_prediction_std"], default=float("inf")
        )
        <= config.mean_sample_prediction_std_gate,
        "track_bootstrap_lower": bootstrap["lower_95"]
        >= config.track_bootstrap_lower_gate,
        "zero_event_dependence": _finite(expansion["zero_event_pearson_drop"])
        >= config.zero_event_pearson_drop_gate,
        "shuffled_event_dependence": _finite(expansion["shuffled_event_pearson_drop"])
        >= config.shuffled_event_pearson_drop_gate,
    }
    passed = all(gates.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(output_dir / "per_seed_summary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_seed_metrics.csv", index=False)
    per_sequence.to_csv(output_dir / "ensemble_per_sequence.csv", index=False)
    aligned.to_csv(output_dir / "ensemble_validation_predictions.csv", index=False)
    result = {
        "artifact_type": "object_event_v4_5_paired_reciprocal_mid_multiseed",
        "status": "paired_mid_passed" if passed else "paired_mid_failed",
        "created_at": datetime.now(UTC).isoformat(),
        "run_root": run_root.resolve().as_posix(),
        "baseline_summary": baseline_summary_path.resolve().as_posix(),
        "config": asdict(config),
        "baseline": {
            "v4_4_neural_weighted_mid": baseline_neural_mid,
            "v4_4_hybrid_weighted_mid": baseline_hybrid_mid,
        },
        "per_seed": per_seed.to_dict(orient="records"),
        "ensemble_metrics": ensemble,
        "relative_mid_improvement_vs_v4_4_hybrid": relative_mid_improvement,
        "track_cluster_bootstrap_pearson": bootstrap,
        "pairwise_min_prediction_pearson": pairwise_min,
        "fragile_sequences": per_sequence[
            (per_sequence["negative_count"] >= config.per_sequence_negative_min_count)
            & (
                per_sequence["negative_accuracy"]
                < config.ensemble_min_sequence_negative_accuracy_gate
            )
        ].to_dict(orient="records"),
        "gates": gates,
        "paired_mid_passed": passed,
        "scientific_contract": {
            "same_v4_2_architecture_all_seeds": True,
            "matching_seed_checkpoint_initialisation": True,
            "event_only": True,
            "official_mid_reported": True,
            "exact_reciprocal_reversal": True,
            "validation_is_not_official_eap_test": True,
            "evttc_not_opened": True,
            "advance_to_learned_foreground_geometry": passed,
            "advance_to_rgb_fusion": False,
            "claim_sota": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    try:
        if output_dir.exists():
            if not args.force:
                raise FileExistsError(f"Output exists: {output_dir}; pass --force")
            shutil.rmtree(output_dir)
        result = aggregate(
            run_root=args.run_root.resolve(),
            baseline_summary_path=args.baseline_summary.resolve(),
            config=_load_config(args.config.resolve()),
            output_dir=output_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if bool(result["paired_mid_passed"]) else 2
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_5_aggregate_operational_failure",
            "status": "operational_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
