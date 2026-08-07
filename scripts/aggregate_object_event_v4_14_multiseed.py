#!/usr/bin/env python3
"""Aggregate locked v4.13 corrections from true v4.12 seeds 7/13/23."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.object_event_v4_4 import branch_metrics, official_eap_metrics, pearson  # noqa: E402
from e_jepa_ttc.object_event_v4_13 import ObjectEventV413Config  # noqa: E402
from e_jepa_ttc.object_event_v4_14 import (  # noqa: E402
    IDENTITY_COLUMNS,
    V414AggregateConfig,
    align_selective_seeds,
    pairwise_seed_metrics,
    robustness_gates,
)


def _load_config(path: Path) -> tuple[ObjectEventV413Config, V414AggregateConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    fusion = ObjectEventV413Config(**cast(dict[str, Any], raw["fusion"]))
    values = dict(raw["aggregate"])
    allowed = {field.name for field in fields(V414AggregateConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown v4.14 aggregate fields: {unknown}")
    values["seeds"] = tuple(int(seed) for seed in values["seeds"])
    return fusion, V414AggregateConfig(**values)


def _per_sequence(frame: pd.DataFrame, prediction: np.ndarray, minimum_negatives: int) -> pd.DataFrame:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for sequence_id, indices in frame.groupby("sequence_id", sort=True).groups.items():
        idx = np.asarray(list(indices), dtype=np.int64)
        y, p = target[idx], prediction[idx]
        negative, positive = y < 0.0, y >= 0.0
        rows.append({
            "sequence_id": str(sequence_id),
            "count": int(len(idx)),
            "negative_count": int(negative.sum()),
            "positive_count": int(positive.sum()),
            "pearson": pearson(y, p),
            "expansion_mae": float(np.mean(np.abs(y - p))),
            "positive_accuracy": float(np.mean(p[positive] >= 0.0)) if positive.any() else 0.0,
            "negative_accuracy": float(np.mean(p[negative] < 0.0)) if negative.any() else 0.0,
            "balanced_sign_accuracy": 0.5 * (
                (float(np.mean(p[positive] >= 0.0)) if positive.any() else 0.0)
                + (float(np.mean(p[negative] < 0.0)) if negative.any() else 0.0)
            ),
        })
    result = pd.DataFrame(rows)
    eligible = result[result["negative_count"] >= minimum_negatives]
    result.attrs["minimum_sequence_negative_accuracy"] = (
        float(eligible["negative_accuracy"].min()) if len(eligible) else 0.0
    )
    result.attrs["minimum_sequence_pearson"] = float(result["pearson"].min())
    return result


def _metrics(frame: pd.DataFrame, prediction: np.ndarray, minimum_negatives: int) -> tuple[dict[str, Any], pd.DataFrame]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    metrics = branch_metrics(target, prediction, frame["delta_t_s"].to_numpy(dtype=np.float64))
    per_sequence = _per_sequence(frame, prediction, minimum_negatives)
    metrics["minimum_sequence_negative_accuracy"] = per_sequence.attrs[
        "minimum_sequence_negative_accuracy"
    ]
    metrics["minimum_sequence_pearson"] = per_sequence.attrs["minimum_sequence_pearson"]
    official = official_eap_metrics(
        target,
        prediction,
        frame["delta_t_s"].to_numpy(dtype=np.float64),
        frame["target_ttc_s"].to_numpy(dtype=np.float64),
    )
    metrics["official_eap"] = official
    metrics["weighted_mid"] = float(official["weighted_mid"])
    metrics["weighted_rte_percent"] = float(official["weighted_rte_percent"])
    return metrics, per_sequence


def _track_bootstrap(frame: pd.DataFrame, prediction: np.ndarray, repeats: int) -> dict[str, float | int]:
    keys = frame["sequence_id"].astype(str) + "|" + frame["track_id"].astype(str)
    unique = np.asarray(sorted(keys.unique()), dtype=object)
    lookup = {str(key): np.flatnonzero(keys.to_numpy(dtype=object) == key) for key in unique}
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(41407)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([lookup[str(key)] for key in sampled])
        values[repeat] = pearson(target[rows], prediction[rows])
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        "cluster_count": int(len(unique)),
        "repeats": int(repeats),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
    }


def _load_seed(direction_root: Path, fusion_root: Path, seed: int, locked: ObjectEventV413Config) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    direction_dir = direction_root / f"seed-{seed}" / "screen"
    fusion_dir = fusion_root / f"seed-{seed}"
    direction_summary_path = direction_dir / "summary.json"
    direction_predictions_path = direction_dir / "validation_predictions.csv"
    fusion_summary_path = fusion_dir / "summary.json"
    fusion_predictions_path = fusion_dir / "validation_predictions.csv"
    for path in (direction_summary_path, direction_predictions_path, fusion_summary_path, fusion_predictions_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    direction_summary = json.loads(direction_summary_path.read_text(encoding="utf-8"))
    if direction_summary.get("artifact_type") != "object_event_v4_12_reversal_balanced_directional_sign":
        raise RuntimeError(f"Unexpected v4.12 artifact for seed {seed}")
    actual_seed = int(direction_summary.get("train_config", {}).get("seed", -1))
    if actual_seed != seed:
        raise RuntimeError(f"v4.12 seed mismatch: path seed {seed}, summary seed {actual_seed}")
    if not direction_summary.get("scientific_contract", {}).get("exact_descriptor_antisymmetry"):
        raise RuntimeError(f"v4.12 seed {seed} lacks exact odd symmetry")

    fusion_summary = json.loads(fusion_summary_path.read_text(encoding="utf-8"))
    if fusion_summary.get("artifact_type") != "object_event_v4_13_conservative_dual_head_fusion":
        raise RuntimeError(f"Unexpected v4.13 artifact for seed {seed}")
    if fusion_summary.get("status") not in {"selective_fusion_passed", "selective_fusion_failed"}:
        raise RuntimeError(f"Incomplete v4.13 result for seed {seed}")
    if ObjectEventV413Config(**fusion_summary["config"]) != locked:
        raise RuntimeError(f"v4.13 fusion parameters changed for seed {seed}")
    fusion_frame = pd.read_csv(fusion_predictions_path).sort_values(
        list(IDENTITY_COLUMNS), kind="stable"
    ).reset_index(drop=True)
    direction_frame = pd.read_csv(direction_predictions_path).sort_values(
        list(IDENTITY_COLUMNS), kind="stable"
    ).reset_index(drop=True)
    if not fusion_frame.loc[:, list(IDENTITY_COLUMNS)].equals(
        direction_frame.loc[:, list(IDENTITY_COLUMNS)]
    ):
        raise RuntimeError(f"v4.12/v4.13 identity mismatch for seed {seed}")
    for column in ("baseline_prediction_expansion", "negative_probability"):
        if not np.allclose(
            fusion_frame[column].to_numpy(dtype=np.float64),
            direction_frame[column].to_numpy(dtype=np.float64),
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise RuntimeError(f"v4.12/v4.13 {column} mismatch for seed {seed}")
    return fusion_summary, fusion_frame, direction_frame


def aggregate(*, direction_root: Path, fusion_root: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    locked, config = _load_config(config_path)
    fusion_summaries: dict[int, dict[str, Any]] = {}
    fusion_frames: dict[int, pd.DataFrame] = {}
    direction_frames: dict[int, pd.DataFrame] = {}
    for seed in config.seeds:
        summary, fusion_frame, direction_frame = _load_seed(
            direction_root, fusion_root, seed, locked
        )
        fusion_summaries[seed] = summary
        fusion_frames[seed] = fusion_frame
        direction_frames[seed] = direction_frame

    aligned = align_selective_seeds(fusion_frames, fusion_config=locked)
    prediction = aligned["consensus_prediction_expansion"].to_numpy(dtype=np.float64)
    consensus_metrics, per_sequence = _metrics(
        aligned, prediction, config.per_sequence_negative_min_count
    )
    baseline_metrics, _ = _metrics(
        aligned,
        aligned["baseline_prediction_expansion"].to_numpy(dtype=np.float64),
        config.per_sequence_negative_min_count,
    )

    # Event-dependence references come from the frozen directional probes and
    # are averaged only after identity alignment.
    zero_values: list[np.ndarray] = []
    shuffled_values: list[np.ndarray] = []
    for seed in config.seeds:
        direction = direction_frames[seed].sort_values(list(IDENTITY_COLUMNS), kind="stable").reset_index(drop=True)
        if not aligned.loc[:, list(IDENTITY_COLUMNS)].equals(direction.loc[:, list(IDENTITY_COLUMNS)]):
            raise RuntimeError(f"v4.12 direction identities do not align for seed {seed}")
        zero_values.append(direction["zero_events_prediction_expansion"].to_numpy(dtype=np.float64))
        shuffled_values.append(direction["shuffled_prediction_expansion"].to_numpy(dtype=np.float64))
    aligned["mean_zero_events_prediction_expansion"] = np.mean(zero_values, axis=0)
    aligned["mean_shuffled_prediction_expansion"] = np.mean(shuffled_values, axis=0)
    target = aligned["target_expansion"].to_numpy(dtype=np.float64)
    diagnostics = {
        "override_count": int(aligned["consensus_override"].sum()),
        "override_rate": float(aligned["consensus_override"].mean()),
        "mean_blend": float(aligned["consensus_blend"].mean()),
        "mean_sample_prediction_std": float(aligned["seed_selective_prediction_std"].mean()),
        "p95_sample_prediction_std": float(aligned["seed_selective_prediction_std"].quantile(0.95)),
        "mean_probability_std": float(aligned["seed_negative_probability_std"].mean()),
        "zero_event_pearson_drop": pearson(target, prediction)
        - pearson(target, aligned["mean_zero_events_prediction_expansion"].to_numpy(dtype=np.float64)),
        "shuffled_event_pearson_drop": pearson(target, prediction)
        - pearson(target, aligned["mean_shuffled_prediction_expansion"].to_numpy(dtype=np.float64)),
    }

    per_seed_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        frame = fusion_frames[seed]
        metrics, _ = _metrics(
            frame,
            frame["selective_prediction_expansion"].to_numpy(dtype=np.float64),
            config.per_sequence_negative_min_count,
        )
        per_seed_rows.append({
            "seed": seed,
            "passed": bool(fusion_summaries[seed].get("passed")),
            "status": str(fusion_summaries[seed].get("status")),
            "pearson": metrics["pearson"],
            "expansion_mae": metrics["expansion_mae"],
            "weighted_mid": metrics["weighted_mid"],
            "positive_accuracy": metrics["positive_accuracy"],
            "negative_accuracy": metrics["negative_accuracy"],
            "balanced_sign_accuracy": metrics["balanced_sign_accuracy"],
            "minimum_sequence_pearson": metrics["minimum_sequence_pearson"],
            "minimum_sequence_negative_accuracy": metrics[
                "minimum_sequence_negative_accuracy"
            ],
        })
    per_seed = pd.DataFrame(per_seed_rows)
    pairwise = pairwise_seed_metrics(fusion_frames)
    bootstrap = _track_bootstrap(aligned, prediction, config.track_bootstrap_repeats)
    gates = robustness_gates(
        per_seed=per_seed,
        consensus=cast(Mapping[str, float], consensus_metrics),
        baseline=cast(Mapping[str, float], baseline_metrics),
        diagnostics=diagnostics,
        bootstrap=cast(Mapping[str, float], bootstrap),
        pairwise=pairwise,
        config=config,
    )
    passed = all(gates.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_dir / "consensus_validation_predictions.csv", index=False)
    per_sequence.to_csv(output_dir / "consensus_validation_per_sequence.csv", index=False)
    per_seed.to_csv(output_dir / "per_seed_summary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_seed_metrics.csv", index=False)

    result = {
        "artifact_type": "object_event_v4_14_locked_dual_head_multiseed",
        "status": "locked_multiseed_passed" if passed else "locked_multiseed_failed",
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "locked_fusion": asdict(locked),
        "seeds": list(config.seeds),
        "per_seed": per_seed.to_dict(orient="records"),
        "baseline_validation_metrics": baseline_metrics,
        "consensus_validation_metrics": consensus_metrics,
        "consensus_track_bootstrap_pearson": bootstrap,
        "diagnostics": diagnostics,
        "pairwise_min_selective_prediction_pearson": float(
            pairwise["selective_prediction_pearson"].min()
        ),
        "pairwise_min_negative_probability_pearson": float(
            pairwise["negative_probability_pearson"].min()
        ),
        "fragile_sequences": per_sequence[
            (per_sequence["negative_count"] >= config.per_sequence_negative_min_count)
            & (
                per_sequence["negative_accuracy"]
                < config.consensus_min_sequence_negative_accuracy_gate
            )
        ].to_dict(orient="records"),
        "gates": gates,
        "passed": passed,
        "scientific_contract": {
            "true_seed_direction_probe_training": True,
            "v413_parameters_locked_from_seed7_before_replication": True,
            "median_probability_consensus_preregistered": True,
            "magnitude_baseline_frozen_v410_ensemble": True,
            "prediction_alignment_by_identity": True,
            "event_only_inference": True,
            "no_validation_retuning": True,
            "completed_failed_seed_screens_are_not_relabelled": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
            "advance_to_integrated_model": passed,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction-root", type=Path, required=True)
    parser.add_argument("--fusion-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            if not args.force:
                raise FileExistsError(f"Output exists: {args.output_dir}; pass --force")
            shutil.rmtree(args.output_dir)
        result = aggregate(
            direction_root=args.direction_root,
            fusion_root=args.fusion_root,
            config_path=args.config,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_14_operational_failure",
            "status": "operational_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
