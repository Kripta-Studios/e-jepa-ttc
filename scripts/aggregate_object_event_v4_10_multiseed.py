#!/usr/bin/env python3
"""Aggregate true-seed v4.9 fixed-fusion screens into a robustness decision."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.object_event_v4_9 import (  # noqa: E402
    FixedFusionConfig,
    dependence_metrics,
    evaluate_prediction,
)
from e_jepa_ttc.object_event_v4_10 import (  # noqa: E402
    V410AggregateConfig,
    align_seed_fusions,
    pairwise_seed_metrics,
    track_cluster_bootstrap,
)


def _load_config(path: Path) -> V410AggregateConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("aggregate"), dict):
        raise ValueError("v4.10 config must contain an aggregate mapping")
    values = dict(raw["aggregate"])
    allowed = {field.name for field in fields(V410AggregateConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown v4.10 aggregate fields: {unknown}")
    values["seeds"] = tuple(int(seed) for seed in values.get("seeds", (7, 13, 23)))
    return V410AggregateConfig(**values)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _load_seed(run_root: Path, seed: int, split: str) -> tuple[dict[str, Any], pd.DataFrame]:
    seed_root = run_root / f"seed-{seed}"
    summary_path = seed_root / "summary.json"
    predictions_path = seed_root / f"{split}_predictions.csv"
    if not summary_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"Missing v4.9 outputs for seed {seed}: {seed_root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("artifact_type") != "object_event_v4_9_fixed_event_fusion":
        raise RuntimeError(f"Unexpected v4.9 artifact for seed {seed}")
    status = str(summary.get("status", ""))
    if status not in {"fusion_screen_passed", "fusion_screen_failed"}:
        raise RuntimeError(
            f"v4.9 seed {seed} is not a completed fixed-fusion screen: {status!r}"
        )
    alpha = float(summary.get("fusion_config", {}).get("alpha", -1.0))
    if alpha != 0.5:
        raise RuntimeError(f"v4.9 seed {seed} used alpha={alpha}, expected 0.5")
    return summary, pd.read_csv(predictions_path)


def _fixed_config(summary: Mapping[str, object]) -> FixedFusionConfig:
    raw = summary.get("fusion_config")
    if not isinstance(raw, Mapping):
        raise TypeError("v4.9 summary lacks fusion_config")
    allowed = {field.name for field in fields(FixedFusionConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    return FixedFusionConfig(**values)


def _per_seed_metrics(
    seed_frames: dict[int, pd.DataFrame],
    *,
    summaries: Mapping[int, Mapping[str, object]],
    fixed_config: FixedFusionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in sorted(seed_frames):
        metrics, _ = evaluate_prediction(
            seed_frames[seed], "fused_prediction_expansion", config=fixed_config
        )
        official = cast(Mapping[str, object], metrics["official_eap"])
        summary = summaries[seed]
        gates_raw = summary.get("gates", {})
        gates = gates_raw if isinstance(gates_raw, Mapping) else {}
        failed_gates = sorted(
            str(name) for name, passed in gates.items() if not bool(passed)
        )
        rows.append(
            {
                "seed": seed,
                "v49_screen_passed": bool(summary.get("passed")),
                "v49_status": str(summary.get("status", "unknown")),
                "v49_failed_gates": ",".join(failed_gates),
                "pearson": metrics["pearson"],
                "expansion_mae": metrics["expansion_mae"],
                "balanced_sign_accuracy": metrics["balanced_sign_accuracy"],
                "negative_accuracy": metrics["negative_accuracy"],
                "weighted_mid": official["weighted_mid"],
                "sequence_macro_pearson": metrics["sequence_macro_pearson"],
                "minimum_sequence_pearson": metrics["minimum_sequence_pearson"],
                "minimum_sequence_negative_accuracy": metrics[
                    "minimum_sequence_negative_accuracy"
                ],
            }
        )
    return pd.DataFrame.from_records(rows)


def _gates(
    *,
    per_seed: pd.DataFrame,
    ensemble: Mapping[str, object],
    dependence: Mapping[str, float],
    bootstrap: Mapping[str, float | int],
    pairwise: pd.DataFrame,
    mean_sample_std: float,
    all_seed_screens: bool,
    config: V410AggregateConfig,
) -> dict[str, bool]:
    official = cast(Mapping[str, object], ensemble["official_eap"])
    pearsons = per_seed["pearson"].to_numpy(dtype=np.float64)
    balanced = per_seed["balanced_sign_accuracy"].to_numpy(dtype=np.float64)
    negatives = per_seed["negative_accuracy"].to_numpy(dtype=np.float64)
    pairwise_min = float(pairwise["prediction_pearson"].min()) if not pairwise.empty else 0.0
    return {
        "all_seed_screens": all_seed_screens if config.require_all_seed_screens else True,
        "mean_seed_pearson": float(pearsons.mean()) >= config.mean_seed_pearson_gate,
        "worst_seed_pearson": float(pearsons.min()) >= config.worst_seed_pearson_gate,
        "seed_pearson_std": float(pearsons.std(ddof=0)) <= config.seed_pearson_std_gate,
        "mean_seed_balanced_sign": float(balanced.mean())
        >= config.mean_seed_balanced_sign_gate,
        "worst_seed_balanced_sign": float(balanced.min())
        >= config.worst_seed_balanced_sign_gate,
        "mean_seed_negative_accuracy": float(negatives.mean())
        >= config.mean_seed_negative_accuracy_gate,
        "ensemble_pearson": _finite(ensemble.get("pearson")) >= config.ensemble_pearson_gate,
        "ensemble_track_bootstrap_lower": _finite(bootstrap.get("lower_95"))
        >= config.ensemble_track_bootstrap_lower_gate,
        "ensemble_weighted_mid": _finite(official.get("weighted_mid"), default=float("inf"))
        <= config.ensemble_weighted_mid_gate,
        "ensemble_balanced_sign": _finite(ensemble.get("balanced_sign_accuracy"))
        >= config.ensemble_balanced_sign_gate,
        "ensemble_negative_accuracy": _finite(ensemble.get("negative_accuracy"))
        >= config.ensemble_negative_accuracy_gate,
        "ensemble_expansion_mae": _finite(
            ensemble.get("expansion_mae"), default=float("inf")
        )
        <= config.ensemble_expansion_mae_gate,
        "ensemble_min_sequence_pearson": _finite(ensemble.get("minimum_sequence_pearson"))
        >= config.ensemble_min_sequence_pearson_gate,
        "ensemble_min_sequence_negative_accuracy": _finite(
            ensemble.get("minimum_sequence_negative_accuracy")
        )
        >= config.ensemble_min_sequence_negative_accuracy_gate,
        "pairwise_prediction_pearson": pairwise_min
        >= config.pairwise_prediction_pearson_gate,
        "mean_sample_prediction_std": mean_sample_std
        <= config.mean_sample_prediction_std_gate,
        "zero_event_dependence": dependence["zero_event_pearson_drop"]
        >= config.zero_event_pearson_drop_gate,
        "shuffled_event_dependence": dependence["shuffled_event_pearson_drop"]
        >= config.shuffled_event_pearson_drop_gate,
    }


def aggregate(*, run_root: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    summaries: dict[int, dict[str, Any]] = {}
    train_frames: dict[int, pd.DataFrame] = {}
    validation_frames: dict[int, pd.DataFrame] = {}
    for seed in config.seeds:
        train_summary, train_frame = _load_seed(run_root, seed, "train")
        validation_summary, validation_frame = _load_seed(run_root, seed, "validation")
        if train_summary != validation_summary:
            raise RuntimeError(f"Summary changed while loading seed {seed}")
        summaries[seed] = train_summary
        train_frames[seed] = train_frame
        validation_frames[seed] = validation_frame

    source_fixed_config = _fixed_config(summaries[config.seeds[0]])
    for seed, summary in summaries.items():
        if _fixed_config(summary) != source_fixed_config:
            raise RuntimeError(f"Fusion config differs for seed {seed}")
    fixed_config = replace(
        source_fixed_config,
        per_sequence_negative_min_count=config.per_sequence_negative_min_count,
    )

    aligned_train = align_seed_fusions(train_frames, split_name="train")
    aligned_validation = align_seed_fusions(validation_frames, split_name="validation")
    train_metrics, train_per_sequence = evaluate_prediction(
        aligned_train, "fused_prediction_expansion", config=fixed_config
    )
    validation_metrics, validation_per_sequence = evaluate_prediction(
        aligned_validation, "fused_prediction_expansion", config=fixed_config
    )
    train_dependence = dependence_metrics(aligned_train)
    validation_dependence = dependence_metrics(aligned_validation)
    per_seed = _per_seed_metrics(
        validation_frames,
        summaries=summaries,
        fixed_config=fixed_config,
    )
    pairwise = pairwise_seed_metrics(validation_frames)
    bootstrap = track_cluster_bootstrap(
        aligned_validation,
        prediction_column="fused_prediction_expansion",
        repeats=config.track_bootstrap_repeats,
        seed=41007,
    )
    mean_sample_std = float(aligned_validation["seed_prediction_std"].mean())
    p95_sample_std = float(aligned_validation["seed_prediction_std"].quantile(0.95))
    all_seed_screens = all(bool(summary.get("passed")) for summary in summaries.values())
    gates = _gates(
        per_seed=per_seed,
        ensemble=validation_metrics,
        dependence=validation_dependence,
        bootstrap=bootstrap,
        pairwise=pairwise,
        mean_sample_std=mean_sample_std,
        all_seed_screens=all_seed_screens,
        config=config,
    )
    passed = all(gates.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_train.to_csv(output_dir / "ensemble_train_predictions.csv", index=False)
    aligned_validation.to_csv(output_dir / "ensemble_validation_predictions.csv", index=False)
    train_per_sequence.to_csv(output_dir / "ensemble_train_per_sequence.csv", index=False)
    validation_per_sequence.to_csv(
        output_dir / "ensemble_validation_per_sequence.csv", index=False
    )
    per_seed.to_csv(output_dir / "per_seed_summary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_seed_metrics.csv", index=False)

    pearsons = per_seed["pearson"].to_numpy(dtype=np.float64)
    balanced = per_seed["balanced_sign_accuracy"].to_numpy(dtype=np.float64)
    negatives = per_seed["negative_accuracy"].to_numpy(dtype=np.float64)
    result = {
        "artifact_type": "object_event_v4_10_true_seed_fixed_fusion_robustness",
        "status": "robust_passed" if passed else "robust_failed",
        "created_at": datetime.now(UTC).isoformat(),
        "run_root": run_root.resolve().as_posix(),
        "config": asdict(config),
        "fixed_fusion_config": asdict(fixed_config),
        "seeds": list(config.seeds),
        "per_seed": per_seed.to_dict(orient="records"),
        "seed_screen_status": {
            str(seed): {
                "passed": bool(summary.get("passed")),
                "status": str(summary.get("status", "unknown")),
                "failed_gates": sorted(
                    str(name)
                    for name, gate_passed in (
                        summary.get("gates", {})
                        if isinstance(summary.get("gates", {}), Mapping)
                        else {}
                    ).items()
                    if not bool(gate_passed)
                ),
            }
            for seed, summary in summaries.items()
        },
        "seed_statistics": {
            "pearson_mean": float(pearsons.mean()),
            "pearson_std": float(pearsons.std(ddof=0)),
            "pearson_min": float(pearsons.min()),
            "balanced_sign_mean": float(balanced.mean()),
            "balanced_sign_min": float(balanced.min()),
            "negative_accuracy_mean": float(negatives.mean()),
        },
        "train_ensemble_metrics": train_metrics,
        "validation_ensemble_metrics": validation_metrics,
        "train_event_dependence": train_dependence,
        "validation_event_dependence": validation_dependence,
        "track_cluster_bootstrap_pearson": bootstrap,
        "pairwise_min_prediction_pearson": float(pairwise["prediction_pearson"].min()),
        "mean_sample_prediction_std": mean_sample_std,
        "p95_sample_prediction_std": p95_sample_std,
        "fragile_sequences": validation_per_sequence[
            (validation_per_sequence["negative_count"] >= config.per_sequence_negative_min_count)
            & (
                validation_per_sequence["negative_accuracy"]
                < config.ensemble_min_sequence_negative_accuracy_gate
            )
        ].to_dict(orient="records"),
        "gates": gates,
        "passed": passed,
        "scientific_contract": {
            "true_seed_specific_training": True,
            "fixed_alpha_0_5_across_seeds": True,
            "same_fixed_sequence_split": True,
            "prediction_alignment_by_identity": True,
            "event_only_inference": True,
            "no_validation_fitted_gate": True,
            "completed_failed_seed_screens_are_aggregated_not_relabelled": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
            "advance_to_integrated_dual_head": passed,
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        output = args.output_dir.resolve()
        if output.exists() and any(output.iterdir()) and not args.force:
            raise FileExistsError(f"Output exists: {output}; pass --force")
        if output.exists() and args.force:
            import shutil

            shutil.rmtree(output)
        result = aggregate(
            run_root=args.run_root.resolve(),
            config_path=args.config.resolve(),
            output_dir=output,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_10_operational_failure",
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
