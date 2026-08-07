#!/usr/bin/env python3
"""Run the train-only Object Event TTC v4.11 sign-router screen."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from e_jepa_ttc.object_event_v4_4 import official_eap_metrics, pearson
from e_jepa_ttc.object_event_v4_10 import track_cluster_bootstrap
from e_jepa_ttc.object_event_v4_11 import (
    IDENTITY_COLUMNS,
    TARGET_COLUMNS,
    V411RouterConfig,
    align_seed_experts,
    apply_negative_repair,
    router_features,
    routing_metrics,
    select_router_train_only,
    validation_gates,
)


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _config_from_yaml(path: Path) -> V411RouterConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = payload.get("router", {})
    if not isinstance(values, Mapping):
        raise ValueError("config router section must be a mapping")
    normalized = dict(values)
    for key in ("seeds", "l2_grid", "negative_threshold_grid", "flip_scale_grid"):
        if key in normalized:
            normalized[key] = tuple(normalized[key])
    return V411RouterConfig(**normalized)


def _load_seed_frames(run_root: Path, seeds: tuple[int, ...], split: str) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        seed_root = run_root / f"seed-{seed}"
        summary_path = seed_root / "summary.json"
        prediction_path = seed_root / f"{split}_predictions.csv"
        if not summary_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"missing complete v4.9 seed {seed} {split} artifacts")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("artifact_type") != "object_event_v4_9_fixed_event_fusion":
            raise RuntimeError(f"seed {seed} is not a v4.9 fusion artifact")
        if summary.get("status") not in {"fusion_screen_passed", "fusion_screen_failed"}:
            raise RuntimeError(f"seed {seed} v4.9 run is incomplete")
        alpha = float(summary.get("fusion_config", {}).get("alpha", -1.0))
        if alpha != 0.5:
            raise RuntimeError(f"seed {seed} does not use fixed alpha=0.5")
        frames[seed] = pd.read_csv(prediction_path)
    return frames


def _add_official_metrics(
    metrics: dict[str, object],
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    config: V411RouterConfig,
) -> None:
    metrics["official_eap"] = official_eap_metrics(
        frame["target_expansion"].to_numpy(dtype=np.float64),
        prediction,
        frame["delta_t_s"].to_numpy(dtype=np.float64),
        frame["target_ttc_s"].to_numpy(dtype=np.float64),
        max_abs_expansion=config.max_abs_expansion,
    )


def _route_branch(
    frame: pd.DataFrame,
    *,
    branch: str,
    model: object,
    selected: Mapping[str, object],
    config: V411RouterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, fixed = router_features(frame, branch=branch, seeds=config.seeds)
    probability = model.predict_negative_probability(features)  # type: ignore[attr-defined]
    routed, repaired = apply_negative_repair(
        fixed,
        probability,
        threshold=float(selected["negative_threshold"]),
        flip_scale=float(selected["flip_scale"]),
        minimum_flip_magnitude=config.minimum_flip_magnitude,
    )
    return routed, probability, repaired


def run(*, run_root: Path, config_path: Path, output_dir: Path) -> dict[str, object]:
    config = _config_from_yaml(config_path)
    train = align_seed_experts(
        _load_seed_frames(run_root, config.seeds, "train"),
        split_name="train",
        seeds=config.seeds,
    )
    validation = align_seed_experts(
        _load_seed_frames(run_root, config.seeds, "validation"),
        split_name="validation",
        seeds=config.seeds,
    )

    model, selected, candidates, train_oof = select_router_train_only(train, config=config)

    baseline_train_features, baseline_train = router_features(
        train, branch="event", seeds=config.seeds
    )
    del baseline_train_features
    baseline_validation_features, baseline_validation = router_features(
        validation, branch="event", seeds=config.seeds
    )
    del baseline_validation_features

    train_baseline_metrics, _ = routing_metrics(train, baseline_train, config=config)
    train_routed = train_oof["routed_prediction_expansion"].to_numpy(dtype=np.float64)
    train_routed_metrics, train_per_sequence = routing_metrics(train, train_routed, config=config)
    _add_official_metrics(train_baseline_metrics, train, baseline_train, config=config)
    _add_official_metrics(train_routed_metrics, train, train_routed, config=config)

    validation_event, validation_probability, validation_repaired = _route_branch(
        validation,
        branch="event",
        model=model,
        selected=selected,
        config=config,
    )
    validation_zero, _, _ = _route_branch(
        validation,
        branch="zero",
        model=model,
        selected=selected,
        config=config,
    )
    validation_shuffled, _, _ = _route_branch(
        validation,
        branch="shuffled",
        model=model,
        selected=selected,
        config=config,
    )

    validation_baseline_metrics, _ = routing_metrics(
        validation, baseline_validation, config=config
    )
    validation_routed_metrics, validation_per_sequence = routing_metrics(
        validation, validation_event, config=config
    )
    _add_official_metrics(
        validation_baseline_metrics, validation, baseline_validation, config=config
    )
    _add_official_metrics(
        validation_routed_metrics, validation, validation_event, config=config
    )

    target = validation["target_expansion"].to_numpy(dtype=np.float64)
    event_pearson = pearson(target, validation_event)
    dependence = {
        "zero_event_pearson_drop": event_pearson - pearson(target, validation_zero),
        "zero_event_mean_abs_change": float(np.mean(np.abs(validation_event - validation_zero))),
        "shuffled_event_pearson_drop": event_pearson
        - pearson(target, validation_shuffled),
        "shuffled_event_mean_abs_change": float(
            np.mean(np.abs(validation_event - validation_shuffled))
        ),
    }

    validation_output = validation.loc[:, [*IDENTITY_COLUMNS, *TARGET_COLUMNS]].copy()
    validation_output["fixed_prediction_expansion"] = baseline_validation
    validation_output["negative_probability"] = validation_probability
    validation_output["routed_prediction_expansion"] = validation_event
    validation_output["routed_zero_events_expansion"] = validation_zero
    validation_output["routed_shuffled_mean_expansion"] = validation_shuffled
    validation_output["negative_repair"] = validation_repaired
    validation_output["seed_prediction_std"] = np.std(
        np.column_stack(
            [
                validation[f"event_fixed_seed_{seed}"].to_numpy(dtype=np.float64)
                for seed in config.seeds
            ]
        ),
        axis=1,
        ddof=0,
    )

    bootstrap_frame = validation_output.copy()
    bootstrap = track_cluster_bootstrap(
        bootstrap_frame,
        prediction_column="routed_prediction_expansion",
        repeats=config.track_bootstrap_repeats,
        seed=411,
    )
    gates = validation_gates(
        baseline=validation_baseline_metrics,
        routed=validation_routed_metrics,
        dependence=dependence,
        bootstrap_lower=float(bootstrap["lower_95"]),
        selection_feasible=bool(selected["feasible"]),
        config=config,
    )
    passed = all(gates.values())

    output_dir.mkdir(parents=True, exist_ok=False)
    candidates.sort_values(
        [
            "feasible",
            "minimum_sequence_negative_accuracy",
            "negative_accuracy",
            "balanced_sign_accuracy",
            "pearson",
        ],
        ascending=[False, False, False, False, False],
        kind="stable",
    ).to_csv(output_dir / "train_grouped_cv_candidates.csv", index=False)
    train_oof.to_csv(output_dir / "train_oof_predictions.csv", index=False)
    train_per_sequence.to_csv(output_dir / "train_oof_per_sequence.csv", index=False)
    validation_output.to_csv(output_dir / "validation_predictions.csv", index=False)
    validation_per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)
    _write_json(output_dir / "router_model.json", model.to_jsonable())

    summary: dict[str, object] = {
        "artifact_type": "object_event_v4_11_train_only_sign_router",
        "status": "sign_router_passed" if passed else "sign_router_failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_v49_run_root": str(run_root.resolve()),
        "config": asdict(config),
        "selected_router": dict(selected),
        "train_oof_baseline_metrics": train_baseline_metrics,
        "train_oof_routed_metrics": train_routed_metrics,
        "validation_baseline_metrics": validation_baseline_metrics,
        "validation_routed_metrics": validation_routed_metrics,
        "validation_event_dependence": dependence,
        "validation_track_cluster_bootstrap_pearson": bootstrap,
        "validation_repair_count": int(validation_repaired.sum()),
        "validation_repair_fraction": float(validation_repaired.mean()),
        "gates": gates,
        "passed": passed,
        "scientific_contract": {
            "router_fit_uses_train_predictions_only": True,
            "hyperparameters_selected_by_leave_one_train_sequence_out": True,
            "sequence_and_track_ids_are_groups_not_features": True,
            "router_features_are_seed_invariant_expert_outputs": True,
            "nonconvex_negative_repair_can_flip_unanimous_false_positive_experts": True,
            "boxes_and_visible_heights_are_not_router_inputs": True,
            "validation_labels_are_evaluation_only": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
            "advance_to_integrated_sign_head": passed,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v49-run-root",
        type=Path,
        default=Path("artifacts/debug/object_event_v4_9_fixed_fusion"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiment/e_jepa_garl_object_event_sign_router_v4_11.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/debug/object_event_v4_11_sign_router"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    try:
        result = run(
            run_root=args.v49_run_root,
            config_path=args.config,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_11_failure",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.output_dir / "failure.json", failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if bool(result["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
