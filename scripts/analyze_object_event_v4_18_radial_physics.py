#!/usr/bin/env python3
"""Run Object Event TTC v4.18 radial/divergence physics bottleneck experiment."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.train_e_jepa_object_event_v4_6 import _materialize  # noqa: E402
from scripts.train_e_jepa_object_event_v4_8 import _load_config as _load_v48_config  # noqa: E402
from scripts.train_e_jepa_object_event_v4_12 import (  # noqa: E402
    _align_ensemble,
    _read_ensemble,
)
from scripts.train_e_jepa_object_event_v4_16 import (  # noqa: E402
    _build_frozen_consensus,
    _folds,
    _json_safe,
    _metrics,
    _parse_checkpoints,
    _resolve_device,
)
from e_jepa_ttc.models.object_event_v4_18 import (  # noqa: E402
    FEATURE_NAMES,
    MonotoneOddPhysicsHead,
    ObjectEventV418Config,
    feature_scales,
    normalise_physics_features,
    radial_physics_features,
    robust_seed_consensus,
)
from e_jepa_ttc.training.object_event_v4_18 import (  # noqa: E402
    raw_physics_score,
    sign_from_negative_logit,
    train_monotone_head,
)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1818
    descriptor_batch_size: int = 12
    fold_count: int = 3
    head_epochs: int = 160
    learning_rate: float = 1.0e-2
    weight_decay: float = 1.0e-3


def _load_config(path: Path) -> tuple[ObjectEventV418Config, TrainConfig, dict[str, float]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.18 config must be a mapping")
    model = ObjectEventV418Config(**dict(raw.get("model", {})))
    train = TrainConfig(**dict(raw.get("train", {})))
    decision = {str(k): float(v) for k, v in dict(raw.get("decision", {})).items()}
    return model, train, decision


@torch.no_grad()
def _extract(
    frozen: Any,
    events: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    branch: str,
    seed: int,
    epsilon: float,
) -> torch.Tensor:
    if branch not in {"event", "zero", "shuffled"}:
        raise KeyError(branch)
    permutation = None
    if branch == "shuffled":
        permutation = torch.randperm(
            len(events), generator=torch.Generator().manual_seed(seed)
        )
    output: list[torch.Tensor] = []
    frozen.eval()
    for start in range(0, len(events), batch_size):
        end = min(start + batch_size, len(events))
        if branch == "zero":
            batch = torch.zeros_like(events[start:end])
        elif branch == "shuffled":
            assert permutation is not None
            batch = events[permutation[start:end]]
        else:
            batch = events[start:end]
        batch = batch.to(device=device, dtype=torch.float32)
        seed_features: list[torch.Tensor] = []
        for extractor in frozen.extractors:
            backbone = extractor.backbone
            _, _, foreground, activity = backbone._foreground_and_features(batch)
            seed_features.append(
                radial_physics_features(
                    foreground,
                    activity,
                    epsilon=epsilon,
                )
            )
        consensus = robust_seed_consensus(
            torch.stack(seed_features, dim=0),
            epsilon=epsilon,
        )
        output.append(consensus.float().cpu())
    return torch.cat(output, dim=0)


def _prediction_from_approach_score(score: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    magnitude = np.asarray(magnitude, dtype=np.float64)
    sign = np.where(score < 0.0, -1.0, 1.0)
    return sign * np.abs(magnitude)


def _prediction_from_negative_logit(logit: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    logit = np.asarray(logit, dtype=np.float64)
    magnitude = np.asarray(magnitude, dtype=np.float64)
    sign = np.where(logit >= 0.0, -1.0, 1.0)
    return sign * np.abs(magnitude)


def _feature_diagnostics(
    frame: pd.DataFrame,
    features: np.ndarray,
) -> list[dict[str, Any]]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(FEATURE_NAMES):
        values = features[:, index].astype(np.float64)
        if np.std(values) <= 1.0e-12 or np.std(target) <= 1.0e-12:
            p = 0.0
        else:
            p = float(np.corrcoef(values, target)[0, 1])
        sign = np.where(values < 0.0, -1.0, 1.0)
        neg = target < 0.0
        pos = ~neg
        rows.append(
            {
                "feature": name,
                "pearson_to_target_expansion": p,
                "positive_accuracy": float(np.mean(sign[pos] > 0.0)) if np.any(pos) else 1.0,
                "negative_accuracy": float(np.mean(sign[neg] < 0.0)) if np.any(neg) else 1.0,
                "mean_abs_value": float(np.mean(np.abs(values))),
            }
        )
    return rows


def _decision(
    *,
    baseline: dict[str, Any],
    physics: dict[str, Any],
    raw: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    pearson_ok = float(physics["pearson"]) >= (
        float(baseline["pearson"]) - thresholds["baseline_pearson_tolerance"]
    )
    fragile_ok = float(physics["minimum_sequence_negative_accuracy"]) >= thresholds[
        "minimum_sequence_negative_accuracy_floor"
    ]
    positive_ok = float(physics["positive_accuracy"]) >= thresholds["positive_accuracy_floor"]
    feature_support = max(
        abs(float(row["pearson_to_target_expansion"])) for row in feature_rows
    ) >= thresholds["feature_validation_abs_pearson_support"]

    if pearson_ok and fragile_ok and positive_ok:
        recommendation = "integrate_radial_physics_then_partial_unfreeze"
    elif feature_support and fragile_ok:
        recommendation = "move_radial_supervision_into_dense_encoder"
    else:
        recommendation = "foreground_geometry_insufficient_use_dense_flow_divergence_supervision"
    return {
        "recommendation": recommendation,
        "comparisons": {
            "pearson_within_baseline_tolerance": pearson_ok,
            "fragile_negative_supported": fragile_ok,
            "positive_accuracy_preserved": positive_ok,
            "at_least_one_physical_feature_has_validation_support": feature_support,
            "raw_physics_pearson": float(raw["pearson"]),
            "trained_physics_pearson": float(physics["pearson"]),
            "baseline_pearson": float(baseline["pearson"]),
        },
        "note": (
            "This is a development decision record, not an independent test gate. "
            "Official eAP test and EvTTC remain unopened."
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    started = time.perf_counter()
    model_config, train_config, decision_config = _load_config(args.config)
    device = _resolve_device(args.device)
    checkpoint_paths = _parse_checkpoints(args.v48_checkpoint)
    frozen, _ = _build_frozen_consensus(
        checkpoint_paths=checkpoint_paths,
        v48_config_path=args.v48_config,
        v412_config_path=args.v412_config,
        device=device,
    )
    base_config, _, _, _, _ = _load_v48_config(args.v48_config)
    train_split, train_manifest = _materialize(
        args.cache_manifest, "train", input_size=base_config.input_size
    )
    validation_split, validation_manifest = _materialize(
        args.cache_manifest, "validation", input_size=base_config.input_size
    )
    train_frame = _align_ensemble(train_split, _read_ensemble(args.ensemble_train))
    validation_frame = _align_ensemble(
        validation_split, _read_ensemble(args.ensemble_validation)
    )

    train_features = _extract(
        frozen,
        train_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="event",
        seed=train_config.seed,
        epsilon=model_config.epsilon,
    )
    validation_features = _extract(
        frozen,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="event",
        seed=train_config.seed,
        epsilon=model_config.epsilon,
    )
    scales = feature_scales(
        train_features,
        minimum_scale=model_config.minimum_feature_scale,
    )
    train_x = normalise_physics_features(
        train_features, scales, clip=model_config.feature_clip
    )
    validation_x = normalise_physics_features(
        validation_features, scales, clip=model_config.feature_clip
    )
    target_train = torch.as_tensor(
        train_frame["target_expansion"].to_numpy(dtype=np.float32)
    )

    held_out_folds = _folds(
        train_frame["sequence_id"].astype(str).tolist(),
        train_config.fold_count,
        train_config.seed,
    )
    all_index = np.arange(len(train_frame), dtype=np.int64)
    oof_logits = np.zeros(len(train_frame), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for fold_index, held_np in enumerate(held_out_folds):
        held = torch.as_tensor(held_np, dtype=torch.long)
        train_np = np.setdiff1d(all_index, held_np, assume_unique=False)
        fit = torch.as_tensor(train_np, dtype=torch.long)
        state, history = train_monotone_head(
            train_x,
            target_train,
            fit,
            epochs=train_config.head_epochs,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            seed=train_config.seed + fold_index,
        )
        model = MonotoneOddPhysicsHead(train_x.shape[1])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            logits = model(train_x[held]).numpy()
        oof_logits[held_np] = logits
        fold_rows.append(
            {
                "fold": fold_index,
                "held_out_sequences": sorted(
                    train_frame.iloc[held_np]["sequence_id"].astype(str).unique().tolist()
                ),
                "final_logged_loss": float(history[-1]["loss"]),
                "positive_weights": model.positive_weights().detach().numpy().tolist(),
            }
        )
        histories.append({"fold": fold_index, "history": history})

    train_magnitude = np.abs(
        train_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    )
    validation_magnitude = np.abs(
        validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    )
    oof_prediction = _prediction_from_negative_logit(oof_logits, train_magnitude)
    oof_metrics, _ = _metrics(
        train_frame,
        oof_prediction,
        minimum_negatives=20,
    )

    final_state, final_history = train_monotone_head(
        train_x,
        target_train,
        torch.arange(len(train_x), dtype=torch.long),
        epochs=train_config.head_epochs,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        seed=train_config.seed,
    )
    final_model = MonotoneOddPhysicsHead(train_x.shape[1])
    final_model.load_state_dict(final_state)
    final_model.eval()
    with torch.no_grad():
        validation_logits = final_model(validation_x).numpy()
        oddness_max = float(final_model.oddness_error(validation_x).max().item())
    physics_prediction = _prediction_from_negative_logit(
        validation_logits,
        validation_magnitude,
    )
    physics_metrics, physics_per_sequence = _metrics(
        validation_frame,
        physics_prediction,
        minimum_negatives=20,
    )

    raw_train_score = raw_physics_score(train_x).numpy()
    raw_validation_score = raw_physics_score(validation_x).numpy()
    raw_prediction = _prediction_from_approach_score(
        raw_validation_score,
        validation_magnitude,
    )
    raw_metrics, raw_per_sequence = _metrics(
        validation_frame,
        raw_prediction,
        minimum_negatives=20,
    )
    baseline_prediction = validation_frame["fused_prediction_expansion"].to_numpy(
        dtype=np.float64
    )
    baseline_metrics, _ = _metrics(
        validation_frame,
        baseline_prediction,
        minimum_negatives=20,
    )

    zero_features = _extract(
        frozen,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="zero",
        seed=train_config.seed,
        epsilon=model_config.epsilon,
    )
    shuffled_features = _extract(
        frozen,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="shuffled",
        seed=train_config.seed,
        epsilon=model_config.epsilon,
    )
    zero_x = normalise_physics_features(zero_features, scales, clip=model_config.feature_clip)
    shuffled_x = normalise_physics_features(
        shuffled_features, scales, clip=model_config.feature_clip
    )
    with torch.no_grad():
        zero_logits = final_model(zero_x).numpy()
        shuffled_logits = final_model(shuffled_x).numpy()

    train_feature_rows = _feature_diagnostics(train_frame, train_x.numpy())
    validation_feature_rows = _feature_diagnostics(
        validation_frame, validation_x.numpy()
    )
    pd.DataFrame(train_feature_rows).to_csv(
        output_dir / "train_feature_diagnostics.csv", index=False
    )
    pd.DataFrame(validation_feature_rows).to_csv(
        output_dir / "validation_feature_diagnostics.csv", index=False
    )

    prediction_rows = validation_frame[
        ["sequence_id", "sample_token", "track_id", "target_expansion", "target_ttc_s", "delta_t_s"]
    ].copy()
    prediction_rows["baseline_prediction_expansion"] = baseline_prediction
    prediction_rows["raw_physics_score"] = raw_validation_score
    prediction_rows["raw_physics_prediction_expansion"] = raw_prediction
    prediction_rows["physics_negative_logit"] = validation_logits
    prediction_rows["physics_prediction_expansion"] = physics_prediction
    prediction_rows.to_csv(output_dir / "validation_predictions.csv", index=False)
    physics_per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)
    raw_per_sequence.to_csv(output_dir / "raw_validation_per_sequence.csv", index=False)

    oof_rows = train_frame[
        ["sequence_id", "sample_token", "track_id", "target_expansion", "target_ttc_s", "delta_t_s"]
    ].copy()
    oof_rows["physics_negative_logit"] = oof_logits
    oof_rows["physics_prediction_expansion"] = oof_prediction
    oof_rows["raw_physics_score"] = raw_train_score
    oof_rows.to_csv(output_dir / "train_oof_predictions.csv", index=False)

    torch.save(
        {
            "artifact_type": "object_event_v4_18_monotone_physics_head",
            "model_state_dict": final_state,
            "feature_names": list(FEATURE_NAMES),
            "feature_scales": scales.numpy().tolist(),
            "config": asdict(model_config),
        },
        output_dir / "monotone_physics_head.pt",
    )

    target_val = validation_frame["target_expansion"].to_numpy(dtype=np.float64)
    zero_sign = _prediction_from_negative_logit(zero_logits, np.ones_like(target_val))
    shuffled_sign = _prediction_from_negative_logit(
        shuffled_logits, np.ones_like(target_val)
    )
    zero_neg_rate = float(np.mean(zero_sign < 0.0))
    shuffled_neg_rate = float(np.mean(shuffled_sign < 0.0))

    decision = _decision(
        baseline=baseline_metrics,
        physics=physics_metrics,
        raw=raw_metrics,
        feature_rows=validation_feature_rows,
        thresholds=decision_config,
    )
    result = {
        "artifact_type": "object_event_v4_18_radial_physics_bottleneck",
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {
            "model": asdict(model_config),
            "train": asdict(train_config),
            "decision": decision_config,
        },
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "feature_names": list(FEATURE_NAMES),
        "feature_scales_train_only": scales.numpy().tolist(),
        "folds": fold_rows,
        "oof_metrics": oof_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "raw_physics_validation_metrics": raw_metrics,
        "monotone_physics_validation_metrics": physics_metrics,
        "diagnostics": {
            "exact_oddness_max_abs": oddness_max,
            "positive_weights": final_model.positive_weights().detach().numpy().tolist(),
            "validation_true_negative_rate": float(np.mean(target_val < 0.0)),
            "raw_physics_predicted_negative_rate": float(np.mean(raw_prediction < 0.0)),
            "monotone_physics_predicted_negative_rate": float(np.mean(physics_prediction < 0.0)),
            "zero_event_negative_rate": zero_neg_rate,
            "shuffled_event_negative_rate": shuffled_neg_rate,
            "zero_event_mean_abs_logit": float(np.mean(np.abs(zero_logits))),
            "shuffled_event_mean_abs_logit": float(np.mean(np.abs(shuffled_logits))),
        },
        "feature_diagnostics": {
            "train": train_feature_rows,
            "validation": validation_feature_rows,
        },
        "decision": decision,
        "scientific_contract": {
            "no_flexible_temporal_descriptor_head": True,
            "explicit_2d_radial_and_divergence_geometry": True,
            "zero_bias_exact_odd_monotone_head": True,
            "three_frozen_true_seed_v48_backbones": True,
            "feature_scales_fit_on_train_only_without_centering": True,
            "grouped_train_oof_before_validation": True,
            "v410_magnitude_is_frozen_for_sign_isolation": True,
            "boxes_heights_sequence_ids_and_track_ids_not_forward_features": True,
            "validation_used_only_for_development_decision": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    (output_dir / "fold_histories.json").write_text(
        json.dumps(_json_safe(histories), indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(_json_safe(result), indent=2), encoding="utf-8"
    )
    print(json.dumps(_json_safe(result), indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--v412-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v48-checkpoint", action="append", required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
