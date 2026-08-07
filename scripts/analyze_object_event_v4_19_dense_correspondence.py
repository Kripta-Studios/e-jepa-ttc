#!/usr/bin/env python3
"""Probe frozen v4.8 maps for dense correspondence/divergence signal."""
from __future__ import annotations

import argparse
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
from scripts.train_e_jepa_object_event_v4_12 import _align_ensemble, _read_ensemble  # noqa: E402
from scripts.train_e_jepa_object_event_v4_16 import (  # noqa: E402
    _build_frozen_consensus,
    _json_safe,
    _metrics,
    _parse_checkpoints,
    _resolve_device,
)
from e_jepa_ttc.models.object_event_v4_19 import (  # noqa: E402
    ObjectEventV419Config,
    antisymmetric_correspondence_scores,
)
from e_jepa_ttc.training.object_event_v4_19 import (  # noqa: E402
    apply_score_calibration,
    equal_physics_consensus,
    fit_score_calibration,
    pearson,
    prediction_from_score,
)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 1919
    descriptor_batch_size: int = 6


def _load_config(path: Path) -> tuple[ObjectEventV419Config, RunConfig, dict[str, float]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.19 config must be a mapping")
    return (
        ObjectEventV419Config(**dict(raw.get("model", {}))),
        RunConfig(**dict(raw.get("run", {}))),
        {str(k): float(v) for k, v in dict(raw.get("decision", {})).items()},
    )


@torch.no_grad()
def _extract_scores(
    frozen: Any,
    events: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    branch: str,
    seed: int,
    config: ObjectEventV419Config,
) -> tuple[np.ndarray, dict[str, float]]:
    if branch not in {"event", "zero", "shuffled"}:
        raise KeyError(branch)
    permutation = None
    if branch == "shuffled":
        permutation = torch.randperm(len(events), generator=torch.Generator().manual_seed(seed))
    outputs: list[torch.Tensor] = []
    seed_disagreements: list[torch.Tensor] = []
    confidence_values: list[torch.Tensor] = []
    translation_values: list[torch.Tensor] = []
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
        per_seed: list[torch.Tensor] = []
        per_seed_confidence: list[torch.Tensor] = []
        per_seed_translation: list[torch.Tensor] = []
        for extractor in frozen.extractors:
            backbone = extractor.backbone
            maps, _, foreground, _ = backbone._foreground_and_features(batch)
            divergence, radial, diag = antisymmetric_correspondence_scores(
                maps[:, 1], maps[:, 2], foreground[:, 1], foreground[:, 2], config
            )
            per_seed.append(torch.stack((divergence, radial), dim=1))
            per_seed_confidence.append(diag["mean_confidence"])
            per_seed_translation.append(diag["translation_magnitude"])
        stacked = torch.stack(per_seed, dim=0)
        median = stacked.median(dim=0).values
        disagreement = (stacked - median[None]).abs().median(dim=0).values
        outputs.append(median.float().cpu())
        seed_disagreements.append(disagreement.float().cpu())
        confidence_values.append(torch.stack(per_seed_confidence).median(dim=0).values.float().cpu())
        translation_values.append(torch.stack(per_seed_translation).median(dim=0).values.float().cpu())
    scores = torch.cat(outputs, dim=0).numpy()
    disagreement = torch.cat(seed_disagreements, dim=0).numpy()
    confidence = torch.cat(confidence_values, dim=0).numpy()
    translation = torch.cat(translation_values, dim=0).numpy()
    return scores, {
        "mean_seed_disagreement_divergence": float(np.mean(disagreement[:, 0])),
        "mean_seed_disagreement_radial": float(np.mean(disagreement[:, 1])),
        "mean_matching_confidence": float(np.mean(confidence)),
        "mean_translation_magnitude": float(np.mean(translation)),
    }


def _component_diagnostics(frame: pd.DataFrame, score: np.ndarray, name: str) -> dict[str, Any]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    neg = target < 0.0
    pos = ~neg
    sign = np.where(score < 0.0, -1.0, 1.0)
    return {
        "feature": name,
        "pearson_to_target_expansion": pearson(score, target),
        "positive_accuracy": float(np.mean(sign[pos] > 0.0)) if np.any(pos) else 1.0,
        "negative_accuracy": float(np.mean(sign[neg] < 0.0)) if np.any(neg) else 1.0,
        "mean_abs_value": float(np.mean(np.abs(score))),
    }


def _evaluate_score(frame: pd.DataFrame, score: np.ndarray, magnitude: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    prediction = prediction_from_score(score, magnitude)
    metrics, per_sequence = _metrics(frame, prediction, minimum_negatives=20)
    return metrics, per_sequence, prediction


def _decision(
    *,
    baseline: dict[str, Any],
    combined: dict[str, Any],
    train_components: list[dict[str, Any]],
    validation_components: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    pearson_ok = float(combined["pearson"]) >= float(baseline["pearson"]) - thresholds["baseline_pearson_tolerance"]
    fragile_ok = float(combined["minimum_sequence_negative_accuracy"]) >= thresholds["minimum_sequence_negative_accuracy_floor"]
    positive_ok = float(combined["positive_accuracy"]) >= thresholds["positive_accuracy_floor"]
    best_train = max(abs(float(row["pearson_to_target_expansion"])) for row in train_components)
    best_val = max(abs(float(row["pearson_to_target_expansion"])) for row in validation_components)
    if pearson_ok and fragile_ok and positive_ok:
        recommendation = "train_dense_flow_decoder_then_partial_unfreeze"
    elif best_val >= thresholds["dense_score_validation_abs_pearson_support"]:
        recommendation = "dense_correspondence_supported_train_box_pseudoflow_decoder"
    elif best_train >= thresholds["dense_score_train_abs_pearson_support"]:
        recommendation = "encoder_correspondence_sequence_specific_add_event_flow_pretraining"
    else:
        recommendation = "encoder_lacks_metric_correspondence_add_event_native_flow_pretraining"
    return {
        "recommendation": recommendation,
        "comparisons": {
            "combined_within_baseline_tolerance": pearson_ok,
            "fragile_negative_supported": fragile_ok,
            "positive_accuracy_preserved": positive_ok,
            "best_train_dense_score_abs_pearson": best_train,
            "best_validation_dense_score_abs_pearson": best_val,
            "baseline_pearson": float(baseline["pearson"]),
            "combined_pearson": float(combined["pearson"]),
        },
        "note": "Development representation probe only; official eAP test and EvTTC remain unopened.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    model_config, run_config, decision_config = _load_config(args.config)
    device = _resolve_device(args.device)
    frozen, _ = _build_frozen_consensus(
        checkpoint_paths=_parse_checkpoints(args.v48_checkpoint),
        v48_config_path=args.v48_config,
        v412_config_path=args.v412_config,
        device=device,
    )
    base_config, _, _, _, _ = _load_v48_config(args.v48_config)
    train_split, train_manifest = _materialize(args.cache_manifest, "train", input_size=base_config.input_size)
    validation_split, validation_manifest = _materialize(args.cache_manifest, "validation", input_size=base_config.input_size)
    train_frame = _align_ensemble(train_split, _read_ensemble(args.ensemble_train))
    validation_frame = _align_ensemble(validation_split, _read_ensemble(args.ensemble_validation))

    train_raw, train_extract_diag = _extract_scores(
        frozen, train_split.events, batch_size=run_config.descriptor_batch_size,
        device=device, branch="event", seed=run_config.seed, config=model_config,
    )
    validation_raw, validation_extract_diag = _extract_scores(
        frozen, validation_split.events, batch_size=run_config.descriptor_batch_size,
        device=device, branch="event", seed=run_config.seed, config=model_config,
    )
    target_train = train_frame["target_expansion"].to_numpy(dtype=np.float64)
    names = ("dense_divergence", "dense_radial_slope")
    calibrations = [
        fit_score_calibration(train_raw[:, i], target_train, minimum_scale=model_config.minimum_score_scale)
        for i in range(2)
    ]
    train_components = [apply_score_calibration(train_raw[:, i], calibrations[i]) for i in range(2)]
    validation_components = [apply_score_calibration(validation_raw[:, i], calibrations[i]) for i in range(2)]
    train_combined = equal_physics_consensus(train_components[0], train_components[1])
    validation_combined = equal_physics_consensus(validation_components[0], validation_components[1])

    train_diag_rows = [_component_diagnostics(train_frame, train_components[i], names[i]) for i in range(2)]
    val_diag_rows = [_component_diagnostics(validation_frame, validation_components[i], names[i]) for i in range(2)]
    train_diag_rows.append(_component_diagnostics(train_frame, train_combined, "equal_dense_consensus"))
    val_diag_rows.append(_component_diagnostics(validation_frame, validation_combined, "equal_dense_consensus"))

    train_magnitude = np.abs(train_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64))
    validation_magnitude = np.abs(validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64))
    baseline_prediction = validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    baseline_metrics, _ = _metrics(validation_frame, baseline_prediction, minimum_negatives=20)

    evaluated: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    per_sequence_frames: list[pd.DataFrame] = []
    for name, score in (
        ("dense_divergence", validation_components[0]),
        ("dense_radial_slope", validation_components[1]),
        ("equal_dense_consensus", validation_combined),
    ):
        metrics, per_sequence, prediction = _evaluate_score(validation_frame, score, validation_magnitude)
        evaluated[name] = metrics
        predictions[name] = prediction
        tmp = per_sequence.copy()
        tmp.insert(0, "probe", name)
        per_sequence_frames.append(tmp)

    zero_raw, zero_extract_diag = _extract_scores(
        frozen, validation_split.events, batch_size=run_config.descriptor_batch_size,
        device=device, branch="zero", seed=run_config.seed, config=model_config,
    )
    shuffled_raw, shuffled_extract_diag = _extract_scores(
        frozen, validation_split.events, batch_size=run_config.descriptor_batch_size,
        device=device, branch="shuffled", seed=run_config.seed, config=model_config,
    )
    zero_components = [apply_score_calibration(zero_raw[:, i], calibrations[i]) for i in range(2)]
    shuffled_components = [apply_score_calibration(shuffled_raw[:, i], calibrations[i]) for i in range(2)]
    zero_combined = equal_physics_consensus(zero_components[0], zero_components[1])
    shuffled_combined = equal_physics_consensus(shuffled_components[0], shuffled_components[1])

    decision = _decision(
        baseline=baseline_metrics,
        combined=evaluated["equal_dense_consensus"],
        train_components=train_diag_rows,
        validation_components=val_diag_rows,
        thresholds=decision_config,
    )

    pd.DataFrame(train_diag_rows).to_csv(args.output_dir / "train_dense_score_diagnostics.csv", index=False)
    pd.DataFrame(val_diag_rows).to_csv(args.output_dir / "validation_dense_score_diagnostics.csv", index=False)
    pd.concat(per_sequence_frames, ignore_index=True).to_csv(args.output_dir / "validation_per_sequence.csv", index=False)

    rows = validation_frame[["sequence_id", "sample_token", "track_id", "target_expansion", "target_ttc_s", "delta_t_s"]].copy()
    rows["baseline_prediction_expansion"] = baseline_prediction
    rows["dense_divergence_score"] = validation_components[0]
    rows["dense_radial_slope_score"] = validation_components[1]
    rows["equal_dense_consensus_score"] = validation_combined
    for name, prediction in predictions.items():
        rows[f"{name}_prediction_expansion"] = prediction
    rows.to_csv(args.output_dir / "validation_predictions.csv", index=False)

    train_rows = train_frame[["sequence_id", "sample_token", "track_id", "target_expansion"]].copy()
    train_rows["dense_divergence_score"] = train_components[0]
    train_rows["dense_radial_slope_score"] = train_components[1]
    train_rows["equal_dense_consensus_score"] = train_combined
    train_rows.to_csv(args.output_dir / "train_scores.csv", index=False)

    result = {
        "artifact_type": "object_event_v4_19_dense_correspondence_probe",
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {"model": asdict(model_config), "run": asdict(run_config), "decision": decision_config},
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "train_only_calibration": {
            names[i]: asdict(calibrations[i]) for i in range(2)
        },
        "baseline_validation_metrics": baseline_metrics,
        "dense_divergence_validation_metrics": evaluated["dense_divergence"],
        "dense_radial_slope_validation_metrics": evaluated["dense_radial_slope"],
        "equal_dense_consensus_validation_metrics": evaluated["equal_dense_consensus"],
        "score_diagnostics": {"train": train_diag_rows, "validation": val_diag_rows},
        "diagnostics": {
            "event_extraction_train": train_extract_diag,
            "event_extraction_validation": validation_extract_diag,
            "zero_extraction": zero_extract_diag,
            "shuffled_extraction": shuffled_extract_diag,
            "zero_mean_abs_consensus_score": float(np.mean(np.abs(zero_combined))),
            "shuffled_mean_abs_consensus_score": float(np.mean(np.abs(shuffled_combined))),
            "validation_true_negative_rate": float(np.mean(validation_frame["target_expansion"].to_numpy(dtype=np.float64) < 0.0)),
            "validation_predicted_negative_rate": float(np.mean(validation_combined < 0.0)),
        },
        "decision": decision,
        "scientific_contract": {
            "no_trainable_sign_or_magnitude_head": True,
            "local_feature_matching_on_frozen_v48_maps": True,
            "translation_invariant_divergence_and_radial_slope": True,
            "endpoint_swap_antisymmetrisation": True,
            "three_true_seed_output_consensus": True,
            "only_orientation_and_scale_fit_on_train": True,
            "v410_magnitude_frozen_for_sign_isolation": True,
            "boxes_heights_sequence_ids_track_ids_not_forward_features": True,
            "validation_used_only_for_development_decision": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
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
