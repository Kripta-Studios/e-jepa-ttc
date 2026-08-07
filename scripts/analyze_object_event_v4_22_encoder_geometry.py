#!/usr/bin/env python3
"""Partially unfreeze v4.8 geometry encoders with train-only box pseudoflow supervision."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

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
from scripts.train_e_jepa_object_event_v4_12 import _align_ensemble, _load_backbone, _read_ensemble  # noqa: E402
from scripts.train_e_jepa_object_event_v4_16 import _json_safe, _metrics, _resolve_device  # noqa: E402
from e_jepa_ttc.models.object_event_v4_19 import dense_flow_scores, local_correlation_flow  # noqa: E402
from e_jepa_ttc.models.object_event_v4_22 import (  # noqa: E402
    ObjectEventV422Config,
    configure_partial_geometry_unfreeze,
    trainable_parameters,
)
from e_jepa_ttc.training.object_event_v4_19 import (  # noqa: E402
    apply_score_calibration,
    fit_score_calibration,
    prediction_from_score,
)
from e_jepa_ttc.training.object_event_v4_20 import box_affine_pseudoflow  # noqa: E402
from e_jepa_ttc.training.object_event_v4_22 import (
    ObjectEventV422LossConfig,
    encoder_pseudoflow_loss,
    vertical_log_scale_from_flow,
)  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2222
    batch_size: int = 8
    epochs: int = 8
    learning_rate: float = 2.0e-5
    weight_decay: float = 1.0e-5
    max_grad_norm: float = 1.0
    last_geometry_parameter_tensors: int = 8


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_config(path: Path) -> tuple[ObjectEventV422Config, TrainConfig, ObjectEventV422LossConfig, dict[str, float]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.22 config must be a mapping")
    return (
        _construct(ObjectEventV422Config, cast(Mapping[str, Any], raw.get("model", {}))),
        _construct(TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))),
        _construct(ObjectEventV422LossConfig, cast(Mapping[str, Any], raw.get("loss", {}))),
        {str(k): float(v) for k, v in dict(raw.get("decision", {})).items()},
    )


def _parse_checkpoints(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        result[int(seed_text)] = Path(path_text)
    if sorted(result) != [7, 13, 23]:
        raise ValueError("v4.22 requires exact seeds 7,13,23")
    return dict(sorted(result.items()))


def _foreground_pair(foreground: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    fg1 = torch.nn.functional.interpolate(
        foreground[:, 1:2].detach().float(), size=target_size, mode="bilinear", align_corners=False
    )[:, 0]
    fg2 = torch.nn.functional.interpolate(
        foreground[:, 2:3].detach().float(), size=target_size, mode="bilinear", align_corners=False
    )[:, 0]
    return torch.sqrt((fg1 * fg2).clamp_min(0.0))


def _flows_from_backbone(backbone: Any, events: torch.Tensor, config: ObjectEventV422Config) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maps, _, foreground, _ = backbone._foreground_and_features(events)
    first, second = maps[:, 1], maps[:, 2]
    fx, fy, conf_f = local_correlation_flow(
        first, second, radius=config.search_radius, temperature=config.correlation_temperature, epsilon=config.epsilon
    )
    rx, ry, conf_r = local_correlation_flow(
        second, first, radius=config.search_radius, temperature=config.correlation_temperature, epsilon=config.epsilon
    )
    foreground_pair = _foreground_pair(foreground, first.shape[-2:])
    return torch.stack((fx, fy), dim=1), torch.stack((rx, ry), dim=1), foreground_pair, 0.5 * (conf_f + conf_r)


def _train_seed(
    checkpoint: Path,
    split: Any,
    *,
    v48_config: Path,
    model_config: ObjectEventV422Config,
    train_config: TrainConfig,
    loss_config: ObjectEventV422LossConfig,
    device: torch.device,
    seed: int,
) -> tuple[Any, list[dict[str, float]], list[str], dict[str, float]]:
    torch.manual_seed(seed)
    backbone, _ = _load_backbone(v48_config_path=v48_config, checkpoint_path=checkpoint)
    backbone = backbone.to(device)
    selected = configure_partial_geometry_unfreeze(backbone, train_config.last_geometry_parameter_tensors)
    geometry = backbone.foreground_model.geometry_encoder
    initial = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in geometry.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        list(trainable_parameters(backbone)),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    count = len(split.events)
    for epoch in range(1, train_config.epochs + 1):
        order = np.arange(count, dtype=np.int64)
        rng.shuffle(order)
        totals: list[float] = []
        flow_values: list[float] = []
        div_values: list[float] = []
        vertical_values: list[float] = []
        anchor_values: list[float] = []
        for start in range(0, count, train_config.batch_size):
            batch_idx = order[start:start + train_config.batch_size]
            idx = torch.as_tensor(batch_idx, dtype=torch.long)
            events = split.events[idx].to(device=device, dtype=torch.float32)
            boxes = split.boxes_xyxy[idx].to(device=device, dtype=torch.float32)
            forward, reverse, _, _ = _flows_from_backbone(backbone, events, model_config)
            h, w = forward.shape[-2:]
            target_f, mask_f = box_affine_pseudoflow(
                boxes, source_height=split.source_height, source_width=split.source_width,
                target_height=h, target_width=w, first_index=1, second_index=2, epsilon=model_config.epsilon,
            )
            target_r, mask_r = box_affine_pseudoflow(
                boxes, source_height=split.source_height, source_width=split.source_width,
                target_height=h, target_width=w, first_index=2, second_index=1, epsilon=model_config.epsilon,
            )
            current = {name: parameter for name, parameter in geometry.named_parameters() if parameter.requires_grad}
            loss, components = encoder_pseudoflow_loss(
                forward, reverse, target_f, mask_f, target_r, mask_r, current, initial, config=loss_config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainable_parameters(backbone)), train_config.max_grad_norm)
            optimizer.step()
            totals.append(float(loss.detach()))
            flow_values.append(float(components["flow"].detach()))
            div_values.append(float(components["divergence"].detach()))
            vertical_values.append(float(components["vertical_scale"].detach()))
            anchor_values.append(float(components["encoder_anchor"].detach()))
        history.append({
            "epoch": float(epoch),
            "loss": float(np.mean(totals)),
            "flow_loss": float(np.mean(flow_values)),
            "divergence_loss": float(np.mean(div_values)),
            "vertical_scale_loss": float(np.mean(vertical_values)),
            "encoder_anchor": float(np.mean(anchor_values)),
        })
    relative_drift = float(history[-1]["encoder_anchor"])
    return backbone, history, selected, {"relative_parameter_drift": relative_drift}


@torch.no_grad()
def _score_backbone(
    backbone: Any,
    split: Any,
    *,
    batch_size: int,
    config: ObjectEventV422Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    divergence_scores: list[torch.Tensor] = []
    vertical_scores: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    translations: list[torch.Tensor] = []
    for start in range(0, len(split.events), batch_size):
        events = split.events[start:start + batch_size].to(device=device, dtype=torch.float32)
        forward, reverse, foreground, confidence_pair = _flows_from_backbone(backbone, events, config)
        conf = confidence_pair.clamp(0.0, 1.0)
        div_f, _, trans_f = dense_flow_scores(
            forward[:, 0], forward[:, 1], foreground, conf,
            foreground_floor=config.foreground_floor, confidence_floor=config.confidence_floor, epsilon=config.epsilon,
        )
        div_r, _, trans_r = dense_flow_scores(
            reverse[:, 0], reverse[:, 1], foreground, conf,
            foreground_floor=config.foreground_floor, confidence_floor=config.confidence_floor, epsilon=config.epsilon,
        )
        vertical_f = vertical_log_scale_from_flow(forward, foreground, epsilon=config.epsilon)
        vertical_r = vertical_log_scale_from_flow(reverse, foreground, epsilon=config.epsilon)
        divergence_scores.append((0.5 * (div_f - div_r)).float().cpu())
        vertical_scores.append((0.5 * (vertical_f - vertical_r)).float().cpu())
        confidences.append(conf.mean(dim=(-2, -1)).float().cpu())
        translations.append((0.5 * (trans_f + trans_r)).float().cpu())
    return (
        torch.cat(divergence_scores).numpy().astype(np.float64),
        torch.cat(vertical_scores).numpy().astype(np.float64),
        {
            "mean_matching_confidence": float(torch.cat(confidences).mean()),
            "mean_translation_magnitude": float(torch.cat(translations).mean()),
        },
    )


def _score_metrics(frame: pd.DataFrame, score: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    predicted_negative = score < 0.0
    target_negative = target < 0.0
    p = float(np.corrcoef(score, target)[0, 1]) if np.std(score) > 1.0e-12 else 0.0
    result = {
        "count": float(len(frame)),
        "pearson_to_target_expansion": p,
        "positive_accuracy": float(np.mean(~predicted_negative[~target_negative])) if np.any(~target_negative) else 0.0,
        "negative_accuracy": float(np.mean(predicted_negative[target_negative])) if np.any(target_negative) else 0.0,
        "predicted_negative_rate": float(np.mean(predicted_negative)),
        "mean_abs_score": float(np.mean(np.abs(score))),
    }
    rows: list[dict[str, float | str]] = []
    sequences = frame["sequence_id"].astype(str).to_numpy()
    for sequence in sorted(set(sequences)):
        mask = sequences == sequence
        local_target = target[mask]
        local_score = score[mask]
        neg = local_target < 0.0
        pos = ~neg
        rows.append({
            "sequence_id": sequence,
            "count": int(mask.sum()),
            "pearson": float(np.corrcoef(local_score, local_target)[0, 1]) if mask.sum() > 1 and np.std(local_score) > 1.0e-12 else 0.0,
            "positive_accuracy": float(np.mean(local_score[pos] >= 0.0)) if np.any(pos) else 0.0,
            "negative_accuracy": float(np.mean(local_score[neg] < 0.0)) if np.any(neg) else 0.0,
        })
    return result, pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v419-summary", type=Path, required=True)
    parser.add_argument("--v420-summary", type=Path, required=True)
    parser.add_argument("--v421-summary", type=Path, required=True)
    parser.add_argument("--v48-checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists; use --force")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    model_config, train_config, loss_config, decision_config = _load_config(args.config)
    device = _resolve_device(args.device)
    checkpoint_paths = _parse_checkpoints(args.v48_checkpoint)
    base_config, _, _, _, _ = _load_v48_config(args.v48_config)
    train_split, train_manifest = _materialize(args.cache_manifest, "train", input_size=base_config.input_size)
    validation_split, validation_manifest = _materialize(args.cache_manifest, "validation", input_size=base_config.input_size)
    train_frame = _align_ensemble(train_split, _read_ensemble(args.ensemble_train))
    validation_frame = _align_ensemble(validation_split, _read_ensemble(args.ensemble_validation))

    seed_records: list[dict[str, Any]] = []
    train_seed_scores: list[np.ndarray] = []
    validation_seed_scores: list[np.ndarray] = []
    train_seed_vertical_scores: list[np.ndarray] = []
    validation_seed_vertical_scores: list[np.ndarray] = []
    for ordinal, (seed, checkpoint) in enumerate(checkpoint_paths.items()):
        backbone, history, selected, drift = _train_seed(
            checkpoint, train_split, v48_config=args.v48_config,
            model_config=model_config, train_config=train_config, loss_config=loss_config,
            device=device, seed=train_config.seed + ordinal,
        )
        train_score, train_vertical, train_diag = _score_backbone(
            backbone, train_split, batch_size=train_config.batch_size, config=model_config, device=device
        )
        validation_score, validation_vertical, validation_diag = _score_backbone(
            backbone, validation_split, batch_size=train_config.batch_size, config=model_config, device=device
        )
        train_metrics, _ = _score_metrics(train_frame, train_score)
        validation_metrics, per_sequence = _score_metrics(validation_frame, validation_score)
        train_vertical_metrics, _ = _score_metrics(train_frame, train_vertical)
        validation_vertical_metrics, vertical_per_sequence = _score_metrics(validation_frame, validation_vertical)
        train_seed_scores.append(train_score)
        validation_seed_scores.append(validation_score)
        train_seed_vertical_scores.append(train_vertical)
        validation_seed_vertical_scores.append(validation_vertical)
        checkpoint_out = args.output_dir / f"adapted_seed_{seed}.pt"
        torch.save({
            "artifact_type": "object_event_v4_22_adapted_v48",
            "seed": seed,
            "model_state_dict": backbone.state_dict(),
            "selected_geometry_parameters": selected,
            "source_checkpoint": str(checkpoint),
        }, checkpoint_out)
        pd.DataFrame(history).to_csv(args.output_dir / f"training_history_seed_{seed}.csv", index=False)
        seed_records.append({
            "seed": seed,
            "source_checkpoint": str(checkpoint),
            "selected_geometry_parameters": selected,
            "final_training_loss": history[-1],
            "representation_drift": drift,
            "train_score_metrics": train_metrics,
            "validation_score_metrics": validation_metrics,
            "validation_per_sequence": per_sequence.to_dict(orient="records"),
            "train_vertical_scale_metrics": train_vertical_metrics,
            "validation_vertical_scale_metrics": validation_vertical_metrics,
            "validation_vertical_scale_per_sequence": vertical_per_sequence.to_dict(orient="records"),
            "train_diagnostics": train_diag,
            "validation_diagnostics": validation_diag,
        })
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_consensus_raw = np.median(np.stack(train_seed_scores, axis=0), axis=0)
    validation_consensus_raw = np.median(np.stack(validation_seed_scores, axis=0), axis=0)
    train_vertical_raw = np.median(np.stack(train_seed_vertical_scores, axis=0), axis=0)
    validation_vertical_raw = np.median(np.stack(validation_seed_vertical_scores, axis=0), axis=0)
    calibration = fit_score_calibration(
        train_consensus_raw,
        train_frame["target_expansion"].to_numpy(dtype=np.float64),
        minimum_scale=1.0e-4,
    )
    train_score = apply_score_calibration(train_consensus_raw, calibration)
    validation_score = apply_score_calibration(validation_consensus_raw, calibration)
    train_metrics, train_per_sequence = _score_metrics(train_frame, train_score)
    validation_metrics, validation_per_sequence = _score_metrics(validation_frame, validation_score)

    vertical_calibration = fit_score_calibration(
        train_vertical_raw,
        train_frame["target_expansion"].to_numpy(dtype=np.float64),
        minimum_scale=1.0e-4,
    )
    train_vertical_score = apply_score_calibration(train_vertical_raw, vertical_calibration)
    validation_vertical_score = apply_score_calibration(validation_vertical_raw, vertical_calibration)
    train_vertical_metrics, train_vertical_per_sequence = _score_metrics(train_frame, train_vertical_score)
    validation_vertical_metrics, validation_vertical_per_sequence = _score_metrics(validation_frame, validation_vertical_score)

    magnitude = np.abs(validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64))
    prediction = prediction_from_score(validation_score, magnitude)
    prediction_metrics, prediction_per_sequence = _metrics(validation_frame, prediction, minimum_negatives=20)
    baseline_prediction = validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    baseline_metrics, _ = _metrics(validation_frame, baseline_prediction, minimum_negatives=20)

    v419 = json.loads(args.v419_summary.read_text(encoding="utf-8"))
    v420 = json.loads(args.v420_summary.read_text(encoding="utf-8"))
    v421 = json.loads(args.v421_summary.read_text(encoding="utf-8"))
    raw_v419_validation = float(v419["decision"]["comparisons"]["best_validation_dense_score_abs_pearson"])
    v420_validation = float(v420["validation_score_metrics"]["pearson_to_target_expansion"])
    oracle_divergence_validation = float(v421["validation_oracle_divergence"]["pearson"])
    oracle_height_validation = float(
        next(row["oriented_pearson"] for row in v421["validation_feature_audit"] if row["feature"] == "box_log_height_ratio")
    )
    val_p = float(validation_metrics["pearson_to_target_expansion"])
    min_seq = float(validation_per_sequence["pearson"].min())
    vertical_p = float(validation_vertical_metrics["pearson_to_target_expansion"])
    vertical_min_seq = float(validation_vertical_per_sequence["pearson"].min())
    best_p = max(val_p, vertical_p)
    best_min_seq = max(min_seq, vertical_min_seq)
    seed_median = float(np.median([r["validation_score_metrics"]["pearson_to_target_expansion"] for r in seed_records]))
    if (
        best_p >= decision_config["strong_validation_pearson"]
        and best_min_seq >= decision_config["strong_min_sequence_pearson"]
    ):
        recommendation = "partial_unfreeze_supported_integrate_geometry_auxiliary_with_ttc"
    elif best_p >= decision_config["neutral_validation_pearson"] and best_min_seq >= decision_config["neutral_min_sequence_pearson"]:
        recommendation = "partial_unfreeze_neutral_integrate_auxiliary_with_stronger_height_constraint"
    else:
        recommendation = "partial_unfreeze_insufficient_keep_v419_representation_redesign_event_height_estimator"

    validation_output = validation_frame.loc[:, ["sequence_id", "sample_token", "track_id", "target_expansion"]].copy()
    validation_output["v422_divergence_score"] = validation_score
    validation_output["v422_vertical_scale_score"] = validation_vertical_score
    validation_output["v422_prediction_expansion"] = prediction
    validation_output.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    train_per_sequence.to_csv(args.output_dir / "train_per_sequence.csv", index=False)
    validation_per_sequence.to_csv(args.output_dir / "validation_per_sequence.csv", index=False)
    train_vertical_per_sequence.to_csv(args.output_dir / "train_vertical_scale_per_sequence.csv", index=False)
    validation_vertical_per_sequence.to_csv(args.output_dir / "validation_vertical_scale_per_sequence.csv", index=False)
    prediction_per_sequence.to_csv(args.output_dir / "validation_prediction_per_sequence.csv", index=False)

    summary = {
        "artifact_type": "object_event_v4_22_encoder_geometry",
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {"model": asdict(model_config), "train": asdict(train_config), "loss": asdict(loss_config), "decision": decision_config},
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "seed_records": seed_records,
        "train_only_score_calibration": asdict(calibration),
        "train_only_vertical_scale_calibration": asdict(vertical_calibration),
        "train_consensus_score_metrics": train_metrics,
        "validation_consensus_score_metrics": validation_metrics,
        "train_vertical_scale_metrics": train_vertical_metrics,
        "validation_vertical_scale_metrics": validation_vertical_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "encoder_divergence_sign_baseline_magnitude_validation_metrics": prediction_metrics,
        "decision": {
            "recommendation": recommendation,
            "comparisons": {
                "raw_v419_validation_divergence_pearson": raw_v419_validation,
                "v420_validation_divergence_pearson": v420_validation,
                "v422_validation_divergence_pearson": val_p,
                "v422_minimum_validation_sequence_pearson": min_seq,
                "v422_seed_median_validation_pearson": seed_median,
                "v422_vertical_scale_validation_pearson": vertical_p,
                "v422_vertical_scale_minimum_validation_sequence_pearson": vertical_min_seq,
                "v421_oracle_box_divergence_validation_pearson": oracle_divergence_validation,
                "v421_oracle_box_height_ratio_validation_pearson": oracle_height_validation,
            },
            "note": "Development-only representation decision; no validation selection and no TTC loss during encoder adaptation.",
        },
        "scientific_contract": {
            "three_true_seed_v48_backbones_adapted_independently": True,
            "only_geometry_encoder_tail_parameters_trainable": True,
            "no_trainable_posthoc_flow_or_sign_head": True,
            "boxes_are_train_only_pseudoflow_targets": True,
            "vertical_height_ratio_is_train_only_geometry_supervision": True,
            "ttc_labels_not_used_in_encoder_loss": True,
            "fixed_epoch_schedule_no_validation_selection": True,
            "validation_boxes_not_used_for_optimisation": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
