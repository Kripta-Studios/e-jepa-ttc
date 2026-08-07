#!/usr/bin/env python3
"""Jointly fine-tune v4.22 geometry and the existing v4.8 TTC/LHR head."""
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
from scripts.analyze_object_event_v4_22_encoder_geometry import _flows_from_backbone, _score_backbone, _score_metrics  # noqa: E402
from e_jepa_ttc.models.object_event_v4_23 import (  # noqa: E402
    configure_joint_geometry_ttc_unfreeze,
    geometry_parameters,
    motion_parameters,
    named_trainable_parameters,
)
from e_jepa_ttc.training.object_event_v4_8 import object_event_v4_8_loss  # noqa: E402
from e_jepa_ttc.training.object_event_v4_20 import box_affine_pseudoflow  # noqa: E402
from e_jepa_ttc.training.object_event_v4_22 import (  # noqa: E402
    ObjectEventV422LossConfig,
    encoder_pseudoflow_loss,
    relative_parameter_anchor,
)
from e_jepa_ttc.training.object_event_v4_23 import ObjectEventV423JointLossConfig, combine_joint_losses  # noqa: E402
from e_jepa_ttc.training.object_event_v4_19 import apply_score_calibration, fit_score_calibration  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2323
    batch_size: int = 8
    epochs: int = 6
    geometry_learning_rate: float = 1.0e-5
    motion_learning_rate: float = 5.0e-5
    weight_decay: float = 1.0e-5
    max_grad_norm: float = 1.0
    last_geometry_parameter_tensors: int = 8


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_config(path: Path) -> tuple[Any, TrainConfig, ObjectEventV423JointLossConfig, ObjectEventV422LossConfig, dict[str, float]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.23 config must be a mapping")
    from e_jepa_ttc.models.object_event_v4_22 import ObjectEventV422Config
    return (
        _construct(ObjectEventV422Config, cast(Mapping[str, Any], raw.get("model", {}))),
        _construct(TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))),
        _construct(ObjectEventV423JointLossConfig, cast(Mapping[str, Any], raw.get("joint_loss", {}))),
        _construct(ObjectEventV422LossConfig, cast(Mapping[str, Any], raw.get("geometry_loss", {}))),
        {str(k): float(v) for k, v in dict(raw.get("decision", {})).items()},
    )


def _parse_checkpoints(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        result[int(seed_text)] = Path(path_text)
    if sorted(result) != [7, 13, 23]:
        raise ValueError("v4.23 requires exact seeds 7,13,23")
    return dict(sorted(result.items()))


def _initial_trainables(backbone: Any) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().float().cpu().clone() for name, parameter in named_trainable_parameters(backbone).items()}


def _train_seed(
    checkpoint: Path,
    split: Any,
    *,
    v48_config: Path,
    model_config: Any,
    train_config: TrainConfig,
    joint_config: ObjectEventV423JointLossConfig,
    geometry_loss_config: ObjectEventV422LossConfig,
    device: torch.device,
    seed: int,
) -> tuple[Any, list[dict[str, float]], dict[str, list[str]], dict[str, float]]:
    torch.manual_seed(seed)
    backbone, _ = _load_backbone(v48_config_path=v48_config, checkpoint_path=checkpoint)
    backbone = backbone.to(device)
    selected = configure_joint_geometry_ttc_unfreeze(backbone, train_config.last_geometry_parameter_tensors)
    initial = _initial_trainables(backbone)
    _, _, _, _, v48_loss_config = _load_v48_config(v48_config)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(geometry_parameters(backbone)), "lr": train_config.geometry_learning_rate},
            {"params": list(motion_parameters(backbone)), "lr": train_config.motion_learning_rate},
        ],
        weight_decay=train_config.weight_decay,
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    count = len(split.events)
    for epoch in range(1, train_config.epochs + 1):
        order = np.arange(count, dtype=np.int64)
        rng.shuffle(order)
        accum: dict[str, list[float]] = {key: [] for key in (
            "loss", "ttc_loss", "geometry_loss", "anchor_loss", "ttc_expansion", "ttc_correlation",
            "ttc_sign", "ttc_pooled_log_eta", "geometry_flow", "geometry_divergence", "geometry_vertical_scale",
        )}
        for start in range(0, count, train_config.batch_size):
            batch_idx = order[start:start + train_config.batch_size]
            idx = torch.as_tensor(batch_idx, dtype=torch.long)
            events = split.events[idx].to(device=device, dtype=torch.float32)
            delta_t = split.delta_t_s[idx].to(device=device, dtype=torch.float32)
            target_ttc = split.target_ttc_s[idx].to(device=device, dtype=torch.float32)
            heights = split.visible_heights_px[idx].to(device=device, dtype=torch.float32)
            boxes = split.boxes_xyxy[idx].to(device=device, dtype=torch.float32)

            output = backbone(events)
            ttc = object_event_v4_8_loss(
                output, delta_t, target_ttc, heights, boxes,
                source_height=split.source_height, source_width=split.source_width,
                config=v48_loss_config,
            )
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
            trainables = named_trainable_parameters(backbone)
            zero_anchor_initial = {name: initial[name] for name in trainables}
            geometry_loss, geometry_parts = encoder_pseudoflow_loss(
                forward, reverse, target_f, mask_f, target_r, mask_r,
                trainables, zero_anchor_initial, config=geometry_loss_config,
            )
            anchor = relative_parameter_anchor(trainables, initial, epsilon=joint_config.epsilon)
            loss = combine_joint_losses(ttc.total, geometry_loss, anchor, config=joint_config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainables.values()), train_config.max_grad_norm)
            optimizer.step()

            accum["loss"].append(float(loss.detach()))
            accum["ttc_loss"].append(float(ttc.total.detach()))
            accum["geometry_loss"].append(float(geometry_loss.detach()))
            accum["anchor_loss"].append(float(anchor.detach()))
            accum["ttc_expansion"].append(float(ttc.components["expansion"].detach()))
            accum["ttc_correlation"].append(float(ttc.components["correlation"].detach()))
            accum["ttc_sign"].append(float(ttc.components["sign"].detach()))
            accum["ttc_pooled_log_eta"].append(float(ttc.components["pooled_log_eta"].detach()))
            accum["geometry_flow"].append(float(geometry_parts["flow"].detach()))
            accum["geometry_divergence"].append(float(geometry_parts["divergence"].detach()))
            accum["geometry_vertical_scale"].append(float(geometry_parts["vertical_scale"].detach()))
        row = {"epoch": float(epoch)}
        row.update({key: float(np.mean(values)) for key, values in accum.items()})
        history.append(row)
    return backbone, history, selected, {"relative_trainable_parameter_drift": float(history[-1]["anchor_loss"])}


@torch.no_grad()
def _predict_expansion(backbone: Any, split: Any, *, batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    backbone.eval()
    for start in range(0, len(split.events), batch_size):
        events = split.events[start:start + batch_size].to(device=device, dtype=torch.float32)
        chunks.append(backbone(events).expansion.detach().float().cpu())
    return torch.cat(chunks).numpy().astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v422-summary", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", action="append", required=True)
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

    model_config, train_config, joint_config, geometry_loss_config, decision_config = _load_config(args.config)
    device = _resolve_device(args.device)
    checkpoints = _parse_checkpoints(args.adapted_checkpoint)
    base_config, _, _, _, _ = _load_v48_config(args.v48_config)
    train_split, train_manifest = _materialize(args.cache_manifest, "train", input_size=base_config.input_size)
    validation_split, validation_manifest = _materialize(args.cache_manifest, "validation", input_size=base_config.input_size)
    train_frame = _align_ensemble(train_split, _read_ensemble(args.ensemble_train))
    validation_frame = _align_ensemble(validation_split, _read_ensemble(args.ensemble_validation))

    train_predictions: list[np.ndarray] = []
    validation_predictions: list[np.ndarray] = []
    train_divergence: list[np.ndarray] = []
    validation_divergence: list[np.ndarray] = []
    train_vertical: list[np.ndarray] = []
    validation_vertical: list[np.ndarray] = []
    seed_records: list[dict[str, Any]] = []

    for ordinal, (seed, checkpoint) in enumerate(checkpoints.items()):
        backbone, history, selected, drift = _train_seed(
            checkpoint, train_split, v48_config=args.v48_config, model_config=model_config,
            train_config=train_config, joint_config=joint_config, geometry_loss_config=geometry_loss_config,
            device=device, seed=train_config.seed + ordinal,
        )
        train_pred = _predict_expansion(backbone, train_split, batch_size=train_config.batch_size, device=device)
        val_pred = _predict_expansion(backbone, validation_split, batch_size=train_config.batch_size, device=device)
        tr_div, tr_vert, tr_diag = _score_backbone(backbone, train_split, batch_size=train_config.batch_size, config=model_config, device=device)
        va_div, va_vert, va_diag = _score_backbone(backbone, validation_split, batch_size=train_config.batch_size, config=model_config, device=device)
        train_predictions.append(train_pred)
        validation_predictions.append(val_pred)
        train_divergence.append(tr_div)
        validation_divergence.append(va_div)
        train_vertical.append(tr_vert)
        validation_vertical.append(va_vert)

        train_metrics, _ = _metrics(train_frame, train_pred, minimum_negatives=20)
        val_metrics, val_per_sequence = _metrics(validation_frame, val_pred, minimum_negatives=20)
        tr_div_metrics, _ = _score_metrics(train_frame, tr_div)
        va_div_metrics, _ = _score_metrics(validation_frame, va_div)
        tr_vert_metrics, _ = _score_metrics(train_frame, tr_vert)
        va_vert_metrics, _ = _score_metrics(validation_frame, va_vert)
        checkpoint_out = args.output_dir / f"joint_seed_{seed}.pt"
        torch.save({
            "artifact_type": "object_event_v4_23_joint_geometry_ttc",
            "seed": seed,
            "model_state_dict": backbone.state_dict(),
            "selected_trainable_parameters": selected,
            "source_v422_checkpoint": str(checkpoint),
        }, checkpoint_out)
        pd.DataFrame(history).to_csv(args.output_dir / f"training_history_seed_{seed}.csv", index=False)
        seed_records.append({
            "seed": seed,
            "source_v422_checkpoint": str(checkpoint),
            "selected_trainable_parameters": selected,
            "final_training_loss": history[-1],
            "representation_drift": drift,
            "train_ttc_metrics": train_metrics,
            "validation_ttc_metrics": val_metrics,
            "validation_ttc_per_sequence": val_per_sequence.to_dict(orient="records"),
            "train_divergence_metrics": tr_div_metrics,
            "validation_divergence_metrics": va_div_metrics,
            "train_vertical_scale_metrics": tr_vert_metrics,
            "validation_vertical_scale_metrics": va_vert_metrics,
            "train_diagnostics": tr_diag,
            "validation_diagnostics": va_diag,
        })
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_consensus = np.median(np.stack(train_predictions, axis=0), axis=0)
    validation_consensus = np.median(np.stack(validation_predictions, axis=0), axis=0)
    train_metrics, train_per_sequence = _metrics(train_frame, train_consensus, minimum_negatives=20)
    validation_metrics, validation_per_sequence = _metrics(validation_frame, validation_consensus, minimum_negatives=20)

    train_div_raw = np.median(np.stack(train_divergence, axis=0), axis=0)
    validation_div_raw = np.median(np.stack(validation_divergence, axis=0), axis=0)
    div_cal = fit_score_calibration(train_div_raw, train_frame["target_expansion"].to_numpy(dtype=np.float64), minimum_scale=1.0e-4)
    train_div_score = apply_score_calibration(train_div_raw, div_cal)
    validation_div_score = apply_score_calibration(validation_div_raw, div_cal)
    train_div_metrics, _ = _score_metrics(train_frame, train_div_score)
    validation_div_metrics, validation_div_per_sequence = _score_metrics(validation_frame, validation_div_score)

    train_vert_raw = np.median(np.stack(train_vertical, axis=0), axis=0)
    validation_vert_raw = np.median(np.stack(validation_vertical, axis=0), axis=0)
    vert_cal = fit_score_calibration(train_vert_raw, train_frame["target_expansion"].to_numpy(dtype=np.float64), minimum_scale=1.0e-4)
    train_vert_score = apply_score_calibration(train_vert_raw, vert_cal)
    validation_vert_score = apply_score_calibration(validation_vert_raw, vert_cal)
    train_vert_metrics, _ = _score_metrics(train_frame, train_vert_score)
    validation_vert_metrics, validation_vert_per_sequence = _score_metrics(validation_frame, validation_vert_score)

    baseline = validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    baseline_metrics, _ = _metrics(validation_frame, baseline, minimum_negatives=20)
    v422 = json.loads(args.v422_summary.read_text(encoding="utf-8"))
    baseline_p = float(baseline_metrics["pearson"])
    joint_p = float(validation_metrics["pearson"])
    baseline_neg = float(baseline_metrics["negative_accuracy"])
    joint_neg = float(validation_metrics["negative_accuracy"])
    joint_min_neg = float(validation_metrics["minimum_sequence_negative_accuracy"])
    geometry_best = max(
        float(validation_div_metrics["pearson_to_target_expansion"]),
        float(validation_vert_metrics["pearson_to_target_expansion"]),
    )
    if (
        joint_p >= baseline_p - decision_config["baseline_pearson_tolerance"]
        and joint_neg >= baseline_neg + decision_config["minimum_negative_accuracy_gain"]
        and joint_min_neg >= decision_config["minimum_sequence_negative_accuracy"]
        and geometry_best >= decision_config["geometry_retention_floor"]
    ):
        recommendation = "joint_geometry_ttc_supported_lock_architecture_run_longer_multiseed"
    elif joint_p >= baseline_p - 0.03 and geometry_best >= decision_config["geometry_retention_floor"]:
        recommendation = "joint_geometry_ttc_promising_keep_architecture_adjust_train_only_loss_schedule"
    else:
        recommendation = "joint_geometry_ttc_not_yet_supported_preserve_v422_geometry_redesign_ttc_readout"

    output = validation_frame.loc[:, ["sequence_id", "sample_token", "track_id", "target_expansion"]].copy()
    output["baseline_prediction_expansion"] = baseline
    output["v423_prediction_expansion"] = validation_consensus
    output["v423_divergence_score"] = validation_div_score
    output["v423_vertical_scale_score"] = validation_vert_score
    output.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    train_per_sequence.to_csv(args.output_dir / "train_ttc_per_sequence.csv", index=False)
    validation_per_sequence.to_csv(args.output_dir / "validation_ttc_per_sequence.csv", index=False)
    validation_div_per_sequence.to_csv(args.output_dir / "validation_divergence_per_sequence.csv", index=False)
    validation_vert_per_sequence.to_csv(args.output_dir / "validation_vertical_scale_per_sequence.csv", index=False)

    summary = {
        "artifact_type": "object_event_v4_23_joint_geometry_ttc",
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {
            "model": asdict(model_config), "train": asdict(train_config),
            "joint_loss": asdict(joint_config), "geometry_loss": asdict(geometry_loss_config),
            "decision": decision_config,
        },
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "seed_records": seed_records,
        "train_consensus_ttc_metrics": train_metrics,
        "validation_consensus_ttc_metrics": validation_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "train_divergence_metrics": train_div_metrics,
        "validation_divergence_metrics": validation_div_metrics,
        "train_vertical_scale_metrics": train_vert_metrics,
        "validation_vertical_scale_metrics": validation_vert_metrics,
        "decision": {
            "recommendation": recommendation,
            "comparisons": {
                "baseline_validation_pearson": baseline_p,
                "v423_validation_pearson": joint_p,
                "baseline_negative_accuracy": baseline_neg,
                "v423_negative_accuracy": joint_neg,
                "v423_minimum_sequence_negative_accuracy": joint_min_neg,
                "v422_divergence_validation_pearson": float(v422["validation_consensus_score_metrics"]["pearson_to_target_expansion"]),
                "v423_divergence_validation_pearson": float(validation_div_metrics["pearson_to_target_expansion"]),
                "v422_vertical_scale_validation_pearson": float(v422["validation_vertical_scale_metrics"]["pearson_to_target_expansion"]),
                "v423_vertical_scale_validation_pearson": float(validation_vert_metrics["pearson_to_target_expansion"]),
            },
            "note": "Development-only fixed-schedule joint fine-tuning; validation never selects epochs, seeds or weights.",
        },
        "scientific_contract": {
            "starts_from_three_independent_v422_adapted_seeds": True,
            "geometry_tail_and_existing_v48_motion_head_only_trainable": True,
            "no_new_posthoc_sign_or_ttc_router": True,
            "ttc_labels_used_on_train_only": True,
            "boxes_and_visible_heights_are_train_only_targets_not_forward_features": True,
            "dense_pseudoflow_divergence_vertical_scale_auxiliary_retained": True,
            "fixed_epoch_schedule_no_validation_selection": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
