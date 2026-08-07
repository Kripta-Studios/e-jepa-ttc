#!/usr/bin/env python3
"""Successive-halving train-only schedule search for joint geometry + TTC."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
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
from e_jepa_ttc.models.object_event_v4_23 import configure_joint_geometry_ttc_unfreeze, named_trainable_parameters  # noqa: E402
from e_jepa_ttc.training.object_event_v4_8 import object_event_v4_8_loss  # noqa: E402
from e_jepa_ttc.training.object_event_v4_20 import box_affine_pseudoflow  # noqa: E402
from e_jepa_ttc.training.object_event_v4_22 import ObjectEventV422LossConfig, encoder_pseudoflow_loss, relative_parameter_anchor  # noqa: E402
from e_jepa_ttc.training.object_event_v4_24 import CandidateMetrics, SelectionConfig, rank_candidates  # noqa: E402
from e_jepa_ttc.training.object_event_v4_19 import apply_score_calibration, fit_score_calibration  # noqa: E402


@dataclass(frozen=True)
class OrchestratorConfig:
    fold_count: int = 3
    stage1_seed: int = 13
    stage1_keep: int = 3
    stage2_seeds: tuple[int, ...] = (7, 23)
    final_seeds: tuple[int, ...] = (7, 13, 23)
    batch_size: int = 8
    final_extra_epochs: int = 2
    max_grad_norm: float = 1.0
    last_geometry_parameter_tensors: int = 8
    seed: int = 2424


@dataclass(frozen=True)
class ArmConfig:
    epochs: int
    train_geometry: bool
    train_motion: bool
    geometry_learning_rate: float
    motion_learning_rate: float
    weight_decay: float
    ttc_weight: float
    geometry_weight: float
    anchor_weight: float


@dataclass
class CVResult:
    arm: str
    seed: int
    ttc_prediction: np.ndarray
    divergence: np.ndarray
    vertical: np.ndarray
    metrics: dict[str, Any]
    fold_records: list[dict[str, Any]]


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    data = dict(values)
    if cls is OrchestratorConfig:
        if "stage2_seeds" in data:
            data["stage2_seeds"] = tuple(int(x) for x in data["stage2_seeds"])
        if "final_seeds" in data:
            data["final_seeds"] = tuple(int(x) for x in data["final_seeds"])
    return cls(**data)


def _load_config(path: Path) -> tuple[Any, OrchestratorConfig, dict[str, ArmConfig], SelectionConfig, ObjectEventV422LossConfig, dict[str, float]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.24 config must be a mapping")
    from e_jepa_ttc.models.object_event_v4_22 import ObjectEventV422Config
    arms = {str(name): _construct(ArmConfig, cast(Mapping[str, Any], values)) for name, values in dict(raw["arms"]).items()}
    if len(arms) < 3:
        raise ValueError("v4.24 requires at least three candidate arms")
    return (
        _construct(ObjectEventV422Config, cast(Mapping[str, Any], raw.get("model", {}))),
        _construct(OrchestratorConfig, cast(Mapping[str, Any], raw.get("orchestrator", {}))),
        arms,
        _construct(SelectionConfig, cast(Mapping[str, Any], raw.get("selection", {}))),
        ObjectEventV422LossConfig(encoder_anchor_weight=0.0),
        {str(k): float(v) for k, v in dict(raw.get("final_decision", {})).items()},
    )


def _parse_checkpoints(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        result[int(seed_text)] = Path(path_text)
    if sorted(result) != [7, 13, 23]:
        raise ValueError("v4.24 requires exact seeds 7,13,23")
    return dict(sorted(result.items()))


def _sequence_folds(sequence_ids: np.ndarray, fold_count: int, seed: int) -> list[np.ndarray]:
    unique = np.array(sorted(set(str(x) for x in sequence_ids)), dtype=object)
    if len(unique) < fold_count:
        raise ValueError("not enough sequences for grouped folds")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    groups = [set(unique[index::fold_count].tolist()) for index in range(fold_count)]
    seq = np.asarray([str(x) for x in sequence_ids], dtype=object)
    return [np.flatnonzero(np.isin(seq, list(group))) for group in groups]


def _subset_split(split: Any, indices: np.ndarray) -> Any:
    if not hasattr(split, "subset"):
        raise TypeError("materialized split lacks subset()")
    return split.subset(torch.as_tensor(indices, dtype=torch.long))


def _configure_arm(backbone: Any, arm: ArmConfig, tail_count: int) -> dict[str, list[str]]:
    selected = configure_joint_geometry_ttc_unfreeze(backbone, tail_count)
    geometry = backbone.foreground_model.geometry_encoder
    if not arm.train_geometry:
        geometry.requires_grad_(False)
        selected["geometry"] = []
        try:
            backbone.motion_config = replace(backbone.motion_config, freeze_foreground=True)
        except TypeError:
            pass
    if not arm.train_motion:
        backbone.temporal_projection.requires_grad_(False)
        backbone.field_head.requires_grad_(False)
        selected["temporal_projection"] = []
        selected["field_head"] = []
    if not named_trainable_parameters(backbone):
        raise ValueError("candidate arm leaves no trainable parameters")
    return selected


def _optimizer(backbone: Any, arm: ArmConfig) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    geometry = [p for p in backbone.foreground_model.geometry_encoder.parameters() if p.requires_grad]
    motion = [p for module in (backbone.temporal_projection, backbone.field_head) for p in module.parameters() if p.requires_grad]
    if geometry:
        groups.append({"params": geometry, "lr": arm.geometry_learning_rate})
    if motion:
        groups.append({"params": motion, "lr": arm.motion_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=arm.weight_decay)


def _train_arm(
    checkpoint: Path,
    split: Any,
    *,
    v48_config: Path,
    model_config: Any,
    arm: ArmConfig,
    geometry_loss_config: ObjectEventV422LossConfig,
    batch_size: int,
    max_grad_norm: float,
    tail_count: int,
    device: torch.device,
    seed: int,
    epochs_override: int | None = None,
) -> tuple[Any, list[dict[str, float]], dict[str, list[str]], float]:
    torch.manual_seed(seed)
    backbone, _ = _load_backbone(v48_config_path=v48_config, checkpoint_path=checkpoint)
    backbone = backbone.to(device)
    selected = _configure_arm(backbone, arm, tail_count)
    initial = {name: p.detach().float().cpu().clone() for name, p in named_trainable_parameters(backbone).items()}
    _, _, _, _, v48_loss_config = _load_v48_config(v48_config)
    optimizer = _optimizer(backbone, arm)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    epochs = int(epochs_override if epochs_override is not None else arm.epochs)
    for epoch in range(1, epochs + 1):
        order = np.arange(len(split.events), dtype=np.int64)
        rng.shuffle(order)
        accum: dict[str, list[float]] = {key: [] for key in ("loss", "ttc", "geometry", "anchor", "ttc_sign", "ttc_correlation")}
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
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
            zero = output.expansion.sum() * 0.0
            geometry_loss = zero
            if arm.geometry_weight > 0.0 and arm.train_geometry:
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
                geometry_initial = {name: initial[name] for name in trainables if name in initial}
                geometry_loss, _ = encoder_pseudoflow_loss(
                    forward, reverse, target_f, mask_f, target_r, mask_r,
                    trainables, geometry_initial, config=geometry_loss_config,
                )
            trainables = named_trainable_parameters(backbone)
            anchor = relative_parameter_anchor(trainables, initial, epsilon=1.0e-6)
            loss = arm.ttc_weight * ttc.total + arm.geometry_weight * geometry_loss + arm.anchor_weight * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainables.values()), max_grad_norm)
            optimizer.step()
            accum["loss"].append(float(loss.detach()))
            accum["ttc"].append(float(ttc.total.detach()))
            accum["geometry"].append(float(geometry_loss.detach()))
            accum["anchor"].append(float(anchor.detach()))
            accum["ttc_sign"].append(float(ttc.components["sign"].detach()))
            accum["ttc_correlation"].append(float(ttc.components["correlation"].detach()))
        history.append({"epoch": float(epoch), **{k: float(np.mean(v)) for k, v in accum.items()}})
    return backbone, history, selected, float(history[-1]["anchor"])


@torch.no_grad()
def _predict(backbone: Any, split: Any, *, batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    backbone.eval()
    for start in range(0, len(split.events), batch_size):
        events = split.events[start:start + batch_size].to(device=device, dtype=torch.float32)
        chunks.append(backbone(events).expansion.detach().float().cpu())
    return torch.cat(chunks).numpy().astype(np.float64)


def _candidate_metrics(frame: pd.DataFrame, prediction: np.ndarray, divergence: np.ndarray, vertical: np.ndarray) -> tuple[dict[str, Any], CandidateMetrics]:
    ttc, _ = _metrics(frame, prediction, minimum_negatives=20)
    div, _ = _score_metrics(frame, divergence)
    vert, _ = _score_metrics(frame, vertical)
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    geometry = max(float(div["pearson_to_target_expansion"]), float(vert["pearson_to_target_expansion"]))
    packed = CandidateMetrics(
        pearson=float(ttc["pearson"]),
        positive_accuracy=float(ttc["positive_accuracy"]),
        negative_accuracy=float(ttc["negative_accuracy"]),
        balanced_sign_accuracy=float(ttc["balanced_sign_accuracy"]),
        minimum_sequence_pearson=float(ttc["minimum_sequence_pearson"]),
        minimum_sequence_negative_accuracy=float(ttc["minimum_sequence_negative_accuracy"]),
        predicted_negative_rate=float(np.mean(prediction < 0.0)),
        true_negative_rate=float(np.mean(target < 0.0)),
        geometry_pearson=geometry,
    )
    return {"ttc": ttc, "divergence": div, "vertical_scale": vert, "candidate": asdict(packed)}, packed


def _evaluate_arm_cv(
    arm_name: str,
    arm: ArmConfig,
    seed: int,
    *,
    checkpoint: Path,
    split: Any,
    frame: pd.DataFrame,
    folds: list[np.ndarray],
    v48_config: Path,
    model_config: Any,
    geometry_loss_config: ObjectEventV422LossConfig,
    orch: OrchestratorConfig,
    device: torch.device,
    output_dir: Path,
) -> CVResult:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    divergence = np.full(len(frame), np.nan, dtype=np.float64)
    vertical = np.full(len(frame), np.nan, dtype=np.float64)
    records: list[dict[str, Any]] = []
    all_idx = np.arange(len(frame), dtype=np.int64)
    for fold_id, held_idx in enumerate(folds):
        print(f"[v4.24] arm={arm_name} seed={seed} fold={fold_id + 1}/{len(folds)} training", flush=True)
        train_idx = np.setdiff1d(all_idx, held_idx, assume_unique=True)
        train_split = _subset_split(split, train_idx)
        held_split = _subset_split(split, held_idx)
        backbone, history, selected, drift = _train_arm(
            checkpoint, train_split, v48_config=v48_config, model_config=model_config, arm=arm,
            geometry_loss_config=geometry_loss_config, batch_size=orch.batch_size,
            max_grad_norm=orch.max_grad_norm, tail_count=orch.last_geometry_parameter_tensors,
            device=device, seed=orch.seed + seed * 100 + fold_id,
        )
        prediction[held_idx] = _predict(backbone, held_split, batch_size=orch.batch_size, device=device)
        div_raw, vert_raw, diagnostics = _score_backbone(backbone, held_split, batch_size=orch.batch_size, config=model_config, device=device)
        divergence[held_idx] = div_raw
        vertical[held_idx] = vert_raw
        held_frame = frame.iloc[held_idx].reset_index(drop=True)
        local, _ = _candidate_metrics(held_frame, prediction[held_idx], div_raw, vert_raw)
        print(
            f"[v4.24] arm={arm_name} seed={seed} fold={fold_id + 1} "
            f"pearson={local['ttc']['pearson']:.4f} neg={local['ttc']['negative_accuracy']:.4f} "
            f"geom={max(local['divergence']['pearson_to_target_expansion'], local['vertical_scale']['pearson_to_target_expansion']):.4f}",
            flush=True,
        )
        records.append({
            "arm": arm_name, "seed": seed, "fold": fold_id,
            "held_out_sequences": sorted(held_frame["sequence_id"].astype(str).unique().tolist()),
            "final_loss": history[-1], "selected_trainable_parameters": selected,
            "relative_drift": drift, "metrics": local, "diagnostics": diagnostics,
        })
        hist_dir = output_dir / "cv_histories"
        hist_dir.mkdir(exist_ok=True)
        pd.DataFrame(history).to_csv(hist_dir / f"{arm_name}_seed{seed}_fold{fold_id}.csv", index=False)
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not (np.isfinite(prediction).all() and np.isfinite(divergence).all() and np.isfinite(vertical).all()):
        raise RuntimeError(f"OOF coverage incomplete for {arm_name} seed {seed}")
    metrics, _ = _candidate_metrics(frame, prediction, divergence, vertical)
    return CVResult(arm_name, seed, prediction, divergence, vertical, metrics, records)


def _ranking_frame(results: dict[str, CVResult], selection: SelectionConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    candidates = {name: CandidateMetrics(**cast(dict[str, Any], result.metrics["candidate"])) for name, result in results.items()}
    rows = rank_candidates(candidates, selection)
    return pd.DataFrame(rows), rows


def _combine_seed_results(arm: str, per_seed: list[CVResult], frame: pd.DataFrame) -> CVResult:
    pred = np.median(np.stack([r.ttc_prediction for r in per_seed], axis=0), axis=0)
    div = np.median(np.stack([r.divergence for r in per_seed], axis=0), axis=0)
    vert = np.median(np.stack([r.vertical for r in per_seed], axis=0), axis=0)
    metrics, _ = _candidate_metrics(frame, pred, div, vert)
    records = [record for result in per_seed for record in result.fold_records]
    return CVResult(arm, -1, pred, div, vert, metrics, records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v422-summary", type=Path, required=True)
    parser.add_argument("--v423-summary", type=Path, required=True)
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

    model_config, orch, arms, selection, geometry_loss_config, final_decision = _load_config(args.config)
    device = _resolve_device(args.device)
    checkpoints = _parse_checkpoints(args.adapted_checkpoint)
    base_config, _, _, _, _ = _load_v48_config(args.v48_config)
    train_split, train_manifest = _materialize(args.cache_manifest, "train", input_size=base_config.input_size)
    validation_split, validation_manifest = _materialize(args.cache_manifest, "validation", input_size=base_config.input_size)
    train_frame = _align_ensemble(train_split, _read_ensemble(args.ensemble_train))
    validation_frame = _align_ensemble(validation_split, _read_ensemble(args.ensemble_validation))
    folds = _sequence_folds(train_frame["sequence_id"].astype(str).to_numpy(), orch.fold_count, orch.seed)

    # Stage 1: every arm, one seed, grouped OOF only on train sequences.
    print(f"[v4.24] stage1: {len(arms)} arms x seed {orch.stage1_seed} x {orch.fold_count} grouped folds", flush=True)
    stage1: dict[str, CVResult] = {}
    fold_records: list[dict[str, Any]] = []
    for arm_name, arm in arms.items():
        result = _evaluate_arm_cv(
            arm_name, arm, orch.stage1_seed, checkpoint=checkpoints[orch.stage1_seed],
            split=train_split, frame=train_frame, folds=folds, v48_config=args.v48_config,
            model_config=model_config, geometry_loss_config=geometry_loss_config,
            orch=orch, device=device, output_dir=args.output_dir,
        )
        stage1[arm_name] = result
        fold_records.extend(result.fold_records)
    stage1_frame, stage1_rows = _ranking_frame(stage1, selection)
    stage1_frame.to_csv(args.output_dir / "stage1_arm_ranking.csv", index=False)
    keep = [str(row["arm"]) for row in stage1_rows[: min(orch.stage1_keep, len(stage1_rows))]]
    print(f"[v4.24] stage1 survivors: {keep}", flush=True)

    # Stage 2: confirm the survivors with the two remaining independent seeds.
    print(f"[v4.24] stage2: confirming survivors with seeds {list(orch.stage2_seeds)}", flush=True)
    all_seed_results: dict[str, list[CVResult]] = {name: [stage1[name]] for name in keep}
    for arm_name in keep:
        arm = arms[arm_name]
        for seed in orch.stage2_seeds:
            result = _evaluate_arm_cv(
                arm_name, arm, seed, checkpoint=checkpoints[seed], split=train_split, frame=train_frame,
                folds=folds, v48_config=args.v48_config, model_config=model_config,
                geometry_loss_config=geometry_loss_config, orch=orch, device=device, output_dir=args.output_dir,
            )
            all_seed_results[arm_name].append(result)
            fold_records.extend(result.fold_records)
    combined = {name: _combine_seed_results(name, results, train_frame) for name, results in all_seed_results.items()}
    stage2_frame, stage2_rows = _ranking_frame(combined, selection)
    stage2_frame.to_csv(args.output_dir / "stage2_arm_ranking.csv", index=False)
    champion = str(stage2_rows[0]["arm"])
    champion_reason = "best_eligible_train_only_multiseed_oof" if bool(stage2_rows[0]["eligible"]) else "best_available_train_only_multiseed_oof_fallback"
    print(f"[v4.24] champion={champion} reason={champion_reason}", flush=True)
    pd.DataFrame([
        {
            "arm": rec["arm"], "seed": rec["seed"], "fold": rec["fold"],
            "held_out_sequences": ";".join(rec["held_out_sequences"]),
            "pearson": rec["metrics"]["ttc"]["pearson"],
            "positive_accuracy": rec["metrics"]["ttc"]["positive_accuracy"],
            "negative_accuracy": rec["metrics"]["ttc"]["negative_accuracy"],
            "minimum_sequence_pearson": rec["metrics"]["ttc"]["minimum_sequence_pearson"],
            "geometry_pearson": max(rec["metrics"]["divergence"]["pearson_to_target_expansion"], rec["metrics"]["vertical_scale"]["pearson_to_target_expansion"]),
            "relative_drift": rec["relative_drift"],
        }
        for rec in fold_records
    ]).to_csv(args.output_dir / "cv_fold_records.csv", index=False)

    # Final: only the champion sees full train; development validation is evaluated once afterwards.
    champion_arm = arms[champion]
    final_epochs = champion_arm.epochs + orch.final_extra_epochs
    train_predictions: list[np.ndarray] = []
    validation_predictions: list[np.ndarray] = []
    train_divergence: list[np.ndarray] = []
    validation_divergence: list[np.ndarray] = []
    train_vertical: list[np.ndarray] = []
    validation_vertical: list[np.ndarray] = []
    final_records: list[dict[str, Any]] = []
    print(f"[v4.24] final: champion full-train seeds={list(orch.final_seeds)} epochs={final_epochs}", flush=True)
    for ordinal, seed in enumerate(orch.final_seeds):
        print(f"[v4.24] final champion={champion} seed={seed} training", flush=True)
        backbone, history, selected, drift = _train_arm(
            checkpoints[seed], train_split, v48_config=args.v48_config, model_config=model_config,
            arm=champion_arm, geometry_loss_config=geometry_loss_config, batch_size=orch.batch_size,
            max_grad_norm=orch.max_grad_norm, tail_count=orch.last_geometry_parameter_tensors,
            device=device, seed=orch.seed + 9000 + ordinal, epochs_override=final_epochs,
        )
        tr_pred = _predict(backbone, train_split, batch_size=orch.batch_size, device=device)
        va_pred = _predict(backbone, validation_split, batch_size=orch.batch_size, device=device)
        tr_div, tr_vert, tr_diag = _score_backbone(backbone, train_split, batch_size=orch.batch_size, config=model_config, device=device)
        va_div, va_vert, va_diag = _score_backbone(backbone, validation_split, batch_size=orch.batch_size, config=model_config, device=device)
        train_predictions.append(tr_pred); validation_predictions.append(va_pred)
        train_divergence.append(tr_div); validation_divergence.append(va_div)
        train_vertical.append(tr_vert); validation_vertical.append(va_vert)
        tr_metrics, _ = _metrics(train_frame, tr_pred, minimum_negatives=20)
        va_metrics, va_per_seq = _metrics(validation_frame, va_pred, minimum_negatives=20)
        checkpoint_out = args.output_dir / f"champion_{champion}_seed_{seed}.pt"
        torch.save({
            "artifact_type": "object_event_v4_24_orchestrated_champion",
            "arm": champion, "seed": seed, "model_state_dict": backbone.state_dict(),
            "source_v422_checkpoint": str(checkpoints[seed]), "selected_trainable_parameters": selected,
        }, checkpoint_out)
        pd.DataFrame(history).to_csv(args.output_dir / f"final_history_{champion}_seed_{seed}.csv", index=False)
        final_records.append({
            "seed": seed, "relative_drift": drift, "train_ttc_metrics": tr_metrics,
            "validation_ttc_metrics": va_metrics, "validation_ttc_per_sequence": va_per_seq.to_dict(orient="records"),
            "train_diagnostics": tr_diag, "validation_diagnostics": va_diag,
        })
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_consensus = np.median(np.stack(train_predictions, axis=0), axis=0)
    validation_consensus = np.median(np.stack(validation_predictions, axis=0), axis=0)
    train_metrics, _ = _metrics(train_frame, train_consensus, minimum_negatives=20)
    validation_metrics, validation_per_sequence = _metrics(validation_frame, validation_consensus, minimum_negatives=20)

    tr_div_raw = np.median(np.stack(train_divergence, axis=0), axis=0)
    va_div_raw = np.median(np.stack(validation_divergence, axis=0), axis=0)
    div_cal = fit_score_calibration(tr_div_raw, train_frame["target_expansion"].to_numpy(dtype=np.float64), minimum_scale=1.0e-4)
    tr_div = apply_score_calibration(tr_div_raw, div_cal); va_div = apply_score_calibration(va_div_raw, div_cal)
    tr_div_metrics, _ = _score_metrics(train_frame, tr_div); va_div_metrics, va_div_per_seq = _score_metrics(validation_frame, va_div)

    tr_vert_raw = np.median(np.stack(train_vertical, axis=0), axis=0)
    va_vert_raw = np.median(np.stack(validation_vertical, axis=0), axis=0)
    vert_cal = fit_score_calibration(tr_vert_raw, train_frame["target_expansion"].to_numpy(dtype=np.float64), minimum_scale=1.0e-4)
    tr_vert = apply_score_calibration(tr_vert_raw, vert_cal); va_vert = apply_score_calibration(va_vert_raw, vert_cal)
    tr_vert_metrics, _ = _score_metrics(train_frame, tr_vert); va_vert_metrics, va_vert_per_seq = _score_metrics(validation_frame, va_vert)

    baseline = validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    baseline_metrics, _ = _metrics(validation_frame, baseline, minimum_negatives=20)
    v423 = json.loads(args.v423_summary.read_text(encoding="utf-8"))
    geometry_best = max(float(va_div_metrics["pearson_to_target_expansion"]), float(va_vert_metrics["pearson_to_target_expansion"]))
    if (
        float(validation_metrics["pearson"]) >= float(baseline_metrics["pearson"]) - final_decision["baseline_pearson_tolerance"]
        and float(validation_metrics["negative_accuracy"]) >= float(baseline_metrics["negative_accuracy"]) - final_decision["baseline_negative_accuracy_tolerance"]
        and float(validation_metrics["minimum_sequence_negative_accuracy"]) >= final_decision["minimum_sequence_negative_accuracy"]
        and geometry_best >= final_decision["geometry_retention_floor"]
    ):
        recommendation = "orchestrated_schedule_supported_lock_and_run_long_multiseed"
    elif float(validation_metrics["pearson"]) > float(v423["validation_consensus_ttc_metrics"]["pearson"]):
        recommendation = "orchestrator_improves_v423_but_not_baseline_refine_geometry_conditioned_readout"
    else:
        recommendation = "schedule_search_exhausted_keep_v422_geometry_redesign_ttc_readout"

    out = validation_frame.loc[:, ["sequence_id", "sample_token", "track_id", "target_expansion"]].copy()
    out["baseline_prediction_expansion"] = baseline
    out["v424_prediction_expansion"] = validation_consensus
    out["v424_divergence_score"] = va_div
    out["v424_vertical_scale_score"] = va_vert
    out.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    validation_per_sequence.to_csv(args.output_dir / "validation_ttc_per_sequence.csv", index=False)
    va_div_per_seq.to_csv(args.output_dir / "validation_divergence_per_sequence.csv", index=False)
    va_vert_per_seq.to_csv(args.output_dir / "validation_vertical_scale_per_sequence.csv", index=False)

    summary = {
        "artifact_type": "object_event_v4_24_train_only_orchestrator",
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {
            "model": asdict(model_config), "orchestrator": asdict(orch),
            "arms": {name: asdict(cfg) for name, cfg in arms.items()},
            "selection": asdict(selection), "final_decision": final_decision,
        },
        "train_manifest": train_manifest, "validation_manifest": validation_manifest,
        "stage1_ranking": stage1_rows,
        "stage1_survivors": keep,
        "stage2_ranking": stage2_rows,
        "champion": champion,
        "champion_selection_reason": champion_reason,
        "champion_train_only_oof_metrics": combined[champion].metrics,
        "final_seed_records": final_records,
        "train_consensus_ttc_metrics": train_metrics,
        "validation_consensus_ttc_metrics": validation_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "train_divergence_metrics": tr_div_metrics, "validation_divergence_metrics": va_div_metrics,
        "train_vertical_scale_metrics": tr_vert_metrics, "validation_vertical_scale_metrics": va_vert_metrics,
        "decision": {
            "recommendation": recommendation,
            "comparisons": {
                "baseline_validation_pearson": float(baseline_metrics["pearson"]),
                "v423_validation_pearson": float(v423["validation_consensus_ttc_metrics"]["pearson"]),
                "v424_validation_pearson": float(validation_metrics["pearson"]),
                "baseline_negative_accuracy": float(baseline_metrics["negative_accuracy"]),
                "v423_negative_accuracy": float(v423["validation_consensus_ttc_metrics"]["negative_accuracy"]),
                "v424_negative_accuracy": float(validation_metrics["negative_accuracy"]),
                "v424_minimum_sequence_negative_accuracy": float(validation_metrics["minimum_sequence_negative_accuracy"]),
                "v424_divergence_validation_pearson": float(va_div_metrics["pearson_to_target_expansion"]),
                "v424_vertical_scale_validation_pearson": float(va_vert_metrics["pearson_to_target_expansion"]),
            },
        },
        "scientific_contract": {
            "five_candidate_schedules_tested": len(arms) == 5,
            "stage1_selection_train_grouped_oof_only": True,
            "stage2_multiseed_confirmation_train_grouped_oof_only": True,
            "only_one_champion_evaluated_on_development_validation": True,
            "no_validation_epoch_arm_or_seed_selection": True,
            "starts_from_v422_geometry_checkpoints": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe({
        "status": summary["status"], "champion": champion,
        "champion_selection_reason": champion_reason,
        "validation_consensus_ttc_metrics": validation_metrics,
        "decision": summary["decision"],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
