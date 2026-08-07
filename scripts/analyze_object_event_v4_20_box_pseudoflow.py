#!/usr/bin/env python3
"""Train and evaluate a geometry-only box-pseudoflow refiner on frozen v4.8 correspondences."""
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
from scripts.train_e_jepa_object_event_v4_12 import _align_ensemble, _read_ensemble  # noqa: E402
from scripts.train_e_jepa_object_event_v4_16 import (  # noqa: E402
    _build_frozen_consensus,
    _json_safe,
    _metrics,
    _parse_checkpoints,
    _resolve_device,
)
from e_jepa_ttc.models.object_event_v4_19 import dense_flow_scores, local_correlation_flow  # noqa: E402
from e_jepa_ttc.models.object_event_v4_20 import BoxPseudoFlowRefiner, ObjectEventV420Config  # noqa: E402
from e_jepa_ttc.training.object_event_v4_20 import (  # noqa: E402
    ObjectEventV420LossConfig,
    box_affine_pseudoflow,
    pseudoflow_loss,
)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2020
    descriptor_batch_size: int = 6
    head_batch_size: int = 32
    fold_count: int = 3
    fold_epochs: int = 20
    final_epochs: int = 20
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 2.0


@dataclass
class DenseInputs:
    forward: torch.Tensor
    reverse: torch.Tensor
    diagnostics: dict[str, float]

    def subset(self, indices: np.ndarray | torch.Tensor) -> "DenseInputs":
        idx = torch.as_tensor(indices, dtype=torch.long)
        return DenseInputs(self.forward[idx], self.reverse[idx], dict(self.diagnostics))


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_config(path: Path) -> tuple[ObjectEventV420Config, TrainConfig, ObjectEventV420LossConfig, dict[str, float], dict[str, float]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.20 config must be a mapping")
    model_raw = dict(raw.get("model", {}))
    extraction = {
        "search_radius": float(model_raw.pop("search_radius", 4)),
        "correlation_temperature": float(model_raw.pop("correlation_temperature", 0.07)),
        "foreground_floor": float(model_raw.pop("foreground_floor", 0.05)),
        "confidence_floor": float(model_raw.pop("confidence_floor", 0.05)),
    }
    return (
        _construct(ObjectEventV420Config, cast(Mapping[str, Any], model_raw)),
        _construct(TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))),
        _construct(ObjectEventV420LossConfig, cast(Mapping[str, Any], raw.get("loss", {}))),
        extraction,
        {str(k): float(v) for k, v in dict(raw.get("decision", {})).items()},
    )


@torch.no_grad()
def _extract_dense_inputs(
    frozen: Any,
    events: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    search_radius: int,
    correlation_temperature: float,
) -> DenseInputs:
    forward_rows: list[torch.Tensor] = []
    reverse_rows: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    disagreements: list[torch.Tensor] = []
    map_shape: tuple[int, int] | None = None
    frozen.eval()
    for start in range(0, len(events), batch_size):
        end = min(start + batch_size, len(events))
        batch = events[start:end].to(device=device, dtype=torch.float32)
        seed_forward: list[torch.Tensor] = []
        seed_reverse: list[torch.Tensor] = []
        seed_conf_f: list[torch.Tensor] = []
        seed_conf_r: list[torch.Tensor] = []
        seed_fg: list[torch.Tensor] = []
        for extractor in frozen.extractors:
            maps, _, foreground, _ = extractor.backbone._foreground_and_features(batch)
            first, second = maps[:, 1], maps[:, 2]
            target_size = first.shape[-2:]
            if map_shape is None:
                map_shape = (int(target_size[0]), int(target_size[1]))
            fg1 = torch.nn.functional.interpolate(
                foreground[:, 1:2].float(), size=target_size, mode="bilinear", align_corners=False
            )[:, 0]
            fg2 = torch.nn.functional.interpolate(
                foreground[:, 2:3].float(), size=target_size, mode="bilinear", align_corners=False
            )[:, 0]
            fg = torch.sqrt((fg1 * fg2).clamp_min(0.0))
            fx, fy, cf = local_correlation_flow(
                first, second, radius=search_radius, temperature=correlation_temperature
            )
            rx, ry, cr = local_correlation_flow(
                second, first, radius=search_radius, temperature=correlation_temperature
            )
            seed_forward.append(torch.stack((fx, fy), dim=1))
            seed_reverse.append(torch.stack((rx, ry), dim=1))
            seed_conf_f.append(cf)
            seed_conf_r.append(cr)
            seed_fg.append(fg)

        flow_f = torch.stack(seed_forward, dim=0)
        flow_r = torch.stack(seed_reverse, dim=0)
        med_f = flow_f.median(dim=0).values
        med_r = flow_r.median(dim=0).values
        conf_f = torch.stack(seed_conf_f, dim=0).median(dim=0).values
        conf_r = torch.stack(seed_conf_r, dim=0).median(dim=0).values
        fg = torch.stack(seed_fg, dim=0).median(dim=0).values
        dis_f = torch.sqrt(((flow_f - med_f[None]).square().sum(dim=2)).median(dim=0).values + 1.0e-8)
        dis_r = torch.sqrt(((flow_r - med_r[None]).square().sum(dim=2)).median(dim=0).values + 1.0e-8)

        forward_rows.append(torch.cat((med_f, conf_f[:, None], fg[:, None], dis_f[:, None]), dim=1).half().cpu())
        reverse_rows.append(torch.cat((med_r, conf_r[:, None], fg[:, None], dis_r[:, None]), dim=1).half().cpu())
        confidences.append(0.5 * (conf_f.mean(dim=(-2, -1)) + conf_r.mean(dim=(-2, -1))).float().cpu())
        disagreements.append(0.5 * (dis_f.mean(dim=(-2, -1)) + dis_r.mean(dim=(-2, -1))).float().cpu())

    if map_shape is None:
        raise ValueError("Cannot extract empty split")
    confidence = torch.cat(confidences)
    disagreement = torch.cat(disagreements)
    return DenseInputs(
        forward=torch.cat(forward_rows),
        reverse=torch.cat(reverse_rows),
        diagnostics={
            "map_height": float(map_shape[0]),
            "map_width": float(map_shape[1]),
            "mean_matching_confidence": float(confidence.mean()),
            "mean_seed_flow_disagreement": float(disagreement.mean()),
        },
    )


def _sequence_folds(sequence_ids: list[str], count: int) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
    unique = sorted(set(sequence_ids))
    if count <= 1 or len(unique) < count:
        raise ValueError("Invalid fold count")
    result = []
    sequence_array = np.asarray(sequence_ids, dtype=object)
    for fold in range(count):
        held = unique[fold::count]
        val = np.flatnonzero(np.isin(sequence_array, held))
        train = np.flatnonzero(~np.isin(sequence_array, held))
        result.append((train, val, held))
    return result


def _train_decoder(
    dense: DenseInputs,
    split: Any,
    indices: np.ndarray,
    *,
    model_config: ObjectEventV420Config,
    train_config: TrainConfig,
    loss_config: ObjectEventV420LossConfig,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[BoxPseudoFlowRefiner, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = BoxPseudoFlowRefiner(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    h, w = dense.forward.shape[-2:]
    for epoch in range(1, epochs + 1):
        shuffled = np.asarray(indices, dtype=np.int64).copy()
        rng.shuffle(shuffled)
        totals: list[float] = []
        flows: list[float] = []
        divergences: list[float] = []
        residuals: list[float] = []
        model.train()
        for start in range(0, len(shuffled), train_config.head_batch_size):
            batch_idx = shuffled[start:start + train_config.head_batch_size]
            idx = torch.as_tensor(batch_idx, dtype=torch.long)
            f = dense.forward[idx].to(device=device, dtype=torch.float32)
            r = dense.reverse[idx].to(device=device, dtype=torch.float32)
            boxes = split.boxes_xyxy[idx].to(device=device, dtype=torch.float32)
            target_f, mask_f = box_affine_pseudoflow(
                boxes,
                source_height=split.source_height,
                source_width=split.source_width,
                target_height=h,
                target_width=w,
                first_index=1,
                second_index=2,
            )
            target_r, mask_r = box_affine_pseudoflow(
                boxes,
                source_height=split.source_height,
                source_width=split.source_width,
                target_height=h,
                target_width=w,
                first_index=2,
                second_index=1,
            )
            refined_f, residual_f = model(f)
            refined_r, residual_r = model(r)
            loss, components = pseudoflow_loss(
                refined_f, residual_f, refined_r, residual_r,
                target_f, mask_f, target_r, mask_r, config=loss_config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            totals.append(float(loss.detach()))
            flows.append(float(components["flow"].detach()))
            divergences.append(float(components["divergence"].detach()))
            residuals.append(float(components["residual"].detach()))
        history.append({
            "epoch": float(epoch),
            "loss": float(np.mean(totals)),
            "flow_loss": float(np.mean(flows)),
            "divergence_loss": float(np.mean(divergences)),
            "residual_loss": float(np.mean(residuals)),
        })
    return model, history


@torch.no_grad()
def _score_decoder(
    model: BoxPseudoFlowRefiner,
    dense: DenseInputs,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    foreground_floor: float,
    confidence_floor: float,
) -> np.ndarray:
    model.eval()
    values: list[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        idx = torch.as_tensor(batch_idx, dtype=torch.long)
        f = dense.forward[idx].to(device=device, dtype=torch.float32)
        r = dense.reverse[idx].to(device=device, dtype=torch.float32)
        refined_f, _ = model(f)
        refined_r, _ = model(r)
        div_f, _, _ = dense_flow_scores(
            refined_f[:, 0], refined_f[:, 1], f[:, 3], f[:, 2],
            foreground_floor=foreground_floor, confidence_floor=confidence_floor,
        )
        div_r, _, _ = dense_flow_scores(
            refined_r[:, 0], refined_r[:, 1], r[:, 3], r[:, 2],
            foreground_floor=foreground_floor, confidence_floor=confidence_floor,
        )
        values.append((0.5 * (div_f - div_r)).float().cpu())
    return torch.cat(values).numpy().astype(np.float64)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _score_diagnostics(frame: pd.DataFrame, score: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    pos = target >= 0.0
    neg = ~pos
    sign = score >= 0.0
    metrics = {
        "count": float(len(frame)),
        "pearson_to_target_expansion": _pearson(score, target),
        "positive_accuracy": float(np.mean(sign[pos])) if np.any(pos) else 1.0,
        "negative_accuracy": float(np.mean(~sign[neg])) if np.any(neg) else 1.0,
        "predicted_negative_rate": float(np.mean(~sign)),
        "mean_abs_score": float(np.mean(np.abs(score))),
    }
    rows = []
    for sequence in sorted(frame["sequence_id"].astype(str).unique()):
        mask = frame["sequence_id"].astype(str).to_numpy() == sequence
        sub_target = target[mask]
        sub_score = score[mask]
        sub_pos = sub_target >= 0.0
        sub_neg = ~sub_pos
        sub_sign = sub_score >= 0.0
        rows.append({
            "sequence_id": sequence,
            "count": int(mask.sum()),
            "pearson": _pearson(sub_score, sub_target),
            "positive_accuracy": float(np.mean(sub_sign[sub_pos])) if np.any(sub_pos) else 1.0,
            "negative_accuracy": float(np.mean(~sub_sign[sub_neg])) if np.any(sub_neg) else 1.0,
        })
    return metrics, pd.DataFrame(rows)


def _sign_magnitude_prediction(score: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(score) < 0.0, -np.abs(magnitude), np.abs(magnitude))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()

    model_config, train_config, loss_config, extraction, decision_config = _load_config(args.config)
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

    train_dense = _extract_dense_inputs(
        frozen, train_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        search_radius=int(extraction["search_radius"]),
        correlation_temperature=extraction["correlation_temperature"],
    )
    validation_dense = _extract_dense_inputs(
        frozen, validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        search_radius=int(extraction["search_radius"]),
        correlation_temperature=extraction["correlation_temperature"],
    )

    oof_score = np.zeros(len(train_split), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold, (train_idx, held_idx, held_sequences) in enumerate(
        _sequence_folds(train_split.sequence_ids, train_config.fold_count)
    ):
        model, history = _train_decoder(
            train_dense, train_split, train_idx,
            model_config=model_config, train_config=train_config, loss_config=loss_config,
            epochs=train_config.fold_epochs, device=device, seed=train_config.seed + fold,
        )
        score = _score_decoder(
            model, train_dense, held_idx,
            batch_size=train_config.head_batch_size, device=device,
            foreground_floor=extraction["foreground_floor"],
            confidence_floor=extraction["confidence_floor"],
        )
        oof_score[held_idx] = score
        held_frame = train_frame.iloc[held_idx].reset_index(drop=True)
        metrics, per_sequence = _score_diagnostics(held_frame, score)
        fold_records.append({
            "fold": fold,
            "held_out_sequences": held_sequences,
            "epochs": train_config.fold_epochs,
            "final_train_loss": history[-1],
            "score_metrics": metrics,
            "per_sequence": per_sequence.to_dict(orient="records"),
        })

    oof_metrics, oof_per_sequence = _score_diagnostics(train_frame, oof_score)

    all_train = np.arange(len(train_split), dtype=np.int64)
    final_model, final_history = _train_decoder(
        train_dense, train_split, all_train,
        model_config=model_config, train_config=train_config, loss_config=loss_config,
        epochs=train_config.final_epochs, device=device, seed=train_config.seed + 100,
    )
    val_indices = np.arange(len(validation_split), dtype=np.int64)
    validation_score = _score_decoder(
        final_model, validation_dense, val_indices,
        batch_size=train_config.head_batch_size, device=device,
        foreground_floor=extraction["foreground_floor"],
        confidence_floor=extraction["confidence_floor"],
    )
    validation_score_metrics, validation_per_sequence = _score_diagnostics(validation_frame, validation_score)

    magnitude = np.abs(validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64))
    decoder_prediction = _sign_magnitude_prediction(validation_score, magnitude)
    decoder_prediction_metrics, decoder_prediction_per_sequence = _metrics(
        validation_frame, decoder_prediction, minimum_negatives=20
    )
    baseline_prediction = validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    baseline_metrics, _ = _metrics(validation_frame, baseline_prediction, minimum_negatives=20)

    v419 = json.loads(args.v419_summary.read_text(encoding="utf-8"))
    raw_val = next(
        row for row in v419["score_diagnostics"]["validation"] if row["feature"] == "dense_divergence"
    )
    raw_train = next(
        row for row in v419["score_diagnostics"]["train"] if row["feature"] == "dense_divergence"
    )
    raw_validation_pearson = float(raw_val["pearson_to_target_expansion"])
    raw_train_pearson = float(raw_train["pearson_to_target_expansion"])

    per_seq_min = float(validation_per_sequence["pearson"].min())
    val_p = float(validation_score_metrics["pearson_to_target_expansion"])
    oof_p = float(oof_metrics["pearson_to_target_expansion"])
    improvement = val_p - raw_validation_pearson
    if (
        val_p >= decision_config["minimum_validation_score_pearson"]
        and oof_p >= decision_config["minimum_oof_score_pearson"]
        and per_seq_min >= decision_config["minimum_all_sequence_score_pearson"]
        and improvement >= decision_config["required_improvement_over_raw_divergence"]
    ):
        recommendation = "pseudoflow_decoder_supported_partial_unfreeze_with_flow_auxiliary_loss"
    elif (
        oof_p >= decision_config["minimum_oof_score_pearson"]
        and per_seq_min >= decision_config["minimum_all_sequence_score_pearson"]
        and val_p >= raw_validation_pearson - 0.02
    ):
        recommendation = "pseudoflow_decoder_neutral_integrate_auxiliary_loss_then_partial_unfreeze"
    else:
        recommendation = "frozen_refiner_insufficient_move_pseudoflow_divergence_supervision_into_encoder"

    decision = {
        "recommendation": recommendation,
        "comparisons": {
            "raw_v419_train_divergence_pearson": raw_train_pearson,
            "raw_v419_validation_divergence_pearson": raw_validation_pearson,
            "v420_oof_divergence_pearson": oof_p,
            "v420_validation_divergence_pearson": val_p,
            "validation_improvement_over_v419_raw": improvement,
            "minimum_validation_sequence_divergence_pearson": per_seq_min,
        },
        "note": "Development-only decision; validation never selects epochs/weights. Official eAP test and EvTTC remain unopened.",
    }

    torch.save({
        "model_state_dict": final_model.state_dict(),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "loss_config": asdict(loss_config),
        "extraction": extraction,
        "epochs": train_config.final_epochs,
    }, args.output_dir / "box_pseudoflow_decoder.pt")

    train_rows = train_frame[["sequence_id", "sample_token", "track_id", "target_expansion"]].copy()
    train_rows["oof_pseudoflow_divergence_score"] = oof_score
    train_rows.to_csv(args.output_dir / "train_oof_scores.csv", index=False)
    oof_per_sequence.to_csv(args.output_dir / "train_oof_per_sequence.csv", index=False)

    val_rows = validation_frame[
        ["sequence_id", "sample_token", "track_id", "target_expansion", "target_ttc_s", "delta_t_s"]
    ].copy()
    val_rows["baseline_prediction_expansion"] = baseline_prediction
    val_rows["pseudoflow_divergence_score"] = validation_score
    val_rows["pseudoflow_sign_baseline_magnitude_prediction"] = decoder_prediction
    val_rows.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    validation_per_sequence.to_csv(args.output_dir / "validation_score_per_sequence.csv", index=False)
    decoder_prediction_per_sequence.to_csv(args.output_dir / "validation_prediction_per_sequence.csv", index=False)
    pd.DataFrame(final_history).to_csv(args.output_dir / "final_training_history.csv", index=False)

    result = {
        "artifact_type": "object_event_v4_20_box_pseudoflow_decoder",
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {
            "model": asdict(model_config),
            "train": asdict(train_config),
            "loss": asdict(loss_config),
            "extraction": extraction,
            "decision": decision_config,
        },
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "dense_input_diagnostics": {
            "train": train_dense.diagnostics,
            "validation": validation_dense.diagnostics,
        },
        "folds": fold_records,
        "oof_score_metrics": oof_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "validation_score_metrics": validation_score_metrics,
        "decoder_sign_baseline_magnitude_validation_metrics": decoder_prediction_metrics,
        "decision": decision,
        "scientific_contract": {
            "three_frozen_true_seed_v48_backbones": True,
            "boxes_are_training_only_pseudoflow_targets": True,
            "boxes_not_forward_inputs": True,
            "ttc_labels_not_used_in_decoder_loss": True,
            "uniform_without_replacement_training": True,
            "fixed_epoch_schedule_no_validation_selection": True,
            "radial_slope_excluded_after_v419_sequence_sign_instability": True,
            "final_scalar_is_endpoint_antisymmetrised_divergence": True,
            "v410_magnitude_frozen_for_sign_isolation": True,
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
    parser.add_argument("--v419-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
