#!/usr/bin/env python3
"""Grouped OOF model selection for Object Event TTC v4.28."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.analyze_object_event_v4_24_orchestrator import _sequence_folds, _subset_split  # noqa: E402
from scripts.train_e_jepa_object_event_v4_6 import _materialize  # noqa: E402
from scripts.train_e_jepa_object_event_v4_8 import _load_config as _load_v48_config  # noqa: E402
from scripts.train_e_jepa_object_event_v4_12 import _load_backbone, _read_ensemble, _align_ensemble  # noqa: E402
from scripts.train_e_jepa_object_event_v4_16 import _json_safe, _metrics, _resolve_device  # noqa: E402
from e_jepa_ttc.models.object_event_v4_22 import configure_partial_geometry_unfreeze  # noqa: E402
from e_jepa_ttc.models.object_event_v4_28 import ObjectEventTTCV428, ObjectEventV428Config  # noqa: E402
from e_jepa_ttc.training.object_event_v4_22 import relative_parameter_anchor  # noqa: E402
from e_jepa_ttc.training.object_event_v4_26 import track_metrics  # noqa: E402
from e_jepa_ttc.training.object_event_v4_27 import target_log_height_ratio  # noqa: E402
from e_jepa_ttc.training.object_event_v4_28 import (  # noqa: E402
    ObjectEventV428LossConfig,
    object_event_v4_28_loss,
)


def _construct(cls: type[Any], raw: Mapping[str, Any]) -> Any:
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(raw))


def _parse_seed_paths(values: list[str]) -> dict[int, Path]:
    out = {int(x.split("=", 1)[0]): Path(x.split("=", 1)[1]) for x in values}
    if sorted(out) != [7, 13, 23]:
        raise ValueError("exact adapted seeds 7,13,23 required")
    return dict(sorted(out.items()))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) <= 1.0e-12 or np.std(b) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _require_finite(name: str, tensor: torch.Tensor, *, context: str) -> None:
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    detached = tensor.detach().float()
    values = detached[finite]
    stats = {
        "shape": list(tensor.shape),
        "nonfinite": int((~finite).sum().item()),
        "finite_min": float(values.min().item()) if values.numel() else None,
        "finite_max": float(values.max().item()) if values.numel() else None,
    }
    raise FloatingPointError(f"v4.28 non-finite {name} at {context}: {stats}")


def _split_frame(split: Any) -> pd.DataFrame:
    target = (split.delta_t_s / split.target_ttc_s).clamp(-0.25, 0.25).cpu().numpy().astype(np.float64)
    return pd.DataFrame({
        "sequence_id": [str(x) for x in split.sequence_ids],
        "sample_token": [str(x) for x in split.sample_tokens],
        "track_id": [str(x) for x in split.track_ids],
        "target_expansion": target,
        "delta_t_s": split.delta_t_s.cpu().numpy(),
        "target_ttc_s": split.target_ttc_s.cpu().numpy(),
    })


def _train_model(
    checkpoint: Path,
    split: Any,
    *,
    v48_config: Path,
    model_cfg: ObjectEventV428Config,
    loss_cfg: ObjectEventV428LossConfig,
    train_cfg: Mapping[str, Any],
    device: torch.device,
    seed: int,
    epochs: int,
    label: str,
) -> tuple[ObjectEventTTCV428, list[dict[str, float]]]:
    torch.manual_seed(seed)
    backbone, _ = _load_backbone(v48_config_path=v48_config, checkpoint_path=checkpoint)
    backbone = backbone.to(device)
    selected = configure_partial_geometry_unfreeze(backbone, int(train_cfg["geometry_tail_tensors"]))
    model = ObjectEventTTCV428(backbone, model_cfg).to(device)
    initial = {
        name: p.detach().float().cpu().clone()
        for name, p in backbone.foreground_model.geometry_encoder.named_parameters()
        if p.requires_grad
    }
    geometry = [p for p in backbone.foreground_model.geometry_encoder.parameters() if p.requires_grad]
    head = model.head_parameters()
    optimizer = torch.optim.AdamW(
        [
            {"params": head, "lr": float(train_cfg["projection_learning_rate"])},
            {"params": geometry, "lr": float(train_cfg["geometry_learning_rate"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    batch_size = int(model_cfg.batch_size)
    tracked = ("loss", "lhr", "expansion", "correlation", "sign", "posterior", "entropy", "anchor")
    for epoch in range(1, epochs + 1):
        order = np.arange(len(split.events), dtype=np.int64)
        rng.shuffle(order)
        accum = {key: [] for key in tracked}
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
            events = split.events[idx].to(device=device, dtype=torch.float32)
            dt = split.delta_t_s[idx].to(device=device, dtype=torch.float32)
            ttc = split.target_ttc_s[idx].to(device=device, dtype=torch.float32)
            heights = split.visible_heights_px[idx].to(device=device, dtype=torch.float32)
            context = f"{label} seed={seed} epoch={epoch} batch_start={start}"
            for input_name, input_tensor in (
                ("events", events), ("delta_t_s", dt), ("target_ttc_s", ttc), ("visible_heights_px", heights)
            ):
                _require_finite(input_name, input_tensor, context=context)
            output = model(events)
            for output_name, output_tensor in (
                ("predicted_log_eta", output.predicted_log_eta),
                ("expansion", output.expansion),
                ("scale_logits", output.scale_logits),
                ("scale_probabilities", output.scale_probabilities),
                ("scale_entropy", output.scale_entropy),
            ):
                _require_finite(output_name, output_tensor, context=context)
            primary, pieces = object_event_v4_28_loss(output, dt, ttc, heights, config=loss_cfg)
            _require_finite("primary_loss", primary, context=context)
            trainables = {
                name: p for name, p in backbone.foreground_model.geometry_encoder.named_parameters() if p.requires_grad
            }
            anchor = relative_parameter_anchor(trainables, initial, epsilon=1.0e-6)
            loss = primary + float(train_cfg["geometry_anchor_weight"]) * anchor
            _require_finite("total_loss", loss, context=context)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                head + geometry, float(train_cfg["max_grad_norm"]), error_if_nonfinite=True
            )
            _require_finite("gradient_norm", grad_norm, context=context)
            optimizer.step()
            accum["loss"].append(float(loss.detach()))
            accum["anchor"].append(float(anchor.detach()))
            for key in ("lhr", "expansion", "correlation", "sign", "posterior", "entropy"):
                _require_finite(f"loss_component:{key}", pieces[key], context=context)
                accum[key].append(float(pieces[key].detach()))
        row = {"epoch": float(epoch), **{key: float(np.mean(values)) for key, values in accum.items()}}
        history.append(row)
        print(
            f"[v4.28] {label} seed={seed} epoch={epoch}/{epochs} "
            f"loss={row['loss']:.4f} lhr={row['lhr']:.4f} post={row['posterior']:.4f} "
            f"corr={row['correlation']:.4f} sign={row['sign']:.4f} ent={row['entropy']:.3f}",
            flush=True,
        )
    model.eval()
    model._v428_selected_geometry = selected  # type: ignore[attr-defined]
    return model, history


@torch.no_grad()
def _predict(
    model: ObjectEventTTCV428,
    split: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expansion, log_eta, entropy, rotation = [], [], [], []
    model.eval()
    for start in range(0, len(split.events), batch_size):
        events = split.events[start:start + batch_size].to(device=device, dtype=torch.float32)
        output = model(events)
        expansion.append(output.expansion.float().cpu())
        log_eta.append(output.predicted_log_eta.float().cpu())
        entropy.append(output.scale_entropy.float().cpu())
        rotation.append(output.expected_rotation_degrees.float().cpu())
    return tuple(
        torch.cat(parts).numpy().astype(np.float64)
        for parts in (expansion, log_eta, entropy, rotation)
    )  # type: ignore[return-value]


def _metrics_bundle(
    frame: pd.DataFrame,
    split: Any,
    prediction: np.ndarray,
    log_eta: np.ndarray,
    entropy: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ttc, per_sequence = _metrics(frame, prediction, minimum_negatives=20)
    target_lhr = target_log_height_ratio(split.visible_heights_px).cpu().numpy().astype(np.float64)
    ttc["log_eta_pearson"] = _pearson(log_eta, target_lhr)
    ttc["log_eta_mae"] = float(np.mean(np.abs(log_eta - target_lhr)))
    ttc["scale_entropy_mean"] = float(np.mean(entropy))
    target_std = float(ttc.get("target_std", np.std(frame["target_expansion"].to_numpy())))
    prediction_std = float(ttc.get("prediction_std", np.std(prediction)))
    ttc["prediction_std_ratio"] = prediction_std / max(target_std, 1.0e-12)
    seq_lhr = []
    extended = frame.assign(pred_log_eta=log_eta, target_log_eta=target_lhr, scale_entropy=entropy)
    for sequence_id, group in extended.groupby("sequence_id", sort=True):
        seq_lhr.append({
            "sequence_id": str(sequence_id),
            "log_eta_pearson": _pearson(group["pred_log_eta"].to_numpy(), group["target_log_eta"].to_numpy()),
            "log_eta_mae": float(np.mean(np.abs(group["pred_log_eta"] - group["target_log_eta"]))),
            "scale_entropy_mean": float(group["scale_entropy"].mean()),
        })
    per_sequence = per_sequence.merge(pd.DataFrame(seq_lhr), on="sequence_id", how="left")
    ttc["minimum_sequence_log_eta_pearson"] = float(per_sequence["log_eta_pearson"].min())
    return ttc, per_sequence


def _selection_objective(metrics: Mapping[str, Any], tracks: Mapping[str, Any]) -> float:
    ratio = max(float(metrics["prediction_std_ratio"]), 1.0e-6)
    amplitude_penalty = abs(math.log(ratio))
    return float(
        float(metrics["pearson"])
        + 0.25 * float(metrics["log_eta_pearson"])
        + 0.15 * float(metrics["balanced_sign_accuracy"])
        + 0.10 * float(metrics["negative_accuracy"])
        + 0.10 * float(tracks["negative_track_macro_accuracy"])
        + 0.08 * float(metrics["minimum_sequence_pearson"])
        - 0.08 * amplitude_penalty
        - 0.03 * float(metrics["scale_entropy_mean"])
    )


def _passes_oof_gate(
    metrics: Mapping[str, Any],
    tracks: Mapping[str, Any],
    selection: Mapping[str, Any],
    v427: Mapping[str, Any],
) -> tuple[bool, dict[str, bool]]:
    v427_metrics = cast(Mapping[str, Any], v427["oof_train_metrics"])
    v427_tracks = cast(Mapping[str, Any], v427["oof_track_metrics"])
    checks = {
        "pearson_absolute": float(metrics["pearson"]) >= float(selection["minimum_oof_pearson"]),
        "negative_accuracy": float(metrics["negative_accuracy"]) >= float(selection["minimum_oof_negative_accuracy"]),
        "balanced_sign": float(metrics["balanced_sign_accuracy"]) >= float(selection["minimum_oof_balanced_sign"]),
        "log_eta_absolute": float(metrics["log_eta_pearson"]) >= float(selection["minimum_oof_log_eta_pearson"]),
        "negative_track_macro": float(tracks["negative_track_macro_accuracy"])
        >= float(selection["minimum_oof_negative_track_macro_accuracy"]),
        "minimum_sequence_pearson": float(metrics["minimum_sequence_pearson"])
        >= float(selection["minimum_oof_sequence_pearson"]),
        "prediction_std_ratio": float(metrics["prediction_std_ratio"])
        >= float(selection["minimum_prediction_std_ratio"]),
        "pearson_gain_over_v427": float(metrics["pearson"])
        >= float(v427_metrics["pearson"]) + float(selection["minimum_pearson_gain_over_v427"]),
        "log_eta_gain_over_v427": float(metrics["log_eta_pearson"])
        >= float(v427_metrics["log_eta_pearson"]) + float(selection["minimum_log_eta_gain_over_v427"]),
        "negative_track_gain_over_v427": float(tracks["negative_track_macro_accuracy"])
        >= float(v427_tracks["negative_track_macro_accuracy"])
        + float(selection["minimum_negative_track_gain_over_v427"]),
    }
    return all(checks.values()), checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v427-summary", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
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
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    loss_cfg = _construct(ObjectEventV428LossConfig, cast(Mapping[str, Any], raw["loss"]))
    train_cfg = cast(Mapping[str, Any], raw["train"])
    selection = cast(Mapping[str, Any], raw["selection"])
    final_decision = cast(Mapping[str, Any], raw["final_decision"])
    arm_raw = cast(Mapping[str, Mapping[str, Any]], raw["arms"])
    arm_configs = {name: _construct(ObjectEventV428Config, config) for name, config in arm_raw.items()}
    if sorted(arm_configs) != ["profile_posterior", "spatial_rotation_posterior"]:
        raise ValueError("v4.28 requires the two preregistered arms")
    checkpoints = _parse_seed_paths(args.adapted_checkpoint)
    v427 = json.loads(args.v427_summary.read_text(encoding="utf-8"))
    if v427.get("artifact_type") != "object_event_v4_27_scale_correlation_lhr":
        raise ValueError("--v427-summary is not the expected v4.27 artifact")
    device = _resolve_device(args.device)
    base_cfg, _, _, _, _ = _load_v48_config(args.v48_config)
    train_split, train_manifest = _materialize(args.cache_manifest, "train", input_size=base_cfg.input_size)
    train_frame = _split_frame(train_split)
    folds = _sequence_folds(
        train_frame["sequence_id"].to_numpy(dtype=object),
        int(train_cfg["fold_count"]),
        int(train_cfg["seed"]),
    )
    all_idx = np.arange(len(train_frame), dtype=np.int64)
    arm_results: dict[str, dict[str, Any]] = {}
    arm_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    history_root = args.output_dir / "cv_histories"
    history_root.mkdir()

    for arm_index, (arm_name, model_cfg) in enumerate(arm_configs.items()):
        print(f"[v4.28] OOF arm={arm_name} matcher={model_cfg.matcher}", flush=True)
        per_seed_pred, per_seed_lhr, per_seed_entropy, per_seed_rotation = [], [], [], []
        fold_records: list[dict[str, Any]] = []
        arm_hist = history_root / arm_name
        arm_hist.mkdir()
        for seed in [int(x) for x in train_cfg["seeds"]]:
            pred = np.full(len(train_frame), np.nan, dtype=np.float64)
            lhr = np.full(len(train_frame), np.nan, dtype=np.float64)
            entropy = np.full(len(train_frame), np.nan, dtype=np.float64)
            rotation = np.full(len(train_frame), np.nan, dtype=np.float64)
            for fold_id, held_idx in enumerate(folds):
                print(
                    f"[v4.28] arm={arm_name} seed={seed} fold={fold_id + 1}/{len(folds)} training",
                    flush=True,
                )
                fit_idx = np.setdiff1d(all_idx, held_idx, assume_unique=True)
                model, history = _train_model(
                    checkpoints[seed],
                    _subset_split(train_split, fit_idx),
                    v48_config=args.v48_config,
                    model_cfg=model_cfg,
                    loss_cfg=loss_cfg,
                    train_cfg=train_cfg,
                    device=device,
                    seed=int(train_cfg["seed"]) + arm_index * 10000 + seed * 100 + fold_id,
                    epochs=int(train_cfg["epochs"]),
                    label=arm_name,
                )
                held_split = _subset_split(train_split, held_idx)
                fold_pred, fold_lhr, fold_entropy, fold_rotation = _predict(
                    model, held_split, batch_size=model_cfg.batch_size, device=device
                )
                pred[held_idx] = fold_pred
                lhr[held_idx] = fold_lhr
                entropy[held_idx] = fold_entropy
                rotation[held_idx] = fold_rotation
                local_frame = train_frame.iloc[held_idx].reset_index(drop=True)
                local_metrics, _ = _metrics_bundle(
                    local_frame, held_split, fold_pred, fold_lhr, fold_entropy
                )
                fold_records.append({
                    "seed": seed,
                    "fold": fold_id,
                    "held_out_sequences": sorted(local_frame["sequence_id"].unique().tolist()),
                    "metrics": local_metrics,
                    "rotation_abs_mean_degrees": float(np.mean(np.abs(fold_rotation))),
                    "final_history": history[-1],
                })
                pd.DataFrame(history).to_csv(arm_hist / f"seed{seed}_fold{fold_id}.csv", index=False)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if not all(np.isfinite(x).all() for x in (pred, lhr, entropy, rotation)):
                raise RuntimeError(f"incomplete OOF coverage for arm={arm_name} seed={seed}")
            per_seed_pred.append(pred)
            per_seed_lhr.append(lhr)
            per_seed_entropy.append(entropy)
            per_seed_rotation.append(rotation)

        oof_pred = np.median(np.stack(per_seed_pred), axis=0)
        oof_lhr = np.median(np.stack(per_seed_lhr), axis=0)
        oof_entropy = np.median(np.stack(per_seed_entropy), axis=0)
        oof_rotation = np.median(np.stack(per_seed_rotation), axis=0)
        metrics, per_sequence = _metrics_bundle(
            train_frame, train_split, oof_pred, oof_lhr, oof_entropy
        )
        tracks, per_track = track_metrics(
            train_frame, oof_pred, minimum_track_samples=8, minimum_negative_track_samples=4
        )
        objective = _selection_objective(metrics, tracks)
        arm_results[arm_name] = {
            "matcher": model_cfg.matcher,
            "objective": objective,
            "oof_metrics": metrics,
            "oof_track_metrics": tracks,
            "rotation_abs_mean_degrees": float(np.mean(np.abs(oof_rotation))),
            "fold_records": fold_records,
        }
        arm_predictions[arm_name] = (oof_pred, oof_lhr, oof_entropy, oof_rotation)
        train_frame.assign(
            v428_prediction_expansion=oof_pred,
            v428_predicted_log_eta=oof_lhr,
            v428_scale_entropy=oof_entropy,
            v428_expected_rotation_degrees=oof_rotation,
        ).to_csv(args.output_dir / f"oof_train_predictions_{arm_name}.csv", index=False)
        per_sequence.to_csv(args.output_dir / f"oof_train_per_sequence_{arm_name}.csv", index=False)
        per_track.to_csv(args.output_dir / f"oof_train_per_track_{arm_name}.csv", index=False)
        print(
            f"[v4.28] arm={arm_name} OOF pearson={metrics['pearson']:.4f} "
            f"log_eta={metrics['log_eta_pearson']:.4f} neg={metrics['negative_accuracy']:.4f} "
            f"balanced={metrics['balanced_sign_accuracy']:.4f} std_ratio={metrics['prediction_std_ratio']:.3f} "
            f"neg_track={tracks['negative_track_macro_accuracy']:.4f} objective={objective:.4f}",
            flush=True,
        )

    champion = max(arm_results, key=lambda name: float(arm_results[name]["objective"]))
    champion_result = arm_results[champion]
    oof_pass, gate_checks = _passes_oof_gate(
        cast(Mapping[str, Any], champion_result["oof_metrics"]),
        cast(Mapping[str, Any], champion_result["oof_track_metrics"]),
        selection,
        v427,
    )
    ranking = pd.DataFrame([
        {
            "arm": name,
            "matcher": result["matcher"],
            "objective": result["objective"],
            "pearson": result["oof_metrics"]["pearson"],
            "log_eta_pearson": result["oof_metrics"]["log_eta_pearson"],
            "negative_accuracy": result["oof_metrics"]["negative_accuracy"],
            "balanced_sign_accuracy": result["oof_metrics"]["balanced_sign_accuracy"],
            "minimum_sequence_pearson": result["oof_metrics"]["minimum_sequence_pearson"],
            "prediction_std_ratio": result["oof_metrics"]["prediction_std_ratio"],
            "scale_entropy_mean": result["oof_metrics"]["scale_entropy_mean"],
            "negative_track_macro_accuracy": result["oof_track_metrics"]["negative_track_macro_accuracy"],
        }
        for name, result in arm_results.items()
    ]).sort_values("objective", ascending=False)
    ranking.to_csv(args.output_dir / "oof_arm_ranking.csv", index=False)

    if not oof_pass:
        if champion == "profile_posterior" and float(champion_result["oof_metrics"]["prediction_std_ratio"]) >= 0.70:
            recommendation = "posterior_calibration_helped_but_1d_geometry_insufficient_next_affine_normal_flow_v429"
        elif champion == "spatial_rotation_posterior":
            recommendation = "spatial_rotation_matcher_best_but_oof_insufficient_next_event_native_affine_flow_v429"
        else:
            recommendation = "v428_oof_failed_next_event_native_affine_flow_v429"
        summary = {
            "artifact_type": "object_event_v4_28_multiscale_posterior",
            "status": "completed_oof_gate_failed",
            "elapsed_seconds": time.perf_counter() - started,
            "v427_reference": {
                "oof_train_metrics": v427["oof_train_metrics"],
                "oof_track_metrics": v427["oof_track_metrics"],
            },
            "arm_results": arm_results,
            "champion": champion,
            "oof_gate_passed": False,
            "oof_gate_checks": gate_checks,
            "decision": {"recommendation": recommendation, "strong_gain": False},
            "scientific_contract": {
                "grouped_sequence_oof": True,
                "two_fixed_arms_predeclared": True,
                "height_labels_training_only": True,
                "boxes_not_forward_features": True,
                "development_validation_not_materialized_after_oof_failure": True,
                "official_eap_test_not_opened": True,
                "evttc_not_opened": True,
            },
            "train_manifest": train_manifest,
            "config": raw,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
        )
        print(json.dumps(_json_safe({"status": summary["status"], "champion": champion, "gate": gate_checks, "recommendation": recommendation}), indent=2))
        return 0

    # The fixed all-seed OOF champion has earned one development-validation evaluation.
    val_split, validation_manifest = _materialize(
        args.cache_manifest, "validation", input_size=base_cfg.input_size
    )
    val_frame = _split_frame(val_split)
    champion_cfg = arm_configs[champion]
    final_predictions, final_lhr, final_entropy, final_rotation = [], [], [], []
    final_records: list[dict[str, Any]] = []
    champion_index = list(arm_configs).index(champion)
    for seed in [int(x) for x in train_cfg["seeds"]]:
        print(f"[v4.28] final champion={champion} full-train seed={seed}", flush=True)
        model, history = _train_model(
            checkpoints[seed],
            train_split,
            v48_config=args.v48_config,
            model_cfg=champion_cfg,
            loss_cfg=loss_cfg,
            train_cfg=train_cfg,
            device=device,
            seed=int(train_cfg["seed"]) + champion_index * 10000 + seed * 1000,
            epochs=int(train_cfg["final_epochs"]),
            label=f"final:{champion}",
        )
        prediction, lhr, entropy, rotation = _predict(
            model, val_split, batch_size=champion_cfg.batch_size, device=device
        )
        final_predictions.append(prediction)
        final_lhr.append(lhr)
        final_entropy.append(entropy)
        final_rotation.append(rotation)
        checkpoint_path = args.output_dir / f"v428_{champion}_seed_{seed}.pt"
        torch.save({"model_state_dict": model.state_dict(), "seed": seed, "arm": champion, "config": raw}, checkpoint_path)
        final_records.append({
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "final_history": history[-1],
            "entropy_mean": float(np.mean(entropy)),
            "rotation_abs_mean_degrees": float(np.mean(np.abs(rotation))),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    val_pred = np.median(np.stack(final_predictions), axis=0)
    val_lhr = np.median(np.stack(final_lhr), axis=0)
    val_entropy = np.median(np.stack(final_entropy), axis=0)
    val_rotation = np.median(np.stack(final_rotation), axis=0)
    val_metrics, val_per_sequence = _metrics_bundle(
        val_frame, val_split, val_pred, val_lhr, val_entropy
    )
    val_tracks, val_per_track = track_metrics(
        val_frame, val_pred, minimum_track_samples=8, minimum_negative_track_samples=4
    )
    v410 = _align_ensemble(val_split, _read_ensemble(args.ensemble_validation))
    v410_pred = v410["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    v410_metrics, _ = _metrics(v410, v410_pred, minimum_negatives=20)
    v410_tracks, _ = track_metrics(
        v410, v410_pred, minimum_track_samples=8, minimum_negative_track_samples=4
    )
    strong = (
        float(val_metrics["pearson"]) >= float(v410_metrics["pearson"]) + float(final_decision["minimum_pearson_gain_over_v410"])
        and float(val_metrics["negative_accuracy"]) >= float(v410_metrics["negative_accuracy"]) + float(final_decision["minimum_negative_accuracy_gain_over_v410"])
        and float(val_metrics["balanced_sign_accuracy"]) >= float(v410_metrics["balanced_sign_accuracy"]) + float(final_decision["minimum_balanced_sign_gain_over_v410"])
        and float(val_metrics["log_eta_pearson"]) >= float(final_decision["minimum_log_eta_pearson"])
        and float(val_tracks["negative_track_macro_accuracy"])
        >= float(v410_tracks["negative_track_macro_accuracy"])
        + float(final_decision["minimum_negative_track_macro_gain_over_v410"])
    )
    if strong:
        recommendation = "v428_supported_lock_architecture_then_long_multiseed_before_official_test"
    elif float(val_metrics["pearson"]) > float(v410_metrics["pearson"]):
        recommendation = "v428_transfers_but_sign_robustness_not_sota_refine_affine_flow_v429"
    else:
        recommendation = "v428_not_better_than_v410_move_to_event_native_affine_flow_v429"

    val_frame.assign(
        v410_prediction_expansion=v410_pred,
        v428_prediction_expansion=val_pred,
        v428_predicted_log_eta=val_lhr,
        v428_scale_entropy=val_entropy,
        v428_expected_rotation_degrees=val_rotation,
    ).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    val_per_sequence.to_csv(args.output_dir / "validation_per_sequence.csv", index=False)
    val_per_track.to_csv(args.output_dir / "validation_per_track.csv", index=False)
    summary = {
        "artifact_type": "object_event_v4_28_multiscale_posterior",
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "v427_reference": {
            "oof_train_metrics": v427["oof_train_metrics"],
            "oof_track_metrics": v427["oof_track_metrics"],
        },
        "arm_results": arm_results,
        "champion": champion,
        "oof_gate_passed": True,
        "oof_gate_checks": gate_checks,
        "v410_validation_metrics": v410_metrics,
        "v410_validation_track_metrics": v410_tracks,
        "validation_metrics": val_metrics,
        "validation_track_metrics": val_tracks,
        "final_seed_records": final_records,
        "decision": {
            "recommendation": recommendation,
            "strong_gain": bool(strong),
            "comparisons": {
                "v410_pearson": float(v410_metrics["pearson"]),
                "v428_pearson": float(val_metrics["pearson"]),
                "v410_negative_accuracy": float(v410_metrics["negative_accuracy"]),
                "v428_negative_accuracy": float(val_metrics["negative_accuracy"]),
                "v410_balanced_sign": float(v410_metrics["balanced_sign_accuracy"]),
                "v428_balanced_sign": float(val_metrics["balanced_sign_accuracy"]),
                "v410_negative_track_macro_accuracy": float(v410_tracks["negative_track_macro_accuracy"]),
                "v428_negative_track_macro_accuracy": float(val_tracks["negative_track_macro_accuracy"]),
            },
        },
        "scientific_contract": {
            "grouped_sequence_oof": True,
            "two_fixed_arms_predeclared": True,
            "height_labels_training_only": True,
            "boxes_not_forward_features": True,
            "validation_evaluated_once_after_fixed_oof_champion": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "config": raw,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(_json_safe({"status": "completed", "champion": champion, "recommendation": recommendation, "oof": champion_result["oof_metrics"], "validation": val_metrics}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
