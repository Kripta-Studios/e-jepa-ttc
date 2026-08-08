#!/usr/bin/env python3
"""Grouped OOF and final development-validation run for Object Event TTC v4.27."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, fields
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
from e_jepa_ttc.models.object_event_v4_27 import ObjectEventTTCV427, ObjectEventV427Config  # noqa: E402
from e_jepa_ttc.training.object_event_v4_22 import relative_parameter_anchor  # noqa: E402
from e_jepa_ttc.training.object_event_v4_26 import track_metrics  # noqa: E402
from e_jepa_ttc.training.object_event_v4_27 import (  # noqa: E402
    ObjectEventV427LossConfig,
    object_event_v4_27_loss,
    target_log_height_ratio,
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
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])




def _require_finite(name: str, tensor: torch.Tensor, *, context: str) -> None:
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    detached = tensor.detach().float()
    finite_values = detached[finite]
    stats = {
        "shape": list(tensor.shape),
        "nonfinite": int((~finite).sum().item()),
        "finite_min": float(finite_values.min().item()) if finite_values.numel() else None,
        "finite_max": float(finite_values.max().item()) if finite_values.numel() else None,
    }
    raise FloatingPointError(f"v4.27 non-finite {name} at {context}: {stats}")


def _require_finite_parameters(parameters: list[torch.Tensor], *, context: str) -> None:
    for index, parameter in enumerate(parameters):
        _require_finite(f"parameter[{index}]", parameter, context=context)

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
    model_cfg: ObjectEventV427Config,
    loss_cfg: ObjectEventV427LossConfig,
    train_cfg: Mapping[str, Any],
    device: torch.device,
    seed: int,
    epochs: int,
) -> tuple[ObjectEventTTCV427, list[dict[str, float]]]:
    torch.manual_seed(seed)
    backbone, _ = _load_backbone(v48_config_path=v48_config, checkpoint_path=checkpoint)
    backbone = backbone.to(device)
    selected = configure_partial_geometry_unfreeze(backbone, int(train_cfg["geometry_tail_tensors"]))
    model = ObjectEventTTCV427(backbone, model_cfg).to(device)
    initial = {
        name: p.detach().float().cpu().clone()
        for name, p in backbone.foreground_model.geometry_encoder.named_parameters()
        if p.requires_grad
    }
    geometry = [p for p in backbone.foreground_model.geometry_encoder.parameters() if p.requires_grad]
    projection = list(model.feature_projection.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": projection, "lr": float(train_cfg["projection_learning_rate"])},
            {"params": geometry, "lr": float(train_cfg["geometry_learning_rate"])},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    batch_size = int(train_cfg["batch_size"])
    for epoch in range(1, epochs + 1):
        order = np.arange(len(split.events), dtype=np.int64)
        rng.shuffle(order)
        accum = {k: [] for k in ("loss", "lhr", "expansion", "correlation", "sign", "entropy", "anchor")}
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
            events = split.events[idx].to(device=device, dtype=torch.float32)
            dt = split.delta_t_s[idx].to(device=device, dtype=torch.float32)
            ttc = split.target_ttc_s[idx].to(device=device, dtype=torch.float32)
            heights = split.visible_heights_px[idx].to(device=device, dtype=torch.float32)
            context = f"seed={seed} epoch={epoch} batch_start={start}"
            for input_name, input_tensor in (("events", events), ("delta_t_s", dt), ("target_ttc_s", ttc), ("visible_heights_px", heights)):
                _require_finite(input_name, input_tensor, context=context)
            output = model(events)
            _require_finite("predicted_log_eta", output.predicted_log_eta, context=context)
            _require_finite("expansion", output.expansion, context=context)
            _require_finite("scale_logits", output.scale_logits, context=context)
            _require_finite("scale_probabilities", output.scale_probabilities, context=context)
            primary, pieces = object_event_v4_27_loss(output, dt, ttc, heights, config=loss_cfg)
            _require_finite("primary_loss", primary, context=context)
            for piece_name, piece_tensor in pieces.items():
                _require_finite(f"loss_component:{piece_name}", piece_tensor, context=context)
            trainables = {
                name: p for name, p in backbone.foreground_model.geometry_encoder.named_parameters() if p.requires_grad
            }
            anchor = relative_parameter_anchor(trainables, initial, epsilon=1.0e-6)
            loss = primary + float(train_cfg["geometry_anchor_weight"]) * anchor
            _require_finite("anchor", anchor, context=context)
            _require_finite("total_loss", loss, context=context)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                projection + geometry,
                float(train_cfg["max_grad_norm"]),
                error_if_nonfinite=True,
            )
            _require_finite("gradient_norm", grad_norm, context=context)
            optimizer.step()
            _require_finite_parameters(projection + geometry, context=context + " after_optimizer")
            accum["loss"].append(float(loss.detach()))
            accum["anchor"].append(float(anchor.detach()))
            for key in ("lhr", "expansion", "correlation", "sign", "entropy"):
                accum[key].append(float(pieces[key].detach()))
        row = {"epoch": float(epoch), **{k: float(np.mean(v)) for k, v in accum.items()}}
        history.append(row)
        print(
            f"[v4.27] seed={seed} epoch={epoch}/{epochs} loss={row['loss']:.4f} "
            f"lhr={row['lhr']:.4f} corr={row['correlation']:.4f} sign={row['sign']:.4f}",
            flush=True,
        )
    model.eval()
    model._v427_selected_geometry = selected  # type: ignore[attr-defined]
    return model, history


@torch.no_grad()
def _predict(model: ObjectEventTTCV427, split: Any, *, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expansion, log_eta, entropy = [], [], []
    model.eval()
    for start in range(0, len(split.events), batch_size):
        events = split.events[start:start + batch_size].to(device=device, dtype=torch.float32)
        output = model(events)
        expansion.append(output.expansion.float().cpu())
        log_eta.append(output.predicted_log_eta.float().cpu())
        entropy.append(output.scale_entropy.float().cpu())
    return (
        torch.cat(expansion).numpy().astype(np.float64),
        torch.cat(log_eta).numpy().astype(np.float64),
        torch.cat(entropy).numpy().astype(np.float64),
    )


def _metrics_bundle(frame: pd.DataFrame, split: Any, prediction: np.ndarray, log_eta: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    ttc, per_sequence = _metrics(frame, prediction, minimum_negatives=20)
    target_lhr = target_log_height_ratio(split.visible_heights_px).cpu().numpy().astype(np.float64)
    ttc["log_eta_pearson"] = _pearson(log_eta, target_lhr)
    ttc["log_eta_mae"] = float(np.mean(np.abs(log_eta - target_lhr)))
    seq_lhr = []
    for sequence_id, group in frame.assign(pred_log_eta=log_eta, target_log_eta=target_lhr).groupby("sequence_id", sort=True):
        seq_lhr.append({
            "sequence_id": str(sequence_id),
            "log_eta_pearson": _pearson(group["pred_log_eta"].to_numpy(), group["target_log_eta"].to_numpy()),
            "log_eta_mae": float(np.mean(np.abs(group["pred_log_eta"] - group["target_log_eta"]))),
        })
    seq_lhr_df = pd.DataFrame(seq_lhr)
    per_sequence = per_sequence.merge(seq_lhr_df, on="sequence_id", how="left")
    ttc["minimum_sequence_log_eta_pearson"] = float(per_sequence["log_eta_pearson"].min())
    return ttc, per_sequence


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-manifest", type=Path, required=True)
    p.add_argument("--v48-config", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ensemble-validation", type=Path, required=True)
    p.add_argument("--adapted-checkpoint", action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists; use --force")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_cfg = _construct(ObjectEventV427Config, cast(Mapping[str, Any], raw["model"]))
    loss_cfg = _construct(ObjectEventV427LossConfig, cast(Mapping[str, Any], raw["loss"]))
    train_cfg = cast(Mapping[str, Any], raw["train"])
    selection = cast(Mapping[str, Any], raw["selection"])
    final_decision = cast(Mapping[str, Any], raw["final_decision"])
    checkpoints = _parse_seed_paths(args.adapted_checkpoint)
    device = _resolve_device(args.device)
    base_cfg, _, _, _, _ = _load_v48_config(args.v48_config)
    train_split, train_manifest = _materialize(args.cache_manifest, "train", input_size=base_cfg.input_size)
    train_frame = _split_frame(train_split)
    folds = _sequence_folds(train_frame["sequence_id"].to_numpy(dtype=object), int(train_cfg["fold_count"]), int(train_cfg["seed"]))
    n = len(train_frame)
    per_seed_predictions, per_seed_lhr = [], []
    fold_records: list[dict[str, Any]] = []
    hist_dir = args.output_dir / "cv_histories"
    hist_dir.mkdir()
    all_idx = np.arange(n, dtype=np.int64)
    for seed in [int(x) for x in train_cfg["seeds"]]:
        pred = np.full(n, np.nan, dtype=np.float64)
        lhr = np.full(n, np.nan, dtype=np.float64)
        for fold_id, held_idx in enumerate(folds):
            print(f"[v4.27] seed={seed} fold={fold_id + 1}/{len(folds)} training", flush=True)
            fit_idx = np.setdiff1d(all_idx, held_idx, assume_unique=True)
            model, history = _train_model(
                checkpoints[seed], _subset_split(train_split, fit_idx), v48_config=args.v48_config,
                model_cfg=model_cfg, loss_cfg=loss_cfg, train_cfg=train_cfg, device=device,
                seed=int(train_cfg["seed"]) + seed * 100 + fold_id, epochs=int(train_cfg["epochs"]),
            )
            held_split = _subset_split(train_split, held_idx)
            fold_pred, fold_lhr, fold_entropy = _predict(model, held_split, batch_size=int(train_cfg["batch_size"]), device=device)
            pred[held_idx], lhr[held_idx] = fold_pred, fold_lhr
            local_frame = train_frame.iloc[held_idx].reset_index(drop=True)
            local_metrics, _ = _metrics_bundle(local_frame, held_split, fold_pred, fold_lhr)
            fold_records.append({
                "seed": seed, "fold": fold_id,
                "held_out_sequences": sorted(local_frame["sequence_id"].unique().tolist()),
                "metrics": local_metrics, "entropy_mean": float(np.mean(fold_entropy)),
                "final_history": history[-1],
            })
            pd.DataFrame(history).to_csv(hist_dir / f"seed{seed}_fold{fold_id}.csv", index=False)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not (np.isfinite(pred).all() and np.isfinite(lhr).all()):
            raise RuntimeError(f"incomplete OOF coverage for seed {seed}")
        per_seed_predictions.append(pred)
        per_seed_lhr.append(lhr)
    oof_pred = np.median(np.stack(per_seed_predictions), axis=0)
    oof_lhr = np.median(np.stack(per_seed_lhr), axis=0)
    oof_metrics, oof_per_sequence = _metrics_bundle(train_frame, train_split, oof_pred, oof_lhr)
    # The v4.26 OOF audit used eight negative samples as the per-track threshold,
    # but no train track reached it after the signed subset construction, making
    # that part of the selector inert. Four retains enough samples to be useful
    # while producing a non-empty negative-track audit on the current eAP train split.
    oof_tracks, oof_per_track = track_metrics(
        train_frame, oof_pred, minimum_track_samples=8, minimum_negative_track_samples=4
    )
    oof_pass = (
        float(oof_metrics["pearson"]) >= float(selection["minimum_oof_pearson"])
        and float(oof_metrics["negative_accuracy"]) >= float(selection["minimum_oof_negative_accuracy"])
        and float(oof_metrics["balanced_sign_accuracy"]) >= float(selection["minimum_oof_balanced_sign"])
        and float(oof_metrics["log_eta_pearson"]) >= float(selection["minimum_oof_log_eta_pearson"])
        and float(oof_tracks["negative_track_macro_accuracy"])
        >= float(selection["minimum_oof_negative_track_macro_accuracy"])
    )
    train_frame.assign(
        v427_oof_prediction_expansion=oof_pred,
        v427_oof_predicted_log_eta=oof_lhr,
    ).to_csv(args.output_dir / "oof_train_predictions.csv", index=False)
    oof_per_sequence.to_csv(args.output_dir / "oof_train_per_sequence.csv", index=False)
    oof_per_track.to_csv(args.output_dir / "oof_train_per_track.csv", index=False)

    # Validation is a scarce development resource.  Do not even materialize it
    # unless the architecture passes the entirely train-derived grouped-OOF gate.
    if not oof_pass:
        recommendation = (
            "scale_correlation_lhr_oof_failed_do_not_open_validation_"
            "move_to_multiscale_2d_event_correlation_v428"
        )
        summary = {
            "artifact_type": "object_event_v4_27_scale_correlation_lhr",
            "status": "completed_oof_gate_failed",
            "elapsed_seconds": time.perf_counter() - started,
            "oof_train_metrics": oof_metrics,
            "oof_track_metrics": oof_tracks,
            "oof_gate_passed": False,
            "decision": {"recommendation": recommendation, "strong_gain": False},
            "scientific_contract": {
                "grouped_sequence_oof": True,
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
        print(json.dumps(_json_safe(summary), indent=2))
        return 0

    # Only now is development-validation opened, once, for the fixed candidate.
    val_split, validation_manifest = _materialize(
        args.cache_manifest, "validation", input_size=base_cfg.input_size
    )
    val_frame = _split_frame(val_split)

    final_predictions, final_lhr = [], []
    final_records = []
    for seed in [int(x) for x in train_cfg["seeds"]]:
        print(f"[v4.27] final full-train seed={seed}", flush=True)
        model, history = _train_model(
            checkpoints[seed], train_split, v48_config=args.v48_config, model_cfg=model_cfg,
            loss_cfg=loss_cfg, train_cfg=train_cfg, device=device, seed=int(train_cfg["seed"]) + seed * 1000,
            epochs=int(train_cfg["final_epochs"]),
        )
        prediction, lhr, entropy = _predict(model, val_split, batch_size=int(train_cfg["batch_size"]), device=device)
        final_predictions.append(prediction)
        final_lhr.append(lhr)
        checkpoint_path = args.output_dir / f"v427_scale_corr_seed_{seed}.pt"
        torch.save({"model_state_dict": model.state_dict(), "seed": seed, "config": raw}, checkpoint_path)
        final_records.append({"seed": seed, "entropy_mean": float(np.mean(entropy)), "final_history": history[-1], "checkpoint": str(checkpoint_path)})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    val_pred = np.median(np.stack(final_predictions), axis=0)
    val_lhr = np.median(np.stack(final_lhr), axis=0)
    val_metrics, val_per_sequence = _metrics_bundle(val_frame, val_split, val_pred, val_lhr)
    val_tracks, val_per_track = track_metrics(
        val_frame, val_pred, minimum_track_samples=8, minimum_negative_track_samples=4
    )

    # Align the sealed v4.10 development-validation benchmark through the same
    # split-aware helper used by the earlier object-event experiments.  Do not
    # fit or calibrate anything on validation here.
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
        recommendation = "scale_correlation_lhr_supported_lock_v427_then_long_multiseed"
    elif oof_pass and float(val_metrics["log_eta_pearson"]) > 0.35:
        recommendation = "scale_correlation_lhr_transfers_but_ttc_not_yet_sota_refine_scale_matcher_v428"
    else:
        recommendation = "scale_correlation_lhr_insufficient_move_to_multiscale_2d_event_correlation_v428"

    val_out = val_frame.copy()
    val_out["v410_prediction_expansion"] = v410_pred
    val_out["v427_prediction_expansion"] = val_pred
    val_out["v427_predicted_log_eta"] = val_lhr
    val_out.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    val_per_sequence.to_csv(args.output_dir / "validation_per_sequence.csv", index=False)
    val_per_track.to_csv(args.output_dir / "validation_per_track.csv", index=False)
    pd.DataFrame(fold_records).to_json(args.output_dir / "fold_records.jsonl", orient="records", lines=True)
    summary = {
        "artifact_type": "object_event_v4_27_scale_correlation_lhr",
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "oof_train_metrics": oof_metrics,
        "oof_track_metrics": oof_tracks,
        "oof_gate_passed": bool(oof_pass),
        "v410_validation_metrics": v410_metrics,
        "v410_validation_track_metrics": v410_tracks,
        "validation_metrics": val_metrics,
        "validation_track_metrics": val_tracks,
        "decision": {
            "recommendation": recommendation,
            "strong_gain": bool(strong),
            "comparisons": {
                "v410_pearson": float(v410_metrics["pearson"]),
                "v427_pearson": float(val_metrics["pearson"]),
                "v410_negative_accuracy": float(v410_metrics["negative_accuracy"]),
                "v427_negative_accuracy": float(val_metrics["negative_accuracy"]),
                "v410_balanced_sign": float(v410_metrics["balanced_sign_accuracy"]),
                "v427_balanced_sign": float(val_metrics["balanced_sign_accuracy"]),
                "v410_negative_track_macro_accuracy": float(v410_tracks["negative_track_macro_accuracy"]),
                "v427_negative_track_macro_accuracy": float(val_tracks["negative_track_macro_accuracy"]),
            },
        },
        "final_seed_records": final_records,
        "scientific_contract": {
            "grouped_sequence_oof": True,
            "height_labels_training_only": True,
            "boxes_not_forward_features": True,
            "validation_evaluated_once_after_fixed_architecture": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "config": raw,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe({"status": "completed", "recommendation": recommendation, "oof": oof_metrics, "validation": val_metrics}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
