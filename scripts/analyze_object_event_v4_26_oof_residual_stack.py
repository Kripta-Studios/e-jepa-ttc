#!/usr/bin/env python3
"""Leak-free grouped OOF residual geometry stack for Object Event TTC v4.26."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.analyze_object_event_v4_22_encoder_geometry import _score_backbone  # noqa: E402
from scripts.analyze_object_event_v4_24_orchestrator import (  # noqa: E402
    _combine_seed_results,
    _evaluate_arm_cv,
    _load_config as _load_v424_config,
    _sequence_folds,
)
from scripts.analyze_object_event_v4_25_geometry_readout import _load_champion  # noqa: E402
from scripts.train_e_jepa_object_event_v4_6 import _materialize  # noqa: E402
from scripts.train_e_jepa_object_event_v4_8 import _load_config as _load_v48_config  # noqa: E402
from scripts.train_e_jepa_object_event_v4_12 import _align_ensemble, _read_ensemble  # noqa: E402
from scripts.train_e_jepa_object_event_v4_16 import _json_safe, _metrics, _resolve_device  # noqa: E402
from e_jepa_ttc.training.object_event_v4_26 import (  # noqa: E402
    ResidualSpec,
    apply_residual_calibration,
    fit_residual_calibration,
    nonnegative_ridge_residual,
    predict_anchored_residual,
    residual_design_matrix,
    track_metrics,
)


def _parse_seed_paths(values: list[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for item in values:
        seed_text, path_text = item.split("=", 1)
        out[int(seed_text)] = Path(path_text)
    if sorted(out) != [7, 13, 23]:
        raise ValueError("exact seeds 7,13,23 required")
    return dict(sorted(out.items()))


@torch.no_grad()
def _predict_backbone(backbone: Any, split: Any, *, batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[torch.Tensor] = []
    backbone.eval()
    for start in range(0, len(split.events), batch_size):
        events = split.events[start:start + batch_size].to(device=device, dtype=torch.float32)
        chunks.append(backbone(events).expansion.detach().float().cpu())
    return torch.cat(chunks).numpy().astype(np.float64)


def _score_full_champions(
    champions: dict[int, Path],
    split: Any,
    *,
    v48_config: Path,
    model_config: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    predictions: list[np.ndarray] = []
    divergences: list[np.ndarray] = []
    verticals: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    for seed, path in champions.items():
        print(f"[v4.26] scoring full-train champion seed={seed}", flush=True)
        backbone = _load_champion(v48_config, path, device)
        predictions.append(_predict_backbone(backbone, split, batch_size=batch_size, device=device))
        div, vert, diag = _score_backbone(
            backbone,
            split,
            batch_size=batch_size,
            config=model_config,
            device=device,
        )
        divergences.append(np.asarray(div, dtype=np.float64))
        verticals.append(np.asarray(vert, dtype=np.float64))
        diagnostics.append({"seed": float(seed), **{k: float(v) for k, v in diag.items()}})
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (
        np.median(np.stack(predictions), axis=0),
        np.median(np.stack(divergences), axis=0),
        np.median(np.stack(verticals), axis=0),
        diagnostics,
    )


def _candidate_objective(
    metrics: dict[str, Any],
    tracks: dict[str, float | int],
    selection: dict[str, float],
) -> tuple[bool, float]:
    target_std = max(float(metrics["target_std"]), 1.0e-8)
    eligible = (
        float(metrics["positive_accuracy"]) >= selection["minimum_positive_accuracy"]
        and float(metrics["negative_accuracy"]) >= selection["minimum_negative_accuracy"]
        and float(metrics["minimum_sequence_pearson"]) >= selection["minimum_sequence_pearson"]
        and float(tracks["track_macro_pearson"]) >= selection["minimum_track_macro_pearson"]
    )
    objective = (
        selection["pearson_weight"] * float(metrics["pearson"])
        + selection["balanced_sign_weight"] * float(metrics["balanced_sign_accuracy"])
        + selection["minimum_sequence_pearson_weight"] * float(metrics["minimum_sequence_pearson"])
        + selection["minimum_sequence_negative_weight"] * float(metrics["minimum_sequence_negative_accuracy"])
        + selection["track_macro_pearson_weight"] * float(tracks["track_macro_pearson"])
        + selection["minimum_negative_track_weight"] * float(tracks["minimum_negative_track_accuracy"])
        - selection["normalized_mae_penalty"] * float(metrics["expansion_mae"]) / target_std
    )
    return bool(eligible), float(objective)


def _fit_residual_candidate(
    spec: ResidualSpec,
    anchor: np.ndarray,
    div_raw: np.ndarray,
    vert_raw: np.ndarray,
    target: np.ndarray,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not spec.features:
        return np.asarray(anchor[eval_idx], dtype=np.float64), {
            "features": [],
            "coefficients": [],
            "divergence_calibration": None,
            "vertical_calibration": None,
        }
    residual = target - anchor
    div_cal = fit_residual_calibration(div_raw[fit_idx], residual[fit_idx])
    vert_cal = fit_residual_calibration(vert_raw[fit_idx], residual[fit_idx])
    div_fit = apply_residual_calibration(div_raw[fit_idx], div_cal)
    div_eval = apply_residual_calibration(div_raw[eval_idx], div_cal)
    vert_fit = apply_residual_calibration(vert_raw[fit_idx], vert_cal)
    vert_eval = apply_residual_calibration(vert_raw[eval_idx], vert_cal)
    x_fit, names = residual_design_matrix(div_fit, vert_fit, spec.features)
    x_eval, _ = residual_design_matrix(div_eval, vert_eval, spec.features)
    coeff = nonnegative_ridge_residual(x_fit, residual[fit_idx], ridge=spec.ridge)
    pred = predict_anchored_residual(anchor[eval_idx], x_eval, coeff)
    return pred, {
        "features": list(names),
        "coefficients": coeff.tolist(),
        "divergence_calibration": asdict(div_cal),
        "vertical_calibration": asdict(vert_cal),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--v424-config", type=Path, required=True)
    parser.add_argument("--v424-summary", type=Path, required=True)
    parser.add_argument("--v425-summary", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--champion-checkpoint", action="append", required=True)
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

    raw = yaml.safe_load(
        (ROOT / "configs/experiment/e_jepa_garl_object_event_oof_residual_stack_v4_26.yaml").read_text(
            encoding="utf-8"
        )
    )
    meta = dict(raw["meta"])
    selection = {str(k): float(v) for k, v in dict(raw["selection"]).items()}
    decision_cfg = {str(k): float(v) for k, v in dict(raw["final_decision"]).items()}
    specs = [
        ResidualSpec(str(row["name"]), tuple(row["features"]), float(row["ridge"]))
        for row in raw["residuals"]
    ]

    model_config, orch, arms, _, geometry_loss_config, _ = _load_v424_config(args.v424_config)
    arm = arms["geometry_only_regularized"]
    adapted = _parse_seed_paths(args.adapted_checkpoint)
    champions = _parse_seed_paths(args.champion_checkpoint)
    device = _resolve_device(args.device)
    base_config, _, _, _, _ = _load_v48_config(args.v48_config)
    train_split, train_manifest = _materialize(args.cache_manifest, "train", input_size=base_config.input_size)
    val_split, val_manifest = _materialize(args.cache_manifest, "validation", input_size=base_config.input_size)
    train_frame = _align_ensemble(train_split, _read_ensemble(args.ensemble_train))
    val_frame = _align_ensemble(val_split, _read_ensemble(args.ensemble_validation))

    # First-level CV. Critically, TTC anchor, divergence and vertical scale all
    # come from the same held-sequence-out model family.
    folds = _sequence_folds(
        train_frame["sequence_id"].astype(str).to_numpy(),
        int(meta["fold_count"]),
        int(meta["seed"]),
    )
    per_seed = []
    for seed in (7, 13, 23):
        print(f"[v4.26] cross-fitting anchor+geometry seed={seed} over {len(folds)} folds", flush=True)
        per_seed.append(
            _evaluate_arm_cv(
                "geometry_only_regularized",
                arm,
                seed,
                checkpoint=adapted[seed],
                split=train_split,
                frame=train_frame,
                folds=folds,
                v48_config=args.v48_config,
                model_config=model_config,
                geometry_loss_config=geometry_loss_config,
                orch=orch,
                device=device,
                output_dir=args.output_dir,
            )
        )
    first_level = _combine_seed_results("geometry_only_regularized", per_seed, train_frame)
    oof_anchor = np.asarray(first_level.ttc_prediction, dtype=np.float64)
    oof_div = np.asarray(first_level.divergence, dtype=np.float64)
    oof_vert = np.asarray(first_level.vertical, dtype=np.float64)
    target = train_frame["target_expansion"].to_numpy(dtype=np.float64)

    # Second-level grouped meta-CV uses only first-level OOF predictions.
    all_idx = np.arange(len(train_frame), dtype=np.int64)
    ranking: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    oof_predictions: dict[str, np.ndarray] = {}
    for spec in specs:
        pred = np.full(len(train_frame), np.nan, dtype=np.float64)
        local_records: list[dict[str, Any]] = []
        for fold_id, held_idx in enumerate(folds):
            fit_idx = np.setdiff1d(all_idx, held_idx, assume_unique=True)
            fold_pred, record = _fit_residual_candidate(
                spec, oof_anchor, oof_div, oof_vert, target, fit_idx, held_idx
            )
            pred[held_idx] = fold_pred
            record.update({
                "readout": spec.name,
                "fold": fold_id,
                "held_out_sequences": sorted(
                    train_frame.iloc[held_idx]["sequence_id"].astype(str).unique().tolist()
                ),
            })
            local_records.append(record)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"incomplete meta-OOF coverage for {spec.name}")
        metrics, _ = _metrics(train_frame, pred, minimum_negatives=20)
        tracks, _ = track_metrics(
            train_frame,
            pred,
            minimum_track_samples=int(meta["minimum_track_samples"]),
            minimum_negative_track_samples=int(meta["minimum_negative_track_samples"]),
        )
        eligible, objective = _candidate_objective(metrics, tracks, selection)
        ranking.append({
            "readout": spec.name,
            "eligible": eligible,
            "objective": objective,
            **{k: metrics[k] for k in (
                "pearson", "expansion_mae", "positive_accuracy", "negative_accuracy",
                "balanced_sign_accuracy", "minimum_sequence_pearson",
                "minimum_sequence_negative_accuracy",
            )},
            **tracks,
        })
        oof_predictions[spec.name] = pred
        fold_records.extend(local_records)

    ranking.sort(key=lambda row: (bool(row["eligible"]), float(row["objective"])), reverse=True)
    control = next(row for row in ranking if row["readout"] == "anchor_control")
    winner = ranking[0]
    minimum_gain = float(meta["minimum_objective_gain_over_control"])
    if winner["readout"] != "anchor_control" and float(winner["objective"]) < float(control["objective"]) + minimum_gain:
        winner = control
    chosen = next(spec for spec in specs if spec.name == winner["readout"])
    print(f"[v4.26] selected residual={chosen.name} objective={winner['objective']:.4f}", flush=True)

    # Fit the final residual mapping on the first-level OOF train predictions,
    # then apply it once to full-train champions on development validation.
    residual_target = target - oof_anchor
    div_cal = fit_residual_calibration(oof_div, residual_target)
    vert_cal = fit_residual_calibration(oof_vert, residual_target)
    oof_div_res = apply_residual_calibration(oof_div, div_cal)
    oof_vert_res = apply_residual_calibration(oof_vert, vert_cal)
    x_oof, selected_features = residual_design_matrix(oof_div_res, oof_vert_res, chosen.features)
    if chosen.features:
        coeff = nonnegative_ridge_residual(x_oof, residual_target, ridge=chosen.ridge)
    else:
        coeff = np.zeros(0, dtype=np.float64)
    final_oof_pred = predict_anchored_residual(oof_anchor, x_oof, coeff)
    oof_metrics, oof_per_seq = _metrics(train_frame, final_oof_pred, minimum_negatives=20)
    oof_track_metrics, oof_per_track = track_metrics(
        train_frame,
        final_oof_pred,
        minimum_track_samples=int(meta["minimum_track_samples"]),
        minimum_negative_track_samples=int(meta["minimum_negative_track_samples"]),
    )

    full_train_anchor, full_train_div, full_train_vert, train_diag = _score_full_champions(
        champions,
        train_split,
        v48_config=args.v48_config,
        model_config=model_config,
        batch_size=orch.batch_size,
        device=device,
    )
    val_anchor, val_div, val_vert, val_diag = _score_full_champions(
        champions,
        val_split,
        v48_config=args.v48_config,
        model_config=model_config,
        batch_size=orch.batch_size,
        device=device,
    )
    val_div_res = apply_residual_calibration(val_div, div_cal)
    val_vert_res = apply_residual_calibration(val_vert, vert_cal)
    x_val, _ = residual_design_matrix(val_div_res, val_vert_res, chosen.features)
    val_pred = predict_anchored_residual(val_anchor, x_val, coeff)

    val_metrics, val_per_seq = _metrics(val_frame, val_pred, minimum_negatives=20)
    anchor_metrics, _ = _metrics(val_frame, val_anchor, minimum_negatives=20)
    v410 = val_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    v410_metrics, _ = _metrics(val_frame, v410, minimum_negatives=20)
    val_track_metrics, val_per_track = track_metrics(
        val_frame,
        val_pred,
        minimum_track_samples=int(meta["minimum_track_samples"]),
        minimum_negative_track_samples=int(meta["minimum_negative_track_samples"]),
    )
    v410_track_metrics, _ = track_metrics(
        val_frame,
        v410,
        minimum_track_samples=int(meta["minimum_track_samples"]),
        minimum_negative_track_samples=int(meta["minimum_negative_track_samples"]),
    )

    if chosen.name == "anchor_control":
        recommendation = "oof_residual_not_selected_proceed_explicit_lhr_v427"
    elif (
        float(val_metrics["pearson"]) >= float(v410_metrics["pearson"]) + decision_cfg["minimum_pearson_gain_over_v410"]
        and float(val_metrics["negative_accuracy"]) >= float(v410_metrics["negative_accuracy"]) + decision_cfg["minimum_negative_accuracy_gain_over_v410"]
        and float(val_metrics["balanced_sign_accuracy"]) >= float(v410_metrics["balanced_sign_accuracy"]) + decision_cfg["minimum_balanced_sign_gain_over_v410"]
    ):
        recommendation = "oof_residual_supported_lock_architecture_then_long_multiseed"
    elif (
        float(val_metrics["pearson"]) >= float(v410_metrics["pearson"]) - decision_cfg["tradeoff_pearson_tolerance"]
        and float(val_metrics["negative_accuracy"]) > float(v410_metrics["negative_accuracy"])
    ):
        recommendation = "oof_residual_tradeoff_promising_do_not_open_test_confirm_long_multiseed"
    else:
        recommendation = "oof_residual_insufficient_proceed_explicit_lhr_v427"

    pd.DataFrame(ranking).to_csv(args.output_dir / "train_only_residual_ranking.csv", index=False)
    pd.DataFrame(fold_records).to_json(
        args.output_dir / "meta_fold_coefficients.jsonl", orient="records", lines=True
    )
    oof_out = train_frame.loc[:, ["sequence_id", "sample_token", "track_id", "target_expansion"]].copy()
    oof_out["oof_anchor_prediction_expansion"] = oof_anchor
    oof_out["v426_oof_prediction_expansion"] = final_oof_pred
    oof_out["oof_divergence_raw"] = oof_div
    oof_out["oof_vertical_raw"] = oof_vert
    oof_out.to_csv(args.output_dir / "oof_train_predictions.csv", index=False)
    oof_per_seq.to_csv(args.output_dir / "oof_train_per_sequence.csv", index=False)
    oof_per_track.to_csv(args.output_dir / "oof_train_per_track.csv", index=False)

    val_out = val_frame.loc[:, ["sequence_id", "sample_token", "track_id", "target_expansion"]].copy()
    val_out["v410_baseline_prediction_expansion"] = v410
    val_out["v424_anchor_prediction_expansion"] = val_anchor
    val_out["v426_prediction_expansion"] = val_pred
    val_out["v426_divergence_residual_proxy"] = val_div_res
    val_out["v426_vertical_residual_proxy"] = val_vert_res
    val_out.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    val_per_seq.to_csv(args.output_dir / "validation_per_sequence.csv", index=False)
    val_per_track.to_csv(args.output_dir / "validation_per_track.csv", index=False)

    summary = {
        "artifact_type": "object_event_v4_26_leak_free_oof_residual_stack",
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "selected_residual": chosen.name,
        "selected_features": list(selected_features),
        "selected_ridge": chosen.ridge,
        "selected_coefficients": coeff.tolist(),
        "oof_residual_calibration": {
            "divergence": asdict(div_cal),
            "vertical": asdict(vert_cal),
        },
        "train_only_ranking": ranking,
        "oof_train_metrics": oof_metrics,
        "oof_train_track_metrics": oof_track_metrics,
        "v410_validation_metrics": v410_metrics,
        "v410_validation_track_metrics": v410_track_metrics,
        "v424_anchor_validation_metrics": anchor_metrics,
        "validation_metrics": val_metrics,
        "validation_track_metrics": val_track_metrics,
        "diagnostics": {
            "full_train_seed_scoring": train_diag,
            "validation_seed_scoring": val_diag,
            "full_train_anchor_mean": float(np.mean(full_train_anchor)),
        },
        "decision": {
            "recommendation": recommendation,
            "comparisons": {
                "v410_pearson": v410_metrics["pearson"],
                "v426_pearson": val_metrics["pearson"],
                "v410_negative_accuracy": v410_metrics["negative_accuracy"],
                "v426_negative_accuracy": val_metrics["negative_accuracy"],
                "v410_balanced_sign": v410_metrics["balanced_sign_accuracy"],
                "v426_balanced_sign": val_metrics["balanced_sign_accuracy"],
                "v410_minimum_sequence_negative_accuracy": v410_metrics["minimum_sequence_negative_accuracy"],
                "v426_minimum_sequence_negative_accuracy": val_metrics["minimum_sequence_negative_accuracy"],
                "v410_minimum_negative_track_accuracy": v410_track_metrics["minimum_negative_track_accuracy"],
                "v426_minimum_negative_track_accuracy": val_track_metrics["minimum_negative_track_accuracy"],
            },
        },
        "scientific_contract": {
            "first_level_anchor_is_grouped_oof": True,
            "first_level_geometry_is_grouped_oof": True,
            "anchor_and_geometry_same_model_family_and_folds": True,
            "meta_selection_uses_only_oof_train_predictions": True,
            "v410_in_sample_train_anchor_not_used_for_selection": True,
            "anchor_coefficient_fixed_exactly_one": True,
            "geometry_residual_coefficients_nonnegative": True,
            "track_metrics_are_diagnostic_and_train_selection_safe": True,
            "validation_evaluated_once_after_selection": True,
            "boxes_not_forward_features": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
        "train_manifest": train_manifest,
        "validation_manifest": val_manifest,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(_json_safe({
        "status": "completed",
        "selected_residual": chosen.name,
        "coefficients": coeff.tolist(),
        "v410_validation_metrics": v410_metrics,
        "v424_anchor_validation_metrics": anchor_metrics,
        "validation_metrics": val_metrics,
        "validation_track_metrics": val_track_metrics,
        "decision": summary["decision"],
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
