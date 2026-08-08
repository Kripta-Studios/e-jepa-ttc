#!/usr/bin/env python3
"""Preregistered v4.29 train-only OOF analyzer.

This entry point intentionally exposes no eAP official-test or EvTTC option.
Development validation is not materialised unless the fixed median all-seed OOF
champion passes every gate; the implementation below fails closed before that
point if prediction coverage or affine diagnostics are incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
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

from e_jepa_ttc.models.object_event_v4_22 import configure_partial_geometry_unfreeze  # noqa: E402
from e_jepa_ttc.models.object_event_v4_28 import (  # noqa: E402
    ObjectEventTTCV428,
    ObjectEventV428Config,
)
from e_jepa_ttc.models.object_event_v4_29 import (  # noqa: E402
    ObjectEventTTCV429,
    ObjectEventV429Config,
)
from e_jepa_ttc.training.object_event_v4_22 import relative_parameter_anchor  # noqa: E402
from e_jepa_ttc.training.object_event_v4_26 import track_metrics  # noqa: E402
from e_jepa_ttc.training.object_event_v4_27 import target_log_height_ratio  # noqa: E402
from e_jepa_ttc.training.object_event_v4_28 import (  # noqa: E402
    ObjectEventV428LossConfig,
    object_event_v4_28_loss,
)
from e_jepa_ttc.training.object_event_v4_29 import (  # noqa: E402
    ObjectEventV429LossConfig,
    object_event_v4_29_loss,
    oof_gates,
    seed_dominance,
)
from scripts.analyze_object_event_v4_24_orchestrator import (  # noqa: E402
    _sequence_folds,
    _subset_split,
)
from scripts.analyze_object_event_v4_28_multiscale_posterior import (  # noqa: E402
    _metrics,
    _resolve_device,
)
from scripts.preflight_object_event_v4_29 import (  # noqa: E402
    parse_seed_paths,
    validate_checkpoints,
    validate_config,
)
from scripts.train_e_jepa_object_event_v4_6 import (  # noqa: E402
    MaterializedV46Split,
    _materialize,
)
from scripts.train_e_jepa_object_event_v4_8 import _load_config as _load_v48_config  # noqa: E402
from scripts.train_e_jepa_object_event_v4_12 import (  # noqa: E402
    _align_ensemble,
    _load_backbone,
    _read_ensemble,
)


def _sha256(path: Path) -> str:
    cached = _CHECKPOINT_HASH_CACHE.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    value = digest.hexdigest()
    _CHECKPOINT_HASH_CACHE[path] = value
    return value


_CHECKPOINT_HASH_CACHE: dict[Path, str] = {}


def isolate_rng(seed: int) -> None:
    """Set all stochastic sources used by construction/optimization; workers are zero."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _frame(split: MaterializedV46Split) -> pd.DataFrame:
    target = (
        (split.delta_t_s / split.target_ttc_s).clamp(-0.25, 0.25).cpu().numpy().astype(np.float64)
    )
    return pd.DataFrame(
        {
            "sequence_id": list(map(str, split.sequence_ids)),
            "sample_token": list(map(str, split.sample_tokens)),
            "track_id": list(map(str, split.track_ids)),
            "target_expansion": target,
            "delta_t_s": split.delta_t_s.cpu().numpy().astype(np.float64),
            "target_ttc_s": split.target_ttc_s.cpu().numpy().astype(np.float64),
        }
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Finite JSON-safe Pearson; constant/short samples have no correlation signal."""
    if (
        len(a) < 2
        or not (np.isfinite(a).all() and np.isfinite(b).all())
        or np.std(a) <= 1e-12
        or np.std(b) <= 1e-12
    ):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def calibration_slopes(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return prediction-on-target slopes: intercept-inclusive then zero-intercept."""
    centered_target = target - target.mean()
    centered_prediction = prediction - prediction.mean()
    intercept = float(
        np.dot(centered_target, centered_prediction)
        / max(np.dot(centered_target, centered_target), 1e-12)
    )
    zero = float(np.dot(target, prediction) / max(np.dot(target, target), 1e-12))
    return intercept, zero


def choose_champion(results: Mapping[str, Mapping[str, Mapping[str, float]]]) -> str:
    """Fixed objective/tie order, including lexically ascending final tie break."""

    def key(name: str) -> tuple[float, float, float, str]:
        metric = results[name]["oof_metrics"]
        return (
            -float(metric.get("pearson", -np.inf)),
            -float(metric.get("minimum_sequence_pearson", -np.inf)),
            -float(metric.get("negative_accuracy", -np.inf)),
            name,
        )

    return sorted(results, key=key)[0]


def factorial_effect_table(cells: list[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    """Paired-fold 3x3 fixed-effect descriptive table with interaction diagnostics."""
    grid = {
        (int(c["backbone_checkpoint_seed"]), int(c["matcher_init_seed"])): float(
            c["cell_metrics"][metric]
        )
        for c in cells
    }
    grand = float(np.mean(list(grid.values())))
    backbone = {
        seed: float(np.mean([grid[(seed, m)] for m in (7, 13, 23)])) for seed in (7, 13, 23)
    }
    matcher = {seed: float(np.mean([grid[(b, seed)] for b in (7, 13, 23)])) for seed in (7, 13, 23)}
    interaction = {
        f"b{b}_m{m}": grid[(b, m)] - backbone[b] - matcher[m] + grand
        for b in (7, 13, 23)
        for m in (7, 13, 23)
    }
    values = list(interaction.values())
    signs = {np.sign(x) for x in values if abs(x) >= 0.03}
    return {
        "metric": metric,
        "grand_mean": grand,
        "backbone_level_means": backbone,
        "matcher_level_means": matcher,
        "backbone_marginal_range": max(backbone.values()) - min(backbone.values()),
        "matcher_marginal_range": max(matcher.values()) - min(matcher.values()),
        "interaction_by_cell": interaction,
        "interaction_range": max(values) - min(values),
        "interaction_max_abs": max(map(abs, values)),
        "crossover_indicator": len(signs) > 1,
    }


def _seed_attribution(
    train_split: MaterializedV46Split,
    frame: pd.DataFrame,
    folds: list[np.ndarray],
    *,
    checkpoints: Mapping[int, Path],
    v48_config: Path,
    train: Mapping[str, Any],
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    """Run the fixed 3x3x3 v4.28-profile attribution before v4.29 arms.

    The checkpoint determines backbone provenance; ``matcher_init_seed`` is set
    immediately before constructing the profile matcher. After construction every
    RNG and the no-worker sampler are reset to a fold-only optimization seed.
    """
    model_cfg = ObjectEventV428Config(
        matcher="profile",
        correlation_dim=48,
        log_scale_min=-0.22,
        log_scale_max=0.22,
        scale_bins=45,
        correlation_temperature=0.04,
        foreground_floor=0.05,
        activity_floor=0.05,
        batch_size=8,
    )
    loss_cfg = ObjectEventV428LossConfig()
    allidx = np.arange(len(frame))
    rows = []
    history_dir = output / "seed_attribution_histories"
    history_dir.mkdir()
    for backbone_seed in (7, 13, 23):
        for matcher_seed in (7, 13, 23):
            prediction = np.full(len(frame), np.nan)
            logeta = np.full(len(frame), np.nan)
            for fold, held in enumerate(folds):
                isolate_rng(matcher_seed)
                backbone, _ = _load_backbone(
                    v48_config_path=v48_config, checkpoint_path=checkpoints[backbone_seed]
                )
                selected = configure_partial_geometry_unfreeze(
                    backbone, int(train["geometry_tail_tensors"])
                )
                model = ObjectEventTTCV428(backbone, model_cfg).to(device)
                opt_seed = int(train["optimization_seed_by_fold"][fold])
                isolate_rng(opt_seed)
                sampler = np.random.default_rng(opt_seed)
                geometry = [
                    p
                    for p in backbone.foreground_model.geometry_encoder.parameters()
                    if p.requires_grad
                ]
                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": model.head_parameters(),
                            "lr": float(train["projection_learning_rate"]),
                        },
                        {"params": geometry, "lr": float(train["geometry_learning_rate"])},
                    ],
                    weight_decay=float(train["weight_decay"]),
                )
                initial = {
                    n: p.detach().clone()
                    for n, p in backbone.foreground_model.geometry_encoder.named_parameters()
                    if p.requires_grad
                }
                hist = []
                fit = np.setdiff1d(allidx, held)
                for epoch in range(int(train["epochs"])):
                    order = fit.copy()
                    sampler.shuffle(order)
                    losses = []
                    for start in range(0, len(order), 8):
                        idx = torch.as_tensor(order[start : start + 8], dtype=torch.long)
                        e = train_split.events[idx].to(device, torch.float32)
                        dt = train_split.delta_t_s[idx].to(device, torch.float32)
                        ttc = train_split.target_ttc_s[idx].to(device, torch.float32)
                        heights = train_split.visible_heights_px[idx].to(device, torch.float32)
                        out = model(e)
                        primary, _ = object_event_v4_28_loss(out, dt, ttc, heights, config=loss_cfg)
                        geometry_encoder = backbone.foreground_model.geometry_encoder
                        anchor = relative_parameter_anchor(
                            {
                                n: p
                                for n, p in geometry_encoder.named_parameters()
                                if p.requires_grad
                            },
                            initial,
                            epsilon=1.0e-6,
                        )
                        total = primary + float(train["geometry_anchor_weight"]) * anchor
                        _finite("attribution_loss", total)
                        optimizer.zero_grad(set_to_none=True)
                        total.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.head_parameters() + geometry,
                            float(train["max_grad_norm"]),
                            error_if_nonfinite=True,
                        )
                        optimizer.step()
                        losses.append(float(total.detach()))
                    hist.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
                with torch.no_grad():
                    for start in range(0, len(held), 8):
                        idx = held[start : start + 8]
                        out = model(train_split.events[idx].to(device, torch.float32))
                        prediction[idx] = out.expansion.cpu().numpy()
                        logeta[idx] = out.predicted_log_eta.cpu().numpy()
                fold_frame = (
                    frame.iloc[held]
                    .reset_index(drop=True)
                    .assign(prediction=prediction[held], predicted_log_eta=logeta[held])
                )
                fold_metrics, _ = _metrics(fold_frame, prediction[held], minimum_negatives=20)
                fold_metrics["log_eta_pearson"] = _pearson(
                    logeta[held],
                    target_log_height_ratio(train_split.visible_heights_px[held]).numpy(),
                )
                fold_frame.to_csv(
                    output
                    / (
                        f"seed_attribution_b{backbone_seed}_m{matcher_seed}_"
                        f"fold{fold}_predictions.csv"
                    ),
                    index=False,
                )
                (
                    history_dir / f"backbone{backbone_seed}_matcher{matcher_seed}_fold{fold}.json"
                ).write_text(json.dumps(hist))
                rows.append(
                    {
                        "backbone_checkpoint_seed": backbone_seed,
                        "matcher_init_seed": matcher_seed,
                        "fold": fold,
                        "optimization_seed": opt_seed,
                        "workers": 0,
                        "checkpoint_sha256": _sha256(checkpoints[backbone_seed]),
                        "trainable_geometry_names": selected,
                        "history": hist,
                        "metrics": fold_metrics,
                    }
                )
            if not (np.isfinite(prediction).all() and np.isfinite(logeta).all()):
                raise RuntimeError("seed attribution OOF coverage incomplete")
            cell_frame = frame.assign(prediction=prediction, predicted_log_eta=logeta)
            cell_frame.to_csv(
                output / f"seed_attribution_predictions_b{backbone_seed}_m{matcher_seed}.csv",
                index=False,
            )
            metrics, _ = _metrics(frame, prediction, minimum_negatives=20)
            target = target_log_height_ratio(train_split.visible_heights_px).numpy()
            rows.append(
                {
                    "backbone_checkpoint_seed": backbone_seed,
                    "matcher_init_seed": matcher_seed,
                    "cell_metrics": {
                        "pearson": metrics["pearson"],
                        "log_eta_pearson": float(np.corrcoef(logeta, target)[0, 1]),
                    },
                }
            )
    cells = [x for x in rows if "cell_metrics" in x]
    pd.DataFrame(cells).to_json(output / "seed_attribution_cells.json", orient="records", indent=2)
    bmeans = {
        seed: float(
            np.mean(
                [
                    x["cell_metrics"]["pearson"]
                    for x in cells
                    if x["backbone_checkpoint_seed"] == seed
                ]
            )
        )
        for seed in (7, 13, 23)
    }
    mmeans = {
        seed: float(
            np.mean([x["cell_metrics"]["pearson"] for x in cells if x["matcher_init_seed"] == seed])
        )
        for seed in (7, 13, 23)
    }
    blhr = {
        seed: float(
            np.mean(
                [
                    x["cell_metrics"]["log_eta_pearson"]
                    for x in cells
                    if x["backbone_checkpoint_seed"] == seed
                ]
            )
        )
        for seed in (7, 13, 23)
    }
    mlhr = {
        seed: float(
            np.mean(
                [
                    x["cell_metrics"]["log_eta_pearson"]
                    for x in cells
                    if x["matcher_init_seed"] == seed
                ]
            )
        )
        for seed in (7, 13, 23)
    }
    pearson_table = factorial_effect_table(cells, "pearson")
    lhr_table = factorial_effect_table(cells, "log_eta_pearson")
    conclusion = (
        "mixed_inconclusive"
        if pearson_table["crossover_indicator"]
        or lhr_table["crossover_indicator"]
        or seed_dominance(mmeans, bmeans) != seed_dominance(mlhr, blhr)
        else seed_dominance(mmeans, bmeans)
    )
    return {
        "seeds": [7, 13, 23],
        "workers": 0,
        "records": rows,
        "pearson_marginal_backbone": bmeans,
        "pearson_marginal_matcher": mmeans,
        "pearson_dominance": seed_dominance(mmeans, bmeans),
        "log_eta_pearson_marginal_backbone": blhr,
        "log_eta_pearson_marginal_matcher": mlhr,
        "log_eta_pearson_dominance": seed_dominance(mlhr, blhr),
        "fixed_effect_tables": [pearson_table, lhr_table],
        "conclusion": conclusion,
        "scope": "limited_to_seeds_7_13_23",
    }


def _finite(name: str, x: torch.Tensor) -> None:
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError(f"non-finite v4.29 {name}")


def _json_safe(value: Any) -> Any:  # noqa: ANN401
    """Convert NumPy/Torch values and non-finite diagnostics to strict JSON."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _run_metadata(
    config_hash: str,
    checkpoint_hashes: Mapping[str, str],
    started: float,
    started_utc: str,
    status: str,
) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unknown"
    return {
        "experiment_id": "object_event_v4_29_local_affine",
        "experiment_name": "local_affine_v4_29",
        "git_commit": commit,
        "config_hash": config_hash,
        "checkpoint_hashes": dict(checkpoint_hashes),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "start_time_utc": started_utc,
        "end_time_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "status": status,
    }


def _train(
    checkpoint: Path,
    split: MaterializedV46Split,
    *,
    base_config: Path,
    model_cfg: ObjectEventV429Config,
    loss_cfg: ObjectEventV429LossConfig,
    train: Mapping[str, Any],
    init_seed: int,
    optimization_seed: int,
    device: torch.device,
    epochs: int,
) -> tuple[ObjectEventTTCV429, list[dict[str, float]], list[str]]:
    """Construct with init seed, then reset all RNGs to the fold-only optimizer seed."""
    isolate_rng(init_seed)
    backbone, _ = _load_backbone(v48_config_path=base_config, checkpoint_path=checkpoint)
    selected = configure_partial_geometry_unfreeze(backbone, int(train["geometry_tail_tensors"]))
    model = ObjectEventTTCV429(backbone, model_cfg).to(device)
    isolate_rng(optimization_seed)
    sampler = np.random.default_rng(optimization_seed)
    geometry = [
        p for p in backbone.foreground_model.geometry_encoder.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.head_parameters(), "lr": float(train["projection_learning_rate"])},
            {"params": geometry, "lr": float(train["geometry_learning_rate"])},
        ],
        weight_decay=float(train["weight_decay"]),
    )
    initial = {
        n: p.detach().clone()
        for n, p in backbone.foreground_model.geometry_encoder.named_parameters()
        if p.requires_grad
    }
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        order = np.arange(len(split.events))
        sampler.shuffle(order)
        values: list[float] = []
        for start in range(0, len(order), model_cfg.batch_size):
            index = torch.as_tensor(order[start : start + model_cfg.batch_size], dtype=torch.long)
            events, dt, ttc, heights = (
                split.events[index].to(device, torch.float32),
                split.delta_t_s[index].to(device, torch.float32),
                split.target_ttc_s[index].to(device, torch.float32),
                split.visible_heights_px[index].to(device, torch.float32),
            )
            output = model(events)
            kwargs: dict[str, Any] = {}
            if loss_cfg.arm == "local_affine_geom_teacher":
                kwargs = {
                    "boxes_xyxy": split.boxes_xyxy[index].to(device, torch.float32),
                    "image_height": int(split.source_height),
                    "image_width": int(split.source_width),
                }
            loss, _ = object_event_v4_29_loss(output, dt, ttc, heights, config=loss_cfg, **kwargs)
            _finite("loss", loss)
            optimizer.zero_grad(set_to_none=True)
            anchor = relative_parameter_anchor(
                {
                    n: p
                    for n, p in backbone.foreground_model.geometry_encoder.named_parameters()
                    if p.requires_grad
                },
                initial,
                epsilon=1.0e-6,
            )
            total = loss + float(train["geometry_anchor_weight"]) * anchor
            _finite("total_loss", total)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.head_parameters() + geometry,
                float(train["max_grad_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            values.append(float(total.detach()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(values))})
    model.eval()
    return model, history, selected


@torch.no_grad()
def _predict(
    model: ObjectEventTTCV429,
    split: MaterializedV46Split,
    device: torch.device,
    events: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    source = split.events if events is None else events
    fields = {
        k: []
        for k in (
            "prediction",
            "log_eta",
            "horizontal",
            "area",
            "valid",
            "det",
            "condition",
            "mass",
            "residual",
            "rotation",
            "shear",
            "translation",
            "boundary",
            "entropy",
            "confidence",
        )
    }
    for start in range(0, len(source), model.config.batch_size):
        o = model(source[start : start + model.config.batch_size].to(device, torch.float32))
        values = {
            "prediction": o.expansion,
            "log_eta": o.predicted_log_eta_vertical,
            "horizontal": o.predicted_log_eta_horizontal,
            "area": o.predicted_log_eta_area,
            "valid": o.affine_12.valid.float(),
            "det": o.affine_12.determinant,
            "condition": o.affine_12.condition_number,
            "mass": o.affine_12.effective_weight_mass,
            "residual": o.affine_12.residual,
            "rotation": o.rotation_radians,
            "shear": o.singular_value_anisotropy,
            "translation": o.translation_magnitude,
            "boundary": o.boundary_probability,
            "entropy": o.correlation_entropy,
            "confidence": o.correlation_confidence,
        }
        for key, value in values.items():
            fields[key].append(value.cpu())
    return {key: torch.cat(value).numpy().astype(np.float64) for key, value in fields.items()}


def aggregate_seed_results(
    per_seed: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Median finite predictions while making any seed failure fail closed.

    Worst-case affine diagnostics are retained on invalid rows so the aggregate
    report records why a constituent seed failed instead of hiding that failure
    behind the other two seeds' median.
    """
    if not per_seed:
        raise ValueError("v4.29 seed aggregation requires at least one seed")
    required = (
        "prediction",
        "log_eta",
        "horizontal",
        "area",
        "det",
        "condition",
        "mass",
        "residual",
    )
    seed_valid = np.stack([result["valid"] > 0.5 for result in per_seed])
    seed_finite = np.stack(
        [
            np.logical_and.reduce([np.isfinite(result[key]) for key in required])
            for result in per_seed
        ]
    )
    aggregate_valid = seed_valid.all(axis=0) & seed_finite.all(axis=0)
    aggregate: dict[str, np.ndarray] = {
        key: np.asarray(np.median(np.stack([result[key] for result in per_seed]), axis=0))
        for key in per_seed[0]
    }
    aggregate["valid"] = np.asarray(aggregate_valid, dtype=np.float64)
    for key in ("prediction", "log_eta", "horizontal", "area"):
        aggregate[key] = np.where(aggregate_valid, aggregate[key], np.nan)
    aggregate["det"] = np.min(np.stack([result["det"] for result in per_seed]), axis=0)
    aggregate["condition"] = np.max(np.stack([result["condition"] for result in per_seed]), axis=0)
    aggregate["mass"] = np.min(np.stack([result["mass"] for result in per_seed]), axis=0)
    aggregate["residual"] = np.max(np.stack([result["residual"] for result in per_seed]), axis=0)
    return aggregate


def _extra_metrics(
    frame: pd.DataFrame,
    split: MaterializedV46Split,
    result: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    finite_required = np.ones_like(result["valid"], dtype=bool)
    for key in (
        "prediction",
        "log_eta",
        "horizontal",
        "area",
        "det",
        "condition",
        "mass",
        "residual",
    ):
        finite_required &= np.isfinite(result[key])
    valid = (result["valid"] > 0.5) & finite_required
    if not valid.all():
        # Never fabricate predictions for invalid fits. The complete OOF gate
        # fails, while valid-only metrics remain available as explicitly
        # labelled diagnostics so the failure can still be investigated.
        finite_prediction = np.isfinite(result["prediction"])
        incomplete: dict[str, Any] = {
            "invalid_affine_fraction": float(1.0 - valid.mean()),
            "complete_finite": False,
            "invalid_rows": int((~valid).sum()),
            "invalid_reason_counts": {
                "nonfinite_prediction": int((~finite_prediction).sum()),
                "nonfinite_log_eta": int((~np.isfinite(result["log_eta"])).sum()),
                "nonfinite_horizontal": int((~np.isfinite(result["horizontal"])).sum()),
                "nonfinite_area": int((~np.isfinite(result["area"])).sum()),
                "nonfinite_determinant": int((~np.isfinite(result["det"])).sum()),
                "nonfinite_condition": int((~np.isfinite(result["condition"])).sum()),
                "nonfinite_effective_mass": int((~np.isfinite(result["mass"])).sum()),
                "nonfinite_residual": int((~np.isfinite(result["residual"])).sum()),
                "determinant_at_or_below_0_05": int((result["det"] <= 0.05).sum()),
                "condition_above_100": int((result["condition"] > 100.0).sum()),
                "effective_mass_below_4": int((result["mass"] < 4.0).sum()),
            },
        }
        if valid.any():
            valid_indices = np.flatnonzero(valid)
            valid_frame = frame.iloc[valid_indices].reset_index(drop=True)
            valid_split = split.subset(torch.as_tensor(valid_indices, dtype=torch.long))
            valid_result = {key: value[valid] for key, value in result.items()}
            incomplete["valid_only_metrics"] = _extra_metrics(
                valid_frame,
                valid_split,
                valid_result,
            )
            incomplete["valid_only_warning"] = (
                "Diagnostic only: incomplete coverage prevents OOF selection or gate passage."
            )
        return incomplete
    prediction, target = result["prediction"], frame.target_expansion.to_numpy(np.float64)
    metrics, _ = _metrics(frame, prediction, minimum_negatives=20)
    lhr = target_log_height_ratio(split.visible_heights_px).numpy().astype(np.float64)
    slope, zero_slope = calibration_slopes(prediction, target)
    metrics.update(
        {
            "log_eta_pearson": _pearson(result["log_eta"], lhr),
            "log_eta_mae": float(np.abs(result["log_eta"] - lhr).mean()),
            "log_eta_area_pearson": _pearson(result["area"], lhr),
            "invalid_affine_fraction": 0.0,
            "complete_finite": True,
            "prediction_std_ratio": float(np.std(prediction) / max(np.std(target), 1e-12)),
            "calibration_slope_zero_intercept": zero_slope,
            "calibration_slope_intercept": slope,
        }
    )
    magnitude = np.abs(target)
    rows = []
    for lo, hi in ((0.01, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, np.inf)):
        mask = (magnitude >= lo) & (magnitude < hi)
        rows.append(
            {
                "bucket": f"[{lo:.2f},{'inf' if np.isinf(hi) else f'{hi:.2f}'})",
                "count": int(mask.sum()),
                "pearson": _pearson(prediction[mask], target[mask]) if mask.any() else None,
                "mae": float(np.abs(prediction[mask] - target[mask]).mean())
                if mask.any()
                else None,
                "mean_abs_gt": float(magnitude[mask].mean()) if mask.any() else None,
                "mean_abs_pred": float(np.abs(prediction[mask]).mean()) if mask.any() else None,
            }
        )
    for row in rows:
        row["magnitude_ratio"] = row["mean_abs_pred"] / row["mean_abs_gt"] if row["count"] else None
    metrics["magnitude_buckets"] = rows
    high = rows[-1]
    metrics["high_magnitude_count"], metrics["high_magnitude_ratio"] = (
        high["count"],
        high["magnitude_ratio"],
    )
    for key in (
        "det",
        "condition",
        "mass",
        "residual",
        "shear",
        "translation",
        "boundary",
        "entropy",
        "confidence",
    ):
        metrics.update(
            {
                key + "_mean": float(np.mean(result[key])),
                key + "_min": float(np.min(result[key])),
                key + "_max": float(np.max(result[key])),
                key + "_p95": float(np.quantile(result[key], 0.95)),
            }
        )
    rotation_abs = np.abs(result["rotation"])
    metrics.update(
        {
            "rotation_abs_mean": float(rotation_abs.mean()),
            "rotation_abs_p95": float(np.quantile(rotation_abs, 0.95)),
            "rotation_abs_max": float(rotation_abs.max()),
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--v48-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--v427-summary", type=Path, required=True)
    parser.add_argument("--v428-summary", type=Path, required=True)
    parser.add_argument("--adapted-checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--v410-summary", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    raw = yaml.safe_load(args.config.read_text())
    validate_config(raw)
    checkpoints = parse_seed_paths(args.adapted_checkpoint)
    validate_checkpoints(checkpoints, args.v48_config)
    if args.output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{args.output_dir} exists; use --force")
        expected_root = (ROOT / "artifacts" / "debug").resolve()
        resolved_output = args.output_dir.resolve()
        if (
            resolved_output.parent != expected_root
            or resolved_output.name != "object_event_v4_29_local_affine"
        ):
            raise ValueError(
                "--force may remove only artifacts/debug/object_event_v4_29_local_affine"
            )
        shutil.rmtree(args.output_dir)
    config_sha256 = _sha256(args.config)
    if sorted(checkpoints) != [7, 13, 23] or sorted(raw["arms"]) != [
        "local_affine_geom_teacher",
        "local_affine_lhr",
    ]:
        raise ValueError("fixed v4.29 seed/arm contract violated")
    v428 = json.loads(args.v428_summary.read_text())
    v427 = json.loads(args.v427_summary.read_text())
    if (
        v428.get("status") != "completed_oof_gate_failed"
        or v427.get("status") != "completed_oof_gate_failed"
    ):
        raise ValueError("sealed failed OOF references required")
    args.output_dir.mkdir(parents=True)
    device = _resolve_device(args.device)
    base, _, _, _, _ = _load_v48_config(args.v48_config)
    train, manifest = _materialize(args.cache_manifest, "train", input_size=base.input_size)
    frame = _frame(train)
    folds = _sequence_folds(frame.sequence_id.to_numpy(object), 3, 429)
    allidx = np.arange(len(frame))
    results = {}
    # Attribution stage is deliberately before either selectable local-affine arm.
    attribution = _seed_attribution(
        train,
        frame,
        folds,
        checkpoints=checkpoints,
        v48_config=args.v48_config,
        train=raw["train"],
        device=device,
        output=args.output_dir,
    )
    # Local arms use the fixed all-seed protocol. Initialization is the matching
    # seed, while optimization is controlled only by the fold.
    for arm_name, arm_raw in raw["arms"].items():
        arm_cfg = ObjectEventV429Config(**arm_raw)
        loss_cfg = ObjectEventV429LossConfig(arm=arm_name, **raw["loss"])
        perseed = []
        records = []
        for seed in raw["train"]["seeds"]:
            accum = {
                k: np.full(len(frame), np.nan)
                for k in (
                    "prediction",
                    "log_eta",
                    "horizontal",
                    "area",
                    "valid",
                    "det",
                    "condition",
                    "mass",
                    "residual",
                    "rotation",
                    "shear",
                    "translation",
                    "boundary",
                    "entropy",
                    "confidence",
                )
            }
            for fold, held in enumerate(folds):
                fit = np.setdiff1d(allidx, held)
                model, hist, names = _train(
                    checkpoints[seed],
                    _subset_split(train, fit),
                    base_config=args.v48_config,
                    model_cfg=arm_cfg,
                    loss_cfg=loss_cfg,
                    train=raw["train"],
                    init_seed=seed,
                    optimization_seed=int(raw["train"]["optimization_seed_by_fold"][fold]),
                    device=device,
                    epochs=int(raw["train"]["epochs"]),
                )
                held_split = _subset_split(train, held)
                got = _predict(model, held_split, device)
                for key in accum:
                    accum[key][held] = got[key]
                local_frame = frame.iloc[held].reset_index(drop=True)
                controls = {
                    "zero_event": _predict(
                        model, held_split, device, torch.zeros_like(held_split.events)
                    ),
                    "temporal_shuffle_t2_t0_t1": _predict(
                        model, held_split, device, held_split.events[:, [2, 0, 1]]
                    ),
                    "endpoint_swap_t0_t2_t1": _predict(
                        model, held_split, device, held_split.events[:, [0, 2, 1]]
                    ),
                }
                control_metrics: dict[str, Any] = {}
                for control_name, control in controls.items():
                    control_frame = local_frame.assign(
                        **{f"control_{key}": value for key, value in control.items()}
                    )
                    control_frame.to_csv(
                        args.output_dir
                        / f"control_{arm_name}_seed{seed}_fold{fold}_{control_name}.csv",
                        index=False,
                    )
                    control_metrics[control_name] = _extra_metrics(local_frame, held_split, control)
                fold_metrics = _extra_metrics(local_frame, held_split, got)
                pd.DataFrame(hist).to_csv(
                    args.output_dir / f"history_{arm_name}_seed{seed}_fold{fold}.csv", index=False
                )
                pd.DataFrame(
                    {**local_frame, **{f"v429_{key}": value for key, value in got.items()}}
                ).to_csv(args.output_dir / f"oof_{arm_name}_seed{seed}_fold{fold}.csv", index=False)
                records.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "held_out_sequences": sorted(local_frame.sequence_id.unique().tolist()),
                        "checkpoint_sha256": _sha256(checkpoints[seed]),
                        "optimization_seed": raw["train"]["optimization_seed_by_fold"][fold],
                        "trainable_geometry_names": names,
                        "history": hist,
                        "metrics": fold_metrics,
                        "controls": control_metrics,
                    }
                )
            perseed.append(accum)
            pd.DataFrame(
                {**frame, **{f"v429_{key}": value for key, value in accum.items()}}
            ).to_csv(
                args.output_dir / f"oof_train_predictions_{arm_name}_seed{seed}.csv", index=False
            )
        median = aggregate_seed_results(perseed)
        metrics = _extra_metrics(frame, train, median)
        if metrics.get("complete_finite", False):
            tracks, pertrack = track_metrics(
                frame,
                median["prediction"],
                minimum_track_samples=8,
                minimum_negative_track_samples=4,
            )
            valid_only_tracks: dict[str, Any] | None = None
            valid_only_pertrack = pd.DataFrame()
            valid_only_persequence = pd.DataFrame()
        else:
            tracks, pertrack = (
                {"negative_track_macro_accuracy": float("nan"), "complete_finite": False},
                pd.DataFrame(),
            )
            valid_mask = median["valid"] > 0.5
            if valid_mask.any():
                valid_frame = frame.loc[valid_mask].reset_index(drop=True)
                valid_prediction = median["prediction"][valid_mask]
                valid_only_tracks, valid_only_pertrack = track_metrics(
                    valid_frame,
                    valid_prediction,
                    minimum_track_samples=8,
                    minimum_negative_track_samples=4,
                )
                _, valid_only_persequence = _metrics(
                    valid_frame,
                    valid_prediction,
                    minimum_negatives=20,
                )
            else:
                valid_only_tracks = None
                valid_only_pertrack = pd.DataFrame()
                valid_only_persequence = pd.DataFrame()
        results[arm_name] = {
            "oof_metrics": metrics,
            "oof_track_metrics": tracks,
            "valid_only_oof_track_metrics": valid_only_tracks,
            "records": records,
        }
        output_frame = pd.DataFrame(
            {**frame, **{f"v429_{key}": value for key, value in median.items()}}
        )
        output_frame.to_csv(args.output_dir / f"oof_train_predictions_{arm_name}.csv", index=False)
        _, persequence = (
            _metrics(frame, median["prediction"], minimum_negatives=20)
            if metrics.get("complete_finite", False)
            else ({}, pd.DataFrame())
        )
        persequence.to_csv(args.output_dir / f"oof_train_per_sequence_{arm_name}.csv", index=False)
        pertrack.to_csv(args.output_dir / f"oof_train_per_track_{arm_name}.csv", index=False)
        valid_only_persequence.to_csv(
            args.output_dir / f"oof_train_per_sequence_{arm_name}_valid_only.csv",
            index=False,
        )
        valid_only_pertrack.to_csv(
            args.output_dir / f"oof_train_per_track_{arm_name}_valid_only.csv",
            index=False,
        )
    champion = choose_champion(results)
    checks = oof_gates(
        results[champion]["oof_metrics"],
        v428["arm_results"][v428["champion"]]["oof_metrics"],
        results[champion]["oof_track_metrics"],
        v427["oof_train_metrics"],
        v428["arm_results"][v428["champion"]]["oof_track_metrics"],
    )
    passed = all(checks.values())
    checkpoint_hashes = {str(seed): _sha256(path) for seed, path in checkpoints.items()}
    summary = {
        "artifact_type": "object_event_v4_29_local_affine",
        "status": "completed_oof_gate_failed",
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_hashes,
        "cache_manifest_hash": _sha256(args.cache_manifest),
        "rng_schedule": raw["train"]["optimization_seed_by_fold"],
        "champion": champion,
        "oof_gate_checks": checks,
        "arm_results": results,
        "seed_attribution": attribution,
        "v427_reference": v427["oof_train_metrics"],
        "v428_reference": v428["arm_results"][v428["champion"]]["oof_metrics"],
        "scientific_contract": {
            "development_validation_not_materialized_after_oof_failure": not passed,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
            "boxes_not_forward_features": True,
            "no_workers": True,
        },
        "train_manifest": manifest,
        "config": raw,
    }
    if not passed:
        summary["run_metadata"] = _run_metadata(
            config_sha256, checkpoint_hashes, started, started_utc, summary["status"]
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False)
        )
        return 0
    # Deferred access: validation files/split are neither read nor materialised above.
    if not args.ensemble_validation.is_file() or not args.v410_summary.is_file():
        raise FileNotFoundError("OOF passed but required v4.10 development references are absent")
    v410_summary = json.loads(args.v410_summary.read_text())
    if v410_summary.get("artifact_type") != "object_event_v4_10_true_seed_fixed_fusion_robustness":
        raise ValueError("--v410-summary is not the expected v4.10 artifact")
    val, val_manifest = _materialize(args.cache_manifest, "validation", input_size=base.input_size)
    val_frame = _frame(val)
    champion_cfg = ObjectEventV429Config(**raw["arms"][champion])
    champion_loss = ObjectEventV429LossConfig(arm=champion, **raw["loss"])
    final_results: list[dict[str, np.ndarray]] = []
    final_records: list[dict[str, Any]] = []
    for seed in raw["train"]["seeds"]:
        model, history, names = _train(
            checkpoints[seed],
            train,
            base_config=args.v48_config,
            model_cfg=champion_cfg,
            loss_cfg=champion_loss,
            train=raw["train"],
            init_seed=seed,
            optimization_seed=42990 + seed,
            device=device,
            epochs=int(raw["train"]["final_epochs"]),
        )
        result = _predict(model, val, device)
        final_results.append(result)
        final_records.append(
            {
                "seed": seed,
                "history": history,
                "trainable_geometry_names": names,
                "checkpoint_sha256": _sha256(checkpoints[seed]),
            }
        )
    median_result = aggregate_seed_results(final_results)
    median_prediction = median_result["prediction"]
    validation_metrics = _extra_metrics(val_frame, val, median_result)
    if validation_metrics.get("complete_finite", False):
        validation_tracks, val_track = track_metrics(
            val_frame,
            median_prediction,
            minimum_track_samples=8,
            minimum_negative_track_samples=4,
        )
        _, val_sequence = _metrics(val_frame, median_prediction, minimum_negatives=20)
    else:
        validation_tracks = {
            "negative_track_macro_accuracy": float("nan"),
            "complete_finite": False,
        }
        val_track = pd.DataFrame()
        val_sequence = pd.DataFrame()
    aligned = _align_ensemble(val, _read_ensemble(args.ensemble_validation))
    v410_prediction = aligned["fused_prediction_expansion"].to_numpy(np.float64)
    v410_metrics, _ = _metrics(aligned, v410_prediction, minimum_negatives=20)
    v410_tracks, _ = track_metrics(
        aligned, v410_prediction, minimum_track_samples=8, minimum_negative_track_samples=4
    )
    decision = raw["final_decision"]
    development_passed = (
        bool(validation_metrics.get("complete_finite", False))
        and float(validation_metrics["pearson"])
        >= float(v410_metrics["pearson"]) + float(decision["minimum_pearson_gain_over_v410"])
        and float(validation_metrics["negative_accuracy"])
        >= float(v410_metrics["negative_accuracy"])
        + float(decision["minimum_negative_accuracy_gain_over_v410"])
        and float(validation_metrics["balanced_sign_accuracy"])
        >= float(v410_metrics["balanced_sign_accuracy"])
        + float(decision["minimum_balanced_sign_gain_over_v410"])
        and float(validation_metrics["log_eta_pearson"])
        >= float(decision["minimum_log_eta_pearson"])
        and float(validation_tracks["negative_track_macro_accuracy"])
        >= float(v410_tracks["negative_track_macro_accuracy"])
        + float(decision["minimum_negative_track_macro_gain_over_v410"])
    )
    summary.update(
        {
            "status": "completed_development_passed"
            if development_passed
            else "completed_development_failed",
            "development_validation_materialized_once": True,
            "validation_manifest": val_manifest,
            "v410_summary_artifact_type": v410_summary.get("artifact_type"),
            "final_seed_records": final_records,
            "validation_metrics": validation_metrics,
            "validation_track_metrics": validation_tracks,
            "v410_validation_metrics": v410_metrics,
            "v410_validation_track_metrics": v410_tracks,
            "development_passed": development_passed,
        }
    )
    val_frame.assign(
        v429_prediction_expansion=median_prediction, v410_prediction_expansion=v410_prediction
    ).to_csv(args.output_dir / "validation_predictions.csv", index=False)
    val_sequence.to_csv(args.output_dir / "validation_per_sequence.csv", index=False)
    val_track.to_csv(args.output_dir / "validation_per_track.csv", index=False)
    summary["run_metadata"] = _run_metadata(
        config_sha256, checkpoint_hashes, started, started_utc, summary["status"]
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, default=str, allow_nan=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
