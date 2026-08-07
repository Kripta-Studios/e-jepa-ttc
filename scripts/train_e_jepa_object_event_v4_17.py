#!/usr/bin/env python3
"""Train Object Event TTC v4.17 signed-anchor temporal sign head."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch.nn import functional
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.train_e_jepa_object_event_v4_16 import (  # noqa: E402
    IDENTITY_COLUMNS,
    ObjectEventV416TrainConfig,
    _align_ensemble,
    _balanced_subset_indices,
    _build_frozen_consensus,
    _folds,
    _git_commit,
    _json_safe,
    _load_v48_config,
    _materialize,
    _metrics,
    _parse_checkpoints,
    _read_ensemble,
    _resolve_device,
    _seed,
    _sha256,
)
from e_jepa_ttc.models.object_event_v4_17 import (  # noqa: E402
    ObjectEventTTCV417,
    ObjectEventV417Config,
)
from e_jepa_ttc.training.object_event_v4_16 import (  # noqa: E402
    causal_window_indices,
    gather_scalar_windows,
    gather_windows,
)
from e_jepa_ttc.training.object_event_v4_17 import (  # noqa: E402
    ObjectEventV417LossConfig,
    signed_anchor_features,
    signed_anchor_logits,
    temporal_sign_loss,
    v417_screen_gates,
)
from e_jepa_ttc.object_event_v4_4 import pearson  # noqa: E402


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_config(
    path: Path,
) -> tuple[
    ObjectEventV417Config,
    ObjectEventV416TrainConfig,
    ObjectEventV417LossConfig,
    dict[str, float],
]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.17 config must be a mapping")
    model = _construct(ObjectEventV417Config, cast(Mapping[str, Any], raw.get("model", {})))
    train = _construct(
        ObjectEventV416TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))
    )
    loss = _construct(
        ObjectEventV417LossConfig, cast(Mapping[str, Any], raw.get("loss", {}))
    )
    gates_raw = raw.get("screen_gates", {})
    if not isinstance(gates_raw, dict) or not gates_raw:
        raise ValueError("screen_gates must be a non-empty mapping")
    gates = {str(key): float(value) for key, value in gates_raw.items()}
    return model, train, loss, gates


@torch.no_grad()
def _extract_features(
    frozen: Any,
    events: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    branch: str,
    seed: int,
    maximum_magnitude: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return consensus descriptor, magnitude anchor and signed anchor."""
    if branch not in {"event", "zero", "shuffled"}:
        raise KeyError(branch)
    permutation = None
    if branch == "shuffled":
        permutation = torch.randperm(
            len(events), generator=torch.Generator().manual_seed(seed)
        )
    descriptors_out: list[torch.Tensor] = []
    magnitudes_out: list[torch.Tensor] = []
    signed_out: list[torch.Tensor] = []
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
        descriptors: list[torch.Tensor] = []
        signed_expansions: list[torch.Tensor] = []
        for extractor in frozen.extractors:
            descriptor, pooled_log_eta = extractor._descriptor_and_base(batch)
            descriptor = functional.layer_norm(
                descriptor,
                (descriptor.shape[-1],),
                weight=None,
                bias=None,
                eps=frozen.config.epsilon,
            )
            signed = (1.0 - torch.exp(pooled_log_eta)).clamp(
                -maximum_magnitude, maximum_magnitude
            )
            descriptors.append(descriptor)
            signed_expansions.append(signed)
        stacked = torch.stack(descriptors, dim=0)
        consensus = torch.cat(
            (stacked.mean(dim=0), stacked.median(dim=0).values), dim=1
        )
        signed_stack = torch.stack(signed_expansions, dim=0)
        signed_anchor = signed_stack.median(dim=0).values
        magnitude_anchor = signed_stack.abs().median(dim=0).values
        descriptors_out.append(consensus.float().cpu())
        magnitudes_out.append(magnitude_anchor.float().cpu())
        signed_out.append(signed_anchor.float().cpu())
    return (
        torch.cat(descriptors_out, dim=0),
        torch.cat(magnitudes_out, dim=0),
        torch.cat(signed_out, dim=0),
    )


def _augment_descriptor(
    descriptor: torch.Tensor,
    signed_anchor: torch.Tensor,
    *,
    anchor_scale: float,
    config: ObjectEventV417Config,
) -> torch.Tensor:
    features = signed_anchor_features(
        signed_anchor,
        train_scale=anchor_scale,
        clip=config.anchor_feature_clip,
    )
    return torch.cat((descriptor, features), dim=1)


def _new_head(
    descriptor_dim: int,
    config: ObjectEventV417Config,
    *,
    seed: int,
    device: torch.device,
) -> ObjectEventTTCV417:
    _seed(seed)
    return ObjectEventTTCV417(descriptor_dim, config).to(device)


def _train_head(
    *,
    model: ObjectEventTTCV417,
    windows: torch.Tensor,
    window_mask: torch.Tensor,
    anchor: torch.Tensor,
    anchor_logit_windows: torch.Tensor,
    target: torch.Tensor,
    history_target: torch.Tensor,
    train_indices: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    loss_config: ObjectEventV417LossConfig,
    device: torch.device,
    seed: int,
    epoch_callback: Any | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_indices), generator=generator)
        sampled = train_indices[order]
        totals = np.zeros(3, dtype=np.float64)
        seen = 0
        for start in range(0, len(sampled), batch_size):
            index = sampled[start : start + batch_size]
            x = windows[index].to(device=device, dtype=torch.float32)
            mask = window_mask[index].to(device=device)
            a = anchor[index].to(device=device, dtype=torch.float32)
            al = anchor_logit_windows[index].to(device=device, dtype=torch.float32)
            y = target[index].to(device=device, dtype=torch.float32)
            hy = history_target[index].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, mask, a, al)
            loss = temporal_sign_loss(
                sign_logit=output.sign_logit,
                instant_sign_logits=output.instant_sign_logits,
                target_expansion=y,
                history_target_expansion=hy,
                history_mask=mask,
                config=loss_config,
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            size = len(index)
            totals += size * np.asarray(
                [
                    float(loss.total.detach().cpu()),
                    float(loss.final_sign.detach().cpu()),
                    float(loss.history_sign.detach().cpu()),
                ]
            )
            seen += size
        values = totals / max(seen, 1)
        row = {
            "epoch": float(epoch),
            "loss": float(values[0]),
            "final_sign_loss": float(values[1]),
            "history_sign_loss": float(values[2]),
        }
        history.append(row)
        if epoch_callback is not None and bool(epoch_callback(epoch, model, row)):
            break
    return copy.deepcopy(model.state_dict()), history


@torch.no_grad()
def _predict(
    model: ObjectEventTTCV417,
    windows: torch.Tensor,
    window_mask: torch.Tensor,
    anchor: torch.Tensor,
    anchor_logit_windows: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    anchor_logits: list[np.ndarray] = []
    residual_logits: list[np.ndarray] = []
    odd_errors: list[np.ndarray] = []
    weight_sum = None
    row_count = 0
    model.eval()
    for start in range(0, len(windows), batch_size):
        end = min(start + batch_size, len(windows))
        x = windows[start:end].to(device=device, dtype=torch.float32)
        mask = window_mask[start:end].to(device=device)
        a = anchor[start:end].to(device=device, dtype=torch.float32)
        al = anchor_logit_windows[start:end].to(device=device, dtype=torch.float32)
        output = model(x, mask, a, al)
        error = model.oddness_error(x, mask, a, al)
        predictions.append(output.signed_expansion.float().cpu().numpy())
        probabilities.append(output.negative_probability.float().cpu().numpy())
        logits.append(output.sign_logit.float().cpu().numpy())
        anchor_logits.append(output.anchor_logit.float().cpu().numpy())
        residual_logits.append(output.residual_logit.float().cpu().numpy())
        odd_errors.append(error.float().cpu().numpy())
        current = output.sign_temporal_weights.float().cpu().sum(dim=0)
        weight_sum = current if weight_sum is None else weight_sum + current
        row_count += len(x)
    return {
        "prediction": np.concatenate(predictions),
        "probability": np.concatenate(probabilities),
        "logit": np.concatenate(logits),
        "anchor_logit": np.concatenate(anchor_logits),
        "residual_logit": np.concatenate(residual_logits),
        "sign_oddness_max_abs": float(np.max(np.concatenate(odd_errors))),
        "mean_sign_temporal_weights": (weight_sum / max(row_count, 1)).numpy().tolist(),
    }


def _overfit_gates(metrics: Mapping[str, Any], oddness: float) -> dict[str, bool]:
    return {
        "balanced_sign": float(metrics["balanced_sign_accuracy"]) >= 0.98,
        "negative_accuracy": float(metrics["negative_accuracy"]) >= 0.98,
        "expansion_mae": float(metrics["expansion_mae"]) <= 0.008,
        "exact_sign_oddness": oddness <= 1.0e-5,
    }


def _overfit_rank(metrics: Mapping[str, Any], gates: Mapping[str, bool]) -> tuple[float, ...]:
    return (
        float(sum(gates.values())),
        float(metrics["balanced_sign_accuracy"]),
        float(metrics["negative_accuracy"]),
        -float(metrics["expansion_mae"]),
        float(metrics["pearson"]),
    )


def _anchor_metrics(
    frame: pd.DataFrame,
    signed_anchor: np.ndarray,
    magnitude_anchor: np.ndarray,
    minimum_negatives: int,
) -> dict[str, Any]:
    metrics, _ = _metrics(frame, signed_anchor, minimum_negatives=minimum_negatives)
    target_magnitude = np.abs(frame["target_expansion"].to_numpy(dtype=np.float64))
    metrics["magnitude_mae"] = float(
        np.mean(np.abs(np.asarray(magnitude_anchor, dtype=np.float64) - target_magnitude))
    )
    return metrics


def run(
    *,
    cache_manifest: Path,
    v48_config_path: Path,
    v412_config_path: Path,
    config_path: Path,
    checkpoint_paths: Mapping[int, Path],
    ensemble_train_path: Path,
    ensemble_validation_path: Path,
    v416_summary_path: Path,
    output_dir: Path,
    device_name: str,
    mode: str,
    force: bool,
) -> dict[str, Any]:
    if mode not in {"overfit", "screen"}:
        raise ValueError("mode must be overfit or screen")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    started_at = datetime.now(UTC)
    started = time.perf_counter()

    model_config, train_config, loss_config, gate_config = _load_config(config_path)
    device = _resolve_device(device_name)
    frozen, checkpoint_payloads = _build_frozen_consensus(
        checkpoint_paths=checkpoint_paths,
        v48_config_path=v48_config_path,
        v412_config_path=v412_config_path,
        device=device,
    )
    base_config, _, _, _, _ = _load_v48_config(v48_config_path)
    train_split, train_manifest = _materialize(
        cache_manifest, "train", input_size=base_config.input_size
    )
    validation_split, validation_manifest = _materialize(
        cache_manifest, "validation", input_size=base_config.input_size
    )
    train_frame = _align_ensemble(train_split, _read_ensemble(ensemble_train_path))
    validation_frame = _align_ensemble(validation_split, _read_ensemble(ensemble_validation_path))

    train_descriptor, train_anchor, train_signed_anchor = _extract_features(
        frozen,
        train_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="event",
        seed=train_config.seed,
        maximum_magnitude=model_config.maximum_magnitude,
    )
    anchor_scale = float(torch.median(train_signed_anchor.abs()).item())
    anchor_scale = max(anchor_scale, 1.0e-4)
    train_descriptor = _augment_descriptor(
        train_descriptor,
        train_signed_anchor,
        anchor_scale=anchor_scale,
        config=model_config,
    )
    train_anchor_logits = signed_anchor_logits(
        train_signed_anchor,
        train_scale=anchor_scale,
        clip=model_config.anchor_feature_clip,
        strength=model_config.anchor_logit_strength,
    )
    descriptor_dim = int(train_descriptor.shape[1])
    train_window_index, train_window_mask_np, train_history_length = causal_window_indices(
        train_frame, window_size=model_config.window_size
    )
    train_windows = gather_windows(train_descriptor, train_window_index, train_window_mask_np)
    train_anchor_logit_windows = gather_scalar_windows(
        train_anchor_logits, train_window_index, train_window_mask_np
    )
    train_window_mask = torch.as_tensor(train_window_mask_np, dtype=torch.bool)
    train_target = torch.as_tensor(train_frame["target_expansion"].to_numpy(dtype=np.float32))
    train_history_target = gather_scalar_windows(
        train_target, train_window_index, train_window_mask_np
    )

    if mode == "overfit":
        selected = _balanced_subset_indices(train_frame, train_config.overfit_samples, train_config.seed)
        frame = train_frame.iloc[selected.numpy()].reset_index(drop=True)
        model = _new_head(descriptor_dim, model_config, seed=train_config.seed, device=device)
        best: dict[str, Any] | None = None
        gate: dict[str, Any] | None = None

        def callback(epoch: int, candidate: ObjectEventTTCV417, row: dict[str, float]) -> bool:
            nonlocal best, gate
            pred = _predict(
                candidate,
                train_windows[selected],
                train_window_mask[selected],
                train_anchor[selected],
                train_anchor_logit_windows[selected],
                batch_size=train_config.head_batch_size,
                device=device,
            )
            metrics, _ = _metrics(
                frame,
                cast(np.ndarray, pred["prediction"]),
                minimum_negatives=train_config.per_sequence_negative_min_count,
            )
            oddness = float(pred["sign_oddness_max_abs"])
            gates = _overfit_gates(metrics, oddness)
            rank = _overfit_rank(metrics, gates)
            payload = {
                "epoch": epoch,
                "state": copy.deepcopy(candidate.state_dict()),
                "metrics": copy.deepcopy(metrics),
                "oddness": oddness,
                "gates": copy.deepcopy(gates),
                "rank": rank,
            }
            if best is None or rank > best["rank"]:
                best = payload
            row.update({
                "eval_balanced_sign_accuracy": float(metrics["balanced_sign_accuracy"]),
                "eval_negative_accuracy": float(metrics["negative_accuracy"]),
                "eval_expansion_mae": float(metrics["expansion_mae"]),
                "eval_passed": float(all(gates.values())),
            })
            if all(gates.values()):
                gate = payload
                return True
            return False

        _, history = _train_head(
            model=model,
            windows=train_windows,
            window_mask=train_window_mask,
            anchor=train_anchor,
            anchor_logit_windows=train_anchor_logit_windows,
            target=train_target,
            history_target=train_history_target,
            train_indices=selected,
            epochs=train_config.overfit_epochs,
            batch_size=train_config.head_batch_size,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            max_grad_norm=train_config.max_grad_norm,
            loss_config=loss_config,
            device=device,
            seed=train_config.seed,
            epoch_callback=callback,
        )
        chosen = gate or best
        assert chosen is not None
        passed = gate is not None
        checkpoint_name = "best_gate_passing.pt" if passed else "best_observed.pt"
        torch.save({
            "artifact_type": "object_event_v4_17_overfit_checkpoint",
            "model_state_dict": chosen["state"],
            "descriptor_dim": descriptor_dim,
            "anchor_scale": anchor_scale,
            "model_config": asdict(model_config),
            "epoch": chosen["epoch"],
        }, output_dir / checkpoint_name)
        pd.DataFrame(history).to_json(output_dir / "history.jsonl", orient="records", lines=True)
        result = {
            "artifact_type": "object_event_v4_17_signed_anchor_temporal",
            "status": "overfit_passed" if passed else "overfit_failed",
            "mode": mode,
            "created_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "completed_epochs": len(history),
            "maximum_epochs": train_config.overfit_epochs,
            "selected_epoch": chosen["epoch"],
            "selected_checkpoint": checkpoint_name,
            "anchor_scale_train_only": anchor_scale,
            "metrics": chosen["metrics"],
            "diagnostics": {"sign_oddness_max_abs": chosen["oddness"]},
            "gates": chosen["gates"],
            "passed": passed,
        }
        (output_dir / "summary.json").write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
        return result

    validation_descriptor, validation_anchor, validation_signed_anchor = _extract_features(
        frozen,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="event",
        seed=train_config.seed + 1,
        maximum_magnitude=model_config.maximum_magnitude,
    )
    validation_descriptor = _augment_descriptor(
        validation_descriptor,
        validation_signed_anchor,
        anchor_scale=anchor_scale,
        config=model_config,
    )
    validation_anchor_logits = signed_anchor_logits(
        validation_signed_anchor,
        train_scale=anchor_scale,
        clip=model_config.anchor_feature_clip,
        strength=model_config.anchor_logit_strength,
    )
    validation_window_index, validation_window_mask_np, validation_history_length = causal_window_indices(
        validation_frame, window_size=model_config.window_size
    )
    validation_windows = gather_windows(
        validation_descriptor, validation_window_index, validation_window_mask_np
    )
    validation_anchor_logit_windows = gather_scalar_windows(
        validation_anchor_logits, validation_window_index, validation_window_mask_np
    )
    validation_window_mask = torch.as_tensor(validation_window_mask_np, dtype=torch.bool)

    fold_rows: list[dict[str, Any]] = []
    oof_prediction = np.full(len(train_frame), np.nan, dtype=np.float64)
    oof_probability = np.full(len(train_frame), np.nan, dtype=np.float64)
    for fold_index, held_out in enumerate(
        _folds(train_frame["sequence_id"].astype(str).tolist(), train_config.fold_count, train_config.seed)
    ):
        train_indices_np = np.setdiff1d(np.arange(len(train_frame)), held_out)
        model = _new_head(descriptor_dim, model_config, seed=train_config.seed + 100 + fold_index, device=device)
        _, history = _train_head(
            model=model,
            windows=train_windows,
            window_mask=train_window_mask,
            anchor=train_anchor,
            anchor_logit_windows=train_anchor_logit_windows,
            target=train_target,
            history_target=train_history_target,
            train_indices=torch.as_tensor(train_indices_np, dtype=torch.long),
            epochs=train_config.fold_epochs,
            batch_size=train_config.head_batch_size,
            learning_rate=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            max_grad_norm=train_config.max_grad_norm,
            loss_config=loss_config,
            device=device,
            seed=train_config.seed + 100 + fold_index,
        )
        held = torch.as_tensor(held_out, dtype=torch.long)
        pred = _predict(
            model,
            train_windows[held],
            train_window_mask[held],
            train_anchor[held],
            train_anchor_logit_windows[held],
            batch_size=train_config.head_batch_size,
            device=device,
        )
        oof_prediction[held_out] = cast(np.ndarray, pred["prediction"])
        oof_probability[held_out] = cast(np.ndarray, pred["probability"])
        held_frame = train_frame.iloc[held_out].reset_index(drop=True)
        fold_metrics, _ = _metrics(
            held_frame,
            cast(np.ndarray, pred["prediction"]),
            minimum_negatives=train_config.per_sequence_negative_min_count,
        )
        fold_rows.append({
            "fold": fold_index,
            "held_out_sequences": sorted(held_frame["sequence_id"].astype(str).unique().tolist()),
            "epochs": train_config.fold_epochs,
            "final_loss": history[-1]["loss"],
            "metrics": fold_metrics,
        })
    if not np.isfinite(oof_prediction).all():
        raise AssertionError("OOF predictions are incomplete")
    oof_metrics, _ = _metrics(
        train_frame,
        oof_prediction,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )

    final_model = _new_head(descriptor_dim, model_config, seed=train_config.seed, device=device)
    final_state, final_history = _train_head(
        model=final_model,
        windows=train_windows,
        window_mask=train_window_mask,
        anchor=train_anchor,
        anchor_logit_windows=train_anchor_logit_windows,
        target=train_target,
        history_target=train_history_target,
        train_indices=torch.arange(len(train_frame), dtype=torch.long),
        epochs=train_config.final_epochs,
        batch_size=train_config.head_batch_size,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        max_grad_norm=train_config.max_grad_norm,
        loss_config=loss_config,
        device=device,
        seed=train_config.seed,
    )
    validation_prediction = _predict(
        final_model,
        validation_windows,
        validation_window_mask,
        validation_anchor,
        validation_anchor_logit_windows,
        batch_size=train_config.head_batch_size,
        device=device,
    )
    prediction = cast(np.ndarray, validation_prediction["prediction"])
    validation_metrics, per_sequence = _metrics(
        validation_frame,
        prediction,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )
    baseline_prediction = validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    baseline_metrics, _ = _metrics(
        validation_frame,
        baseline_prediction,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )
    anchor_metrics = _anchor_metrics(
        validation_frame,
        validation_signed_anchor.numpy().astype(np.float64),
        validation_anchor.numpy().astype(np.float64),
        train_config.per_sequence_negative_min_count,
    )

    zero_descriptor, zero_anchor, zero_signed = _extract_features(
        frozen,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="zero",
        seed=train_config.seed + 2,
        maximum_magnitude=model_config.maximum_magnitude,
    )
    shuffled_descriptor, shuffled_anchor, shuffled_signed = _extract_features(
        frozen,
        validation_split.events,
        batch_size=train_config.descriptor_batch_size,
        device=device,
        branch="shuffled",
        seed=train_config.seed + 3,
        maximum_magnitude=model_config.maximum_magnitude,
    )
    zero_descriptor = _augment_descriptor(zero_descriptor, zero_signed, anchor_scale=anchor_scale, config=model_config)
    shuffled_descriptor = _augment_descriptor(shuffled_descriptor, shuffled_signed, anchor_scale=anchor_scale, config=model_config)
    zero_anchor_logits = signed_anchor_logits(
        zero_signed, train_scale=anchor_scale, clip=model_config.anchor_feature_clip,
        strength=model_config.anchor_logit_strength,
    )
    shuffled_anchor_logits = signed_anchor_logits(
        shuffled_signed, train_scale=anchor_scale, clip=model_config.anchor_feature_clip,
        strength=model_config.anchor_logit_strength,
    )
    zero_windows = gather_windows(zero_descriptor, validation_window_index, validation_window_mask_np)
    shuffled_windows = gather_windows(shuffled_descriptor, validation_window_index, validation_window_mask_np)
    zero_anchor_logit_windows = gather_scalar_windows(
        zero_anchor_logits, validation_window_index, validation_window_mask_np
    )
    shuffled_anchor_logit_windows = gather_scalar_windows(
        shuffled_anchor_logits, validation_window_index, validation_window_mask_np
    )
    zero_result = _predict(
        final_model, zero_windows, validation_window_mask, zero_anchor, zero_anchor_logit_windows,
        batch_size=train_config.head_batch_size, device=device,
    )
    shuffled_result = _predict(
        final_model, shuffled_windows, validation_window_mask, shuffled_anchor, shuffled_anchor_logit_windows,
        batch_size=train_config.head_batch_size, device=device,
    )
    target_validation = validation_frame["target_expansion"].to_numpy(dtype=np.float64)
    predicted_negative = cast(np.ndarray, validation_prediction["logit"]) >= 0.0
    anchor_negative = validation_signed_anchor.numpy() < 0.0
    true_negative = target_validation < 0.0
    oof_predicted_negative = oof_probability >= 0.5
    train_true_negative = train_frame["target_expansion"].to_numpy(dtype=np.float64) < 0.0
    diagnostics = {
        "zero_event_pearson_drop": pearson(target_validation, prediction) - pearson(target_validation, cast(np.ndarray, zero_result["prediction"])),
        "shuffled_event_pearson_drop": pearson(target_validation, prediction) - pearson(target_validation, cast(np.ndarray, shuffled_result["prediction"])),
        "sign_oddness_max_abs": float(validation_prediction["sign_oddness_max_abs"]),
        "mean_history_length": float(np.mean(validation_history_length)),
        "full_history_fraction": float(np.mean(validation_history_length >= model_config.window_size)),
        "mean_sign_temporal_weights": validation_prediction["mean_sign_temporal_weights"],
        "anchor_scale_train_only": anchor_scale,
        "signed_anchor_validation_pearson": float(anchor_metrics["pearson"]),
        "signed_anchor_validation_negative_accuracy": float(anchor_metrics["negative_accuracy"]),
        "signed_anchor_validation_min_sequence_negative_accuracy": float(anchor_metrics["minimum_sequence_negative_accuracy"]),
        "train_true_negative_rate": float(np.mean(train_true_negative)),
        "oof_predicted_negative_rate": float(np.mean(oof_predicted_negative)),
        "validation_true_negative_rate": float(np.mean(true_negative)),
        "signed_anchor_predicted_negative_rate": float(np.mean(anchor_negative)),
        "validation_predicted_negative_rate": float(np.mean(predicted_negative)),
        "temporal_vs_anchor_flip_rate": float(np.mean(predicted_negative != anchor_negative)),
        "mean_abs_anchor_logit": float(np.mean(np.abs(cast(np.ndarray, validation_prediction["anchor_logit"])))),
        "mean_abs_residual_logit": float(np.mean(np.abs(cast(np.ndarray, validation_prediction["residual_logit"])))),
        "uniform_without_replacement_sampling": True,
        "sign_importance_reweighting": False,
    }
    gate_diagnostics = {
        key: float(value)
        for key, value in diagnostics.items()
        if isinstance(value, (int, float, np.floating))
    }
    gates = v417_screen_gates(
        oof=cast(Mapping[str, float], oof_metrics),
        validation=cast(Mapping[str, float], validation_metrics),
        baseline=cast(Mapping[str, float], baseline_metrics),
        anchor=cast(Mapping[str, float], anchor_metrics),
        diagnostics=gate_diagnostics,
        gates=gate_config,
    )
    passed = all(gates.values())

    validation_output = validation_frame.loc[:, list(IDENTITY_COLUMNS) + [
        "delta_t_s", "target_ttc_s", "target_expansion"
    ]].copy()
    validation_output["history_length"] = validation_history_length
    validation_output["baseline_prediction_expansion"] = baseline_prediction
    validation_output["v48_signed_anchor_expansion"] = validation_signed_anchor.numpy()
    validation_output["v48_anchor_magnitude"] = validation_anchor.numpy()
    validation_output["negative_probability"] = cast(np.ndarray, validation_prediction["probability"])
    validation_output["anchor_logit"] = cast(np.ndarray, validation_prediction["anchor_logit"])
    validation_output["residual_logit"] = cast(np.ndarray, validation_prediction["residual_logit"])
    validation_output["prediction_expansion"] = prediction
    validation_output["zero_events_prediction_expansion"] = cast(np.ndarray, zero_result["prediction"])
    validation_output["shuffled_prediction_expansion"] = cast(np.ndarray, shuffled_result["prediction"])
    validation_output.to_csv(output_dir / "validation_predictions.csv", index=False)
    per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)

    oof_output = train_frame.loc[:, list(IDENTITY_COLUMNS) + ["target_expansion", "fused_prediction_expansion"]].copy()
    oof_output["history_length"] = train_history_length
    oof_output["v48_signed_anchor_expansion"] = train_signed_anchor.numpy()
    oof_output["oof_negative_probability"] = oof_probability
    oof_output["oof_prediction_expansion"] = oof_prediction
    oof_output.to_csv(output_dir / "train_oof_predictions.csv", index=False)
    pd.DataFrame(final_history).to_json(output_dir / "final_history.jsonl", orient="records", lines=True)
    (output_dir / "folds.json").write_text(json.dumps(_json_safe(fold_rows), indent=2), encoding="utf-8")
    torch.save({
        "artifact_type": "object_event_v4_17_signed_anchor_temporal_checkpoint",
        "model_state_dict": final_state,
        "descriptor_dim": descriptor_dim,
        "anchor_scale": anchor_scale,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "checkpoint_paths": {seed: path.resolve().as_posix() for seed, path in checkpoint_paths.items()},
    }, output_dir / "signed_anchor_temporal.pt")

    v416 = json.loads(v416_summary_path.read_text(encoding="utf-8"))
    result = {
        "artifact_type": "object_event_v4_17_signed_anchor_temporal",
        "status": "screen_passed" if passed else "screen_failed",
        "mode": mode,
        "created_at": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": {
            "model": asdict(model_config),
            "train": asdict(train_config),
            "loss": asdict(loss_config),
            "screen_gates": gate_config,
        },
        "anchor_scale_train_only": anchor_scale,
        "source_v416_status": v416.get("status"),
        "cache_manifest": cache_manifest.resolve().as_posix(),
        "cache_manifest_sha256": _sha256(cache_manifest),
        "v48_checkpoints": {
            seed: {"path": path.resolve().as_posix(), "sha256": _sha256(path), "epoch": checkpoint_payloads[seed].get("epoch")}
            for seed, path in checkpoint_paths.items()
        },
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "folds": fold_rows,
        "oof_metrics": oof_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "signed_anchor_validation_metrics": anchor_metrics,
        "temporal_validation_metrics": validation_metrics,
        "diagnostics": diagnostics,
        "gates": gates,
        "passed": passed,
        "scientific_contract": {
            "v416_failed_result_is_diagnostic_source_not_relabelled": True,
            "three_frozen_true_seed_v48_backbones": True,
            "signed_v48_anchor_is_event_only_forward_feature": True,
            "signed_anchor_scale_estimated_from_train_only": True,
            "anchor_plus_bounded_temporal_residual_logit": True,
            "magnitude_is_frozen_v48_multiseed_median": True,
            "only_temporal_residual_sign_head_is_trainable": True,
            "uniform_without_replacement_epoch_sampling": True,
            "no_sequence_or_sign_importance_reweighting": True,
            "causal_track_windows": True,
            "track_and_sequence_ids_are_grouping_metadata_not_forward_features": True,
            "exact_odd_temporal_sign_head": True,
            "validation_not_used_for_epoch_or_hyperparameter_selection": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
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
    parser.add_argument("--v416-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("overfit", "screen"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            cache_manifest=args.cache_manifest,
            v48_config_path=args.v48_config,
            v412_config_path=args.v412_config,
            config_path=args.config,
            checkpoint_paths=_parse_checkpoints(args.v48_checkpoint),
            ensemble_train_path=args.ensemble_train,
            ensemble_validation_path=args.ensemble_validation,
            v416_summary_path=args.v416_summary,
            output_dir=args.output_dir,
            device_name=args.device,
            mode=args.mode,
            force=args.force,
        )
        print(json.dumps(_json_safe(result), indent=2))
        return 0 if result.get("passed") else 2
    except Exception as exc:
        failure = {
            "artifact_type": "object_event_v4_17_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
