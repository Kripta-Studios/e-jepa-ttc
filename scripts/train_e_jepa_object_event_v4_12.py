#!/usr/bin/env python3
"""Train the Object Event TTC v4.12 reversal-balanced directional sign probe."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.train_e_jepa_object_event_v4_2 import (  # noqa: E402
    _autocast,
    _git_commit,
    _json_safe,
    _resolve_device,
    _seed,
    _sha256,
)
from scripts.train_e_jepa_object_event_v4_6 import (  # noqa: E402
    MaterializedV46Split,
    _materialize,
)
from scripts.train_e_jepa_object_event_v4_8 import _load_config as _load_v48_config  # noqa: E402
from e_jepa_ttc.models.object_event_v4_8 import ObjectEventTTCV48  # noqa: E402
from e_jepa_ttc.models.object_event_v4_12 import (  # noqa: E402
    ObjectEventTTCV412,
    ObjectEventV412Config,
)
from e_jepa_ttc.object_event_v4_4 import (  # noqa: E402
    branch_metrics,
    official_eap_metrics,
    pearson,
    sequence_sign_weights,
)
from e_jepa_ttc.training.object_event_v4_12 import (  # noqa: E402
    ObjectEventV412LossConfig,
    directional_sign_checkpoint_gates,
    directional_sign_gates,
    reversal_balanced_sign_loss,
)

IDENTITY_COLUMNS = ("sequence_id", "sample_token", "track_id")
ENSEMBLE_COLUMNS = (
    *IDENTITY_COLUMNS,
    "delta_t_s",
    "target_ttc_s",
    "target_expansion",
    "fused_prediction_expansion",
    "fused_zero_events_expansion",
    "fused_shuffled_mean_expansion",
)


@dataclass(frozen=True)
class ObjectEventV412TrainConfig:
    batch_size: int = 32
    maximum_epochs: int = 20
    minimum_epochs: int = 6
    patience_epochs: int = 6
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 2.0
    precision: str = "fp32"
    seed: int = 7
    overfit_samples: int = 64
    overfit_maximum_epochs: int = 80
    negative_threshold: float = 0.5
    per_sequence_negative_min_count: int = 20
    overfit_balanced_sign_gate: float = 0.95
    overfit_negative_accuracy_gate: float = 0.95
    overfit_reverse_accuracy_gate: float = 0.95
    overfit_antisymmetry_ceiling: float = 0.35
    screen_pearson_floor: float = 0.63
    screen_pearson_max_drop: float = 0.025
    screen_mae_tolerance: float = 0.0015
    screen_balanced_sign_gate: float = 0.76
    screen_negative_accuracy_gate: float = 0.66
    screen_min_sequence_negative_accuracy_gate: float = 0.30
    screen_reverse_accuracy_gate: float = 0.80
    zero_event_pearson_drop_gate: float = 0.55
    shuffled_event_pearson_drop_gate: float = 0.55

    def __post_init__(self) -> None:
        if min(
            self.batch_size,
            self.maximum_epochs,
            self.minimum_epochs,
            self.patience_epochs,
            self.overfit_samples,
            self.overfit_maximum_epochs,
            self.per_sequence_negative_min_count,
        ) <= 0:
            raise ValueError("v4.12 integer controls must be positive")
        if self.minimum_epochs > self.maximum_epochs:
            raise ValueError("minimum_epochs exceeds maximum_epochs")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if min(self.learning_rate, self.max_grad_norm) <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0.0 < self.negative_threshold < 1.0:
            raise ValueError("negative_threshold must lie in (0,1)")


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_probe_config(
    path: Path,
) -> tuple[ObjectEventV412Config, ObjectEventV412TrainConfig, ObjectEventV412LossConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.12 config must be a mapping")
    return (
        _construct(ObjectEventV412Config, cast(Mapping[str, Any], raw.get("probe", {}))),
        _construct(ObjectEventV412TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))),
        _construct(ObjectEventV412LossConfig, cast(Mapping[str, Any], raw.get("loss", {}))),
    )


def _load_backbone(
    *,
    v48_config_path: Path,
    checkpoint_path: Path,
) -> tuple[ObjectEventTTCV48, dict[str, Any]]:
    base_config, foreground_config, motion_config, _, _ = _load_v48_config(v48_config_path)
    model = ObjectEventTTCV48(base_config, foreground_config, motion_config)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid v4.8 checkpoint: {checkpoint_path}")
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model_state_dict"]), strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, cast(dict[str, Any], payload)


def _read_ensemble(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in ENSEMBLE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{path} contains duplicate identities")
    return frame.loc[:, list(ENSEMBLE_COLUMNS)].copy()


def _align_ensemble(split: MaterializedV46Split, frame: pd.DataFrame) -> pd.DataFrame:
    wanted = pd.DataFrame(
        {
            "sequence_id": split.sequence_ids,
            "sample_token": split.sample_tokens,
            "track_id": split.track_ids,
            "_order": np.arange(len(split), dtype=np.int64),
        }
    )
    aligned = wanted.merge(frame, on=list(IDENTITY_COLUMNS), how="left", validate="one_to_one")
    if aligned["target_expansion"].isna().any():
        raise ValueError("ensemble predictions do not cover the materialized split")
    aligned = aligned.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
    delta = split.delta_t_s.numpy().astype(np.float64)
    target_ttc = split.target_ttc_s.numpy().astype(np.float64)
    if not np.allclose(aligned["delta_t_s"], delta, atol=1.0e-8, rtol=0.0):
        raise ValueError("delta_t mismatch between cache and ensemble")
    if not np.allclose(aligned["target_ttc_s"], target_ttc, atol=1.0e-6, rtol=0.0):
        raise ValueError("TTC target mismatch between cache and ensemble")
    return aligned


def _balanced_subset_indices(frame: pd.DataFrame, count: int, seed: int) -> torch.Tensor:
    if count > len(frame):
        raise ValueError("overfit sample count exceeds split")
    rng = np.random.default_rng(seed)
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    negative = np.flatnonzero(target < 0.0)
    positive = np.flatnonzero(target >= 0.0)
    half = count // 2
    if len(negative) < half or len(positive) < count - half:
        raise ValueError("insufficient signs for balanced overfit subset")
    chosen = np.concatenate(
        (
            rng.choice(negative, size=half, replace=False),
            rng.choice(positive, size=count - half, replace=False),
        )
    )
    rng.shuffle(chosen)
    return torch.as_tensor(chosen, dtype=torch.long)


def _sample_weights(frame: pd.DataFrame) -> torch.Tensor:
    values = sequence_sign_weights(
        frame["sequence_id"].astype(str).tolist(),
        frame["target_expansion"].to_numpy(dtype=np.float64),
        cap=10.0,
    )
    return torch.as_tensor(values, dtype=torch.float64)


def _per_sequence(frame: pd.DataFrame, prediction: np.ndarray, *, minimum_negatives: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    work = frame.loc[:, list(IDENTITY_COLUMNS)].copy()
    work["target_expansion"] = target
    work["prediction_expansion"] = prediction
    for sequence_id, group in work.groupby("sequence_id", sort=True):
        y = group["target_expansion"].to_numpy(dtype=np.float64)
        p = group["prediction_expansion"].to_numpy(dtype=np.float64)
        negative = y < 0.0
        positive = ~negative
        rows.append(
            {
                "sequence_id": str(sequence_id),
                "count": len(group),
                "negative_count": int(negative.sum()),
                "positive_count": int(positive.sum()),
                "pearson": pearson(y, p),
                "expansion_mae": float(np.mean(np.abs(y - p))),
                "positive_accuracy": float(np.mean(p[positive] >= 0.0)) if positive.any() else 0.0,
                "negative_accuracy": float(np.mean(p[negative] < 0.0)) if negative.any() else 0.0,
            }
        )
    result = pd.DataFrame.from_records(rows)
    eligible = result[result["negative_count"] >= minimum_negatives]
    result.attrs["minimum_sequence_negative_accuracy"] = (
        float(eligible["negative_accuracy"].min()) if not eligible.empty else 0.0
    )
    result.attrs["minimum_sequence_pearson"] = float(result["pearson"].min())
    return result


@torch.no_grad()
def _predict(
    model: ObjectEventTTCV412,
    split: MaterializedV46Split,
    frame: pd.DataFrame,
    *,
    batch_size: int,
    device: torch.device,
    threshold: float,
    branch: str,
    shuffle_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if branch == "event":
        magnitude_column = "fused_prediction_expansion"
    elif branch == "zero":
        magnitude_column = "fused_zero_events_expansion"
    elif branch == "shuffled":
        magnitude_column = "fused_shuffled_mean_expansion"
    else:
        raise KeyError(branch)
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    reverse_probabilities: list[np.ndarray] = []
    rng = torch.Generator().manual_seed(shuffle_seed)
    permutation = torch.randperm(len(split), generator=rng) if branch == "shuffled" else None
    model.eval()
    for start in range(0, len(split), batch_size):
        end = min(start + batch_size, len(split))
        if branch == "zero":
            events = torch.zeros_like(split.events[start:end]).to(device=device, dtype=torch.float32)
        elif branch == "shuffled":
            assert permutation is not None
            events = split.events[permutation[start:end]].to(device=device, dtype=torch.float32)
        else:
            events = split.events[start:end].to(device=device, dtype=torch.float32)
        magnitude = torch.as_tensor(
            frame[magnitude_column].to_numpy(dtype=np.float32)[start:end],
            device=device,
        )
        output = model(events, magnitude_expansion=magnitude, negative_threshold=threshold)
        predictions.append(output.signed_expansion.float().cpu().numpy())
        probabilities.append(output.negative_probability.float().cpu().numpy())
        logits.append(output.sign_logits.float().cpu().numpy())
        reverse_probabilities.append(torch.sigmoid(-output.sign_logits).float().cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(probabilities),
        np.concatenate(logits),
        np.concatenate(reverse_probabilities),
    )


def _metrics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    probability: np.ndarray,
    logits: np.ndarray,
    reverse_probability: np.ndarray,
    *,
    minimum_negatives: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    metrics: dict[str, Any] = branch_metrics(
        target,
        prediction,
        frame["delta_t_s"].to_numpy(dtype=np.float64),
    )
    per_sequence = _per_sequence(frame, prediction, minimum_negatives=minimum_negatives)
    reverse_target = target >= 0.0
    reverse_prediction = reverse_probability >= 0.5
    metrics["reverse_sign_accuracy"] = float(np.mean(reverse_prediction == reverse_target))
    metrics["antisymmetry_mean_abs"] = float(
        np.mean(np.abs(probability + reverse_probability - 1.0))
    )
    metrics["minimum_sequence_negative_accuracy"] = per_sequence.attrs[
        "minimum_sequence_negative_accuracy"
    ]
    metrics["minimum_sequence_pearson"] = per_sequence.attrs["minimum_sequence_pearson"]
    metrics["negative_probability_mean"] = float(np.mean(probability))
    metrics["sign_logit_abs_mean"] = float(np.mean(np.abs(logits)))
    metrics["official_eap"] = official_eap_metrics(
        target,
        prediction,
        frame["delta_t_s"].to_numpy(dtype=np.float64),
        frame["target_ttc_s"].to_numpy(dtype=np.float64),
    )
    return metrics, per_sequence


def _thresholds(config: ObjectEventV412TrainConfig) -> dict[str, float]:
    return {
        field.name: float(getattr(config, field.name))
        for field in fields(config)
        if field.name.endswith("gate") or field.name.endswith("ceiling") or field.name.endswith("floor") or field.name.endswith("drop") or field.name.endswith("tolerance")
    }


def _selection_objective(metrics: Mapping[str, Any], *, mode: str) -> float:
    if mode == "overfit":
        return (
            80.0 * (1.0 - float(metrics["balanced_sign_accuracy"]))
            + 80.0 * (1.0 - float(metrics["negative_accuracy"]))
            + 50.0 * (1.0 - float(metrics["reverse_sign_accuracy"]))
            + 20.0 * float(metrics["antisymmetry_mean_abs"])
        )
    return (
        100.0 * (1.0 - float(metrics["minimum_sequence_negative_accuracy"]))
        + 60.0 * (1.0 - float(metrics["negative_accuracy"]))
        + 40.0 * (1.0 - float(metrics["balanced_sign_accuracy"]))
        + 30.0 * (1.0 - float(metrics["pearson"]))
        + 500.0 * float(metrics["expansion_mae"])
    )


def run(
    *,
    cache_manifest: Path,
    v48_config_path: Path,
    probe_config_path: Path,
    v48_checkpoint: Path,
    ensemble_train_path: Path,
    ensemble_validation_path: Path,
    output_dir: Path,
    device_name: str,
    mode: str,
    force: bool,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    if mode not in {"overfit", "screen"}:
        raise ValueError("mode must be overfit or screen")
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    probe_config, train_config, loss_config = _load_probe_config(probe_config_path)
    _seed(train_config.seed)
    device = _resolve_device(device_name)
    backbone, backbone_payload = _load_backbone(
        v48_config_path=v48_config_path,
        checkpoint_path=v48_checkpoint,
    )
    model = ObjectEventTTCV412(backbone, probe_config).to(device)
    trainable = list(model.sign_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    base_config, _, _, _, _ = _load_v48_config(v48_config_path)
    train_split, train_manifest = _materialize(
        cache_manifest, "train", input_size=base_config.input_size
    )
    validation_split, validation_manifest = _materialize(
        cache_manifest, "validation", input_size=base_config.input_size
    )
    train_frame = _align_ensemble(train_split, _read_ensemble(ensemble_train_path))
    validation_frame = _align_ensemble(
        validation_split, _read_ensemble(ensemble_validation_path)
    )

    if mode == "overfit":
        indices = _balanced_subset_indices(
            train_frame,
            train_config.overfit_samples,
            train_config.seed,
        )
        train_split = train_split.subset(indices)
        train_frame = train_frame.iloc[indices.tolist()].reset_index(drop=True)
        validation_split = train_split
        validation_frame = train_frame.copy()
        maximum_epochs = train_config.overfit_maximum_epochs
        minimum_epochs = 1
        patience_epochs = maximum_epochs
    else:
        maximum_epochs = train_config.maximum_epochs
        minimum_epochs = train_config.minimum_epochs
        patience_epochs = train_config.patience_epochs

    weights = _sample_weights(train_frame)
    best_path = output_dir / "best_observed.pt"
    gate_path = output_dir / "best_gate_passing.pt"
    best_objective = float("inf")
    best_gate_objective = float("inf")
    best_epoch = 0
    best_gate_epoch = 0
    gate_count = 0
    without_improvement = 0
    history: list[dict[str, Any]] = []
    thresholds = _thresholds(train_config)

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        indices = torch.multinomial(weights, len(train_split), replacement=True)
        total_loss = 0.0
        total_examples = 0
        components = Counter()
        for start in range(0, len(train_split), train_config.batch_size):
            batch_indices = indices[start : start + train_config.batch_size]
            events = train_split.events[batch_indices].to(device=device, dtype=torch.float32)
            target = torch.as_tensor(
                train_frame["target_expansion"].to_numpy(dtype=np.float32)[batch_indices.numpy()],
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, train_config.precision):
                original_logits, reversed_logits, _, _ = model.paired_sign_logits(events)
                loss_output = reversal_balanced_sign_loss(
                    original_logits,
                    reversed_logits,
                    target,
                    config=loss_config,
                )
            loss_output.total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, train_config.max_grad_norm)
            optimizer.step()
            size = int(events.shape[0])
            total_examples += size
            total_loss += float(loss_output.total.detach().cpu()) * size
            for name, value in loss_output.components.items():
                components[name] += float(value.detach().cpu()) * size

        prediction, probability, logits, reverse_probability = _predict(
            model,
            validation_split,
            validation_frame,
            batch_size=train_config.batch_size,
            device=device,
            threshold=train_config.negative_threshold,
            branch="event",
            shuffle_seed=train_config.seed + epoch,
        )
        metrics, _ = _metrics(
            validation_frame,
            prediction,
            probability,
            logits,
            reverse_probability,
            minimum_negatives=train_config.per_sequence_negative_min_count,
        )
        baseline = branch_metrics(
            validation_frame["target_expansion"].to_numpy(dtype=np.float64),
            validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64),
            validation_frame["delta_t_s"].to_numpy(dtype=np.float64),
        )
        objective = _selection_objective(metrics, mode=mode)
        gates = directional_sign_checkpoint_gates(
            mode=mode,
            metrics=cast(Mapping[str, float], metrics),
            baseline=baseline,
            thresholds=thresholds,
        )
        passed = all(gates.values())
        payload = {
            "artifact_type": "object_event_v4_12_checkpoint",
            "epoch": epoch,
            "sign_head_state_dict": model.sign_head.state_dict(),
            "selection_objective": objective,
            "validation_metrics": metrics,
            "validation_gates": gates,
            "v48_checkpoint": v48_checkpoint.resolve().as_posix(),
        }
        if objective < best_objective - 1.0e-8:
            best_objective = objective
            best_epoch = epoch
            without_improvement = 0
            torch.save(payload, best_path)
        else:
            without_improvement += 1
        if passed:
            gate_count += 1
            if objective < best_gate_objective - 1.0e-8:
                best_gate_objective = objective
                best_gate_epoch = epoch
                without_improvement = 0
                torch.save(payload, gate_path)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "train_components": {
                name: value / max(total_examples, 1) for name, value in components.items()
            },
            "validation_metrics": metrics,
            "validation_gates": gates,
            "epoch_gate_passed": passed,
            "best_epoch": best_epoch,
            "best_gate_epoch": best_gate_epoch,
        }
        history.append(cast(dict[str, Any], _json_safe(row)))
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history[-1], sort_keys=True) + "\n")
        if epoch >= minimum_epochs and without_improvement >= patience_epochs:
            break

    selected_path = gate_path if gate_path.exists() else best_path
    selected_reason = "gate_passing" if gate_path.exists() else "best_observed"
    selected_payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    model.sign_head.load_state_dict(selected_payload["sign_head_state_dict"], strict=True)
    model.to(device)

    baseline_prediction = validation_frame["fused_prediction_expansion"].to_numpy(dtype=np.float64)
    baseline_metrics = branch_metrics(
        validation_frame["target_expansion"].to_numpy(dtype=np.float64),
        baseline_prediction,
        validation_frame["delta_t_s"].to_numpy(dtype=np.float64),
    )
    baseline_per_sequence = _per_sequence(
        validation_frame,
        baseline_prediction,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )
    baseline_metrics["minimum_sequence_negative_accuracy"] = baseline_per_sequence.attrs[
        "minimum_sequence_negative_accuracy"
    ]
    baseline_metrics["minimum_sequence_pearson"] = baseline_per_sequence.attrs[
        "minimum_sequence_pearson"
    ]
    baseline_metrics["official_eap"] = official_eap_metrics(
        validation_frame["target_expansion"].to_numpy(dtype=np.float64),
        baseline_prediction,
        validation_frame["delta_t_s"].to_numpy(dtype=np.float64),
        validation_frame["target_ttc_s"].to_numpy(dtype=np.float64),
    )

    event_prediction, probability, logits, reverse_probability = _predict(
        model,
        validation_split,
        validation_frame,
        batch_size=train_config.batch_size,
        device=device,
        threshold=train_config.negative_threshold,
        branch="event",
        shuffle_seed=train_config.seed + 1200,
    )
    zero_prediction, _, _, _ = _predict(
        model,
        validation_split,
        validation_frame,
        batch_size=train_config.batch_size,
        device=device,
        threshold=train_config.negative_threshold,
        branch="zero",
        shuffle_seed=train_config.seed + 1201,
    )
    shuffled_prediction, _, _, _ = _predict(
        model,
        validation_split,
        validation_frame,
        batch_size=train_config.batch_size,
        device=device,
        threshold=train_config.negative_threshold,
        branch="shuffled",
        shuffle_seed=train_config.seed + 1202,
    )
    final_metrics, per_sequence = _metrics(
        validation_frame,
        event_prediction,
        probability,
        logits,
        reverse_probability,
        minimum_negatives=train_config.per_sequence_negative_min_count,
    )
    target = validation_frame["target_expansion"].to_numpy(dtype=np.float64)
    final_metrics["zero_event_pearson_drop"] = pearson(target, event_prediction) - pearson(
        target, zero_prediction
    )
    final_metrics["shuffled_event_pearson_drop"] = pearson(
        target, event_prediction
    ) - pearson(target, shuffled_prediction)
    final_gates = directional_sign_gates(
        mode=mode,
        metrics=cast(Mapping[str, float], final_metrics),
        baseline=cast(Mapping[str, float], baseline_metrics),
        thresholds=thresholds,
    )
    final_passed = all(final_gates.values())

    output = validation_frame.loc[:, list(IDENTITY_COLUMNS) + [
        "delta_t_s",
        "target_ttc_s",
        "target_expansion",
    ]].copy()
    output["baseline_prediction_expansion"] = baseline_prediction
    output["directional_prediction_expansion"] = event_prediction
    output["negative_probability"] = probability
    output["reverse_negative_probability"] = reverse_probability
    output["zero_events_prediction_expansion"] = zero_prediction
    output["shuffled_prediction_expansion"] = shuffled_prediction
    output.to_csv(output_dir / "validation_predictions.csv", index=False)
    per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)

    summary: dict[str, Any] = {
        "artifact_type": "object_event_v4_12_reversal_balanced_directional_sign",
        "status": f"{mode}_passed" if final_passed else f"{mode}_failed",
        "mode": mode,
        "created_at": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cache_manifest": cache_manifest.resolve().as_posix(),
        "cache_manifest_sha256": _sha256(cache_manifest),
        "v48_config": v48_config_path.resolve().as_posix(),
        "probe_config": probe_config_path.resolve().as_posix(),
        "v48_checkpoint": v48_checkpoint.resolve().as_posix(),
        "v48_checkpoint_sha256": _sha256(v48_checkpoint),
        "v48_checkpoint_epoch": backbone_payload.get("epoch"),
        "probe_model_config": asdict(probe_config),
        "train_config": asdict(train_config),
        "loss_config": asdict(loss_config),
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
        "completed_epochs": len(history),
        "best_observed_epoch": best_epoch,
        "best_gate_epoch": best_gate_epoch,
        "gate_passing_epoch_count": gate_count,
        "selected_checkpoint": selected_path.name,
        "selection_reason": selected_reason,
        "baseline_validation_metrics": baseline_metrics,
        "directional_validation_metrics": final_metrics,
        "gates": final_gates,
        "passed": final_passed,
        "scientific_contract": {
            "event_only_sign_probe": True,
            "v48_backbone_frozen": True,
            "magnitude_source_is_fixed_v410_ensemble": True,
            "sign_probe_trained_on_train_split_only": True,
            "temporal_reversal_has_opposite_sign_target": True,
            "exact_descriptor_antisymmetry": True,
            "odd_bias_free_sign_head": True,
            "checkpoint_selection_uses_core_gates_only": True,
            "final_screen_still_requires_event_dependence": True,
            "boxes_and_visible_heights_not_probe_inputs": True,
            "sequence_and_track_ids_not_probe_features": True,
            "validation_is_development_only": True,
            "official_eap_test_not_opened": True,
            "evttc_not_opened": True,
            "advance_to_integrated_dual_head": final_passed,
        },
    }
    summary = cast(dict[str, Any], _json_safe(summary))
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument(
        "--v48-config",
        type=Path,
        default=Path("configs/experiment/e_jepa_garl_object_event_dense_motion_v4_8.yaml"),
    )
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=Path(
            "configs/experiment/e_jepa_garl_object_event_directional_sign_probe_v4_12.yaml"
        ),
    )
    parser.add_argument("--v48-checkpoint", type=Path, required=True)
    parser.add_argument("--ensemble-train", type=Path, required=True)
    parser.add_argument("--ensemble-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("overfit", "screen"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        result = run(
            cache_manifest=args.cache_manifest,
            v48_config_path=args.v48_config,
            probe_config_path=args.probe_config,
            v48_checkpoint=args.v48_checkpoint,
            ensemble_train_path=args.ensemble_train,
            ensemble_validation_path=args.ensemble_validation,
            output_dir=args.output_dir,
            device_name=args.device,
            mode=args.mode,
            force=args.force,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_12_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0 if bool(result["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
