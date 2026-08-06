#!/usr/bin/env python3
"""Train Object Event TTC v4.7 high-resolution foreground extent screen."""

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
    _branch_metrics,
    _evaluate,
    _git_commit,
    _json_safe,
    _pearson,
    _resolve_device,
    _seed,
    _sha256,
)
from scripts.train_e_jepa_object_event_v4_6 import (  # noqa: E402
    MaterializedV46Split,
    _balanced_overfit_subset,
    _materialize,
    _sampling_weights,
)
from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config  # noqa: E402
from e_jepa_ttc.models.object_event_v4_7 import (  # noqa: E402
    ObjectEventTTCV47,
    ObjectEventV47Config,
)
from e_jepa_ttc.object_event_v4_4 import official_eap_metrics  # noqa: E402
from e_jepa_ttc.training.object_event_v4_6 import boxes_to_feature_masks  # noqa: E402
from e_jepa_ttc.training.object_event_v4_7 import (  # noqa: E402
    ObjectEventV47LossConfig,
    object_event_v4_7_loss,
)


@dataclass(frozen=True)
class ObjectEventV47TrainConfig:
    batch_size: int = 24
    maximum_epochs: int = 18
    minimum_epochs: int = 6
    patience_epochs: int = 6
    encoder_learning_rate: float = 2.0e-5
    decoder_learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 2.0
    precision: str = "fp32"
    seed: int = 7
    shuffle_repeats_during_training: int = 2
    shuffle_repeats_final: int = 8
    bootstrap_repeats_during_training: int = 250
    bootstrap_repeats_final: int = 2000
    overfit_samples: int = 64
    overfit_maximum_epochs: int = 100
    overfit_pearson_gate: float = 0.97
    overfit_balanced_sign_gate: float = 0.95
    overfit_height_pearson_gate: float = 0.97
    overfit_mask_iou_gate: float = 0.80
    screen_height_pearson_gate: float = 0.50
    screen_min_sequence_height_pearson_gate: float = 0.25
    screen_expansion_pearson_gate: float = 0.50
    screen_balanced_sign_gate: float = 0.68
    screen_negative_accuracy_gate: float = 0.55
    screen_mask_iou_gate: float = 0.70
    zero_event_pearson_drop_gate: float = 0.40
    shuffled_event_pearson_drop_gate: float = 0.40
    per_sequence_negative_min_count: int = 20

    def __post_init__(self) -> None:
        integers = (
            self.batch_size,
            self.maximum_epochs,
            self.minimum_epochs,
            self.patience_epochs,
            self.shuffle_repeats_during_training,
            self.shuffle_repeats_final,
            self.bootstrap_repeats_during_training,
            self.bootstrap_repeats_final,
            self.overfit_samples,
            self.overfit_maximum_epochs,
            self.per_sequence_negative_min_count,
        )
        if min(integers) <= 0:
            raise ValueError("v4.7 integer controls must be positive")
        if self.minimum_epochs > self.maximum_epochs:
            raise ValueError("minimum_epochs exceeds maximum_epochs")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if min(self.encoder_learning_rate, self.decoder_learning_rate) <= 0.0:
            raise ValueError("v4.7 learning rates must be positive")


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_config(path: Path) -> tuple[
    ObjectEventV41Config,
    ObjectEventV47Config,
    ObjectEventV47TrainConfig,
    ObjectEventV47LossConfig,
]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.7 config must be a mapping")
    return (
        _construct(ObjectEventV41Config, cast(Mapping[str, Any], raw.get("model", {}))),
        _construct(ObjectEventV47Config, cast(Mapping[str, Any], raw.get("geometry", {}))),
        _construct(ObjectEventV47TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))),
        _construct(ObjectEventV47LossConfig, cast(Mapping[str, Any], raw.get("loss", {}))),
    )


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _load_v46_geometry(
    model: ObjectEventTTCV47,
    checkpoint: Path,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid v4.6 checkpoint: {checkpoint}")
    state = cast(dict[str, torch.Tensor], payload["model_state_dict"])
    model.load_v46_geometry_state_dict(state)
    return cast(dict[str, Any], payload)


@torch.no_grad()
def _geometry_predictions(
    model: ObjectEventTTCV47,
    split: MaterializedV46Split,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    height_log_eta: list[np.ndarray] = []
    log_heights: list[np.ndarray] = []
    soft_ious: list[np.ndarray] = []
    top_positions: list[np.ndarray] = []
    bottom_positions: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(split), batch_size):
        end = min(start + batch_size, len(split))
        events = split.events[start:end].to(device=device, dtype=torch.float32)
        boxes = split.boxes_xyxy[start:end].to(device=device, dtype=torch.float32)
        output = model(events)
        masks = boxes_to_feature_masks(
            boxes,
            source_height=split.source_height,
            source_width=split.source_width,
            target_height=output.foreground_logits.shape[-2],
            target_width=output.foreground_logits.shape[-1],
        )
        probability = output.foreground_probabilities[:, 1:3]
        intersection = (probability * masks).sum(dim=(-2, -1))
        union = probability.sum(dim=(-2, -1)) + masks.sum(dim=(-2, -1)) - intersection
        soft_iou = (intersection + 1.0e-6) / (union + 1.0e-6)
        height_log_eta.append(output.height_log_eta.float().cpu().numpy())
        log_heights.append(output.predicted_log_heights.float().cpu().numpy())
        soft_ious.append(soft_iou.mean(dim=1).float().cpu().numpy())
        top_positions.append(output.top_positions.float().cpu().numpy())
        bottom_positions.append(output.bottom_positions.float().cpu().numpy())
    return {
        "height_log_eta": np.concatenate(height_log_eta),
        "log_heights": np.concatenate(log_heights),
        "soft_iou": np.concatenate(soft_ious),
        "top_positions": np.concatenate(top_positions),
        "bottom_positions": np.concatenate(bottom_positions),
    }


def _per_sequence_height_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sequence_id, frame in rows.groupby("sequence_id", sort=True):
        target = frame["target_height_log_eta"].to_numpy(dtype=np.float64)
        prediction = frame["height_log_eta"].to_numpy(dtype=np.float64)
        records.append(
            {
                "sequence_id": sequence_id,
                "height_count": len(frame),
                "height_log_eta_pearson": _pearson(target, prediction),
                "height_log_eta_mae": float(np.mean(np.abs(target - prediction))),
                "foreground_soft_iou": float(frame["foreground_soft_iou"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def _evaluate_v47(
    model: ObjectEventTTCV47,
    split: MaterializedV46Split,
    *,
    train_config: ObjectEventV47TrainConfig,
    loss_config: ObjectEventV47LossConfig,
    device: torch.device,
    final: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows, per_sequence, metrics = _evaluate(
        model,
        split.base_view(),
        batch_size=train_config.batch_size,
        device=device,
        max_abs_expansion=loss_config.max_abs_expansion,
        shuffle_repeats=(
            train_config.shuffle_repeats_final
            if final
            else train_config.shuffle_repeats_during_training
        ),
        bootstrap_repeats=(
            train_config.bootstrap_repeats_final
            if final
            else train_config.bootstrap_repeats_during_training
        ),
        seed=seed,
    )
    extra = _geometry_predictions(
        model, split, batch_size=train_config.batch_size, device=device
    )
    target_expansion = rows["target_expansion"].to_numpy(dtype=np.float64)
    prediction = rows["prediction_expansion"].to_numpy(dtype=np.float64)
    delta_t = rows["delta_t_s"].to_numpy(dtype=np.float64)
    target_ttc = rows["target_ttc_s"].to_numpy(dtype=np.float64)
    target_height = (
        torch.log(split.visible_heights_px[:, 0])
        - torch.log(split.visible_heights_px[:, 1])
    ).numpy()
    target_log_eta = np.log1p(np.clip(-target_expansion, -0.999999, None))
    rows["target_height_log_eta"] = target_height
    rows["target_log_eta"] = target_log_eta
    rows["height_log_eta"] = extra["height_log_eta"]
    rows["foreground_soft_iou"] = extra["soft_iou"]
    rows["predicted_log_height_t1"] = extra["log_heights"][:, 1]
    rows["predicted_log_height_t2"] = extra["log_heights"][:, 2]
    rows["top_t1"] = extra["top_positions"][:, 1]
    rows["top_t2"] = extra["top_positions"][:, 2]
    rows["bottom_t1"] = extra["bottom_positions"][:, 1]
    rows["bottom_t2"] = extra["bottom_positions"][:, 2]

    height_per_sequence = _per_sequence_height_metrics(rows)
    per_sequence = per_sequence.merge(height_per_sequence, on="sequence_id", how="left")
    negative_counts = (
        rows.assign(negative=rows["target_expansion"] < 0.0)
        .groupby("sequence_id")["negative"]
        .sum()
        .astype(int)
    )
    per_sequence["negative_count"] = (
        per_sequence["sequence_id"].map(negative_counts).fillna(0).astype(int)
    )
    eligible = per_sequence[
        per_sequence["negative_count"] >= train_config.per_sequence_negative_min_count
    ]
    augmented = dict(metrics)
    augmented["official_eap"] = official_eap_metrics(
        target_expansion,
        prediction,
        delta_t,
        target_ttc,
        max_abs_expansion=loss_config.max_abs_expansion,
    )
    augmented["geometry"] = {
        "height_log_eta_pearson": _pearson(target_height, extra["height_log_eta"]),
        "height_log_eta_mae": float(
            np.mean(np.abs(target_height - extra["height_log_eta"]))
        ),
        "foreground_soft_iou": float(np.mean(extra["soft_iou"])),
        "minimum_sequence_height_pearson": float(
            height_per_sequence["height_log_eta_pearson"].min()
        ),
        "macro_sequence_height_pearson": float(
            height_per_sequence["height_log_eta_pearson"].mean()
        ),
        "oracle_height_to_ttc_log_eta_pearson": _pearson(
            target_log_eta, target_height
        ),
    }
    augmented["per_sequence"]["minimum_eligible_negative_accuracy"] = (
        float(eligible["negative_accuracy"].min()) if not eligible.empty else 0.0
    )
    augmented["per_sequence"]["eligible_negative_sequence_count"] = int(len(eligible))
    return rows, per_sequence, cast(dict[str, Any], _json_safe(augmented))


def _selection_objective(metrics: Mapping[str, Any], *, mode: str) -> float:
    """Rank checkpoints using only metrics meaningful for the current mode.

    The overfit subset contains very few examples from some sequences. Per-sequence
    Pearson can therefore be undefined or equal to -1.0 and must not dominate the
    checkpoint choice. Screen mode keeps the stricter sequence-level terms.
    """
    event = cast(Mapping[str, object], metrics["event"])
    geometry = cast(Mapping[str, object], metrics["geometry"])
    common = (
        80.0 * (1.0 - _finite(geometry.get("height_log_eta_pearson")))
        + 50.0 * (1.0 - _finite(event.get("pearson")))
        + 40.0 * (1.0 - _finite(event.get("balanced_sign_accuracy")))
        + 40.0 * (1.0 - _finite(geometry.get("foreground_soft_iou")))
        + 500.0 * _finite(geometry.get("height_log_eta_mae"), 1.0)
    )
    if mode == "overfit":
        return common
    if mode != "screen":
        raise ValueError(f"Unknown v4.7 mode: {mode}")
    sequence = cast(Mapping[str, object], metrics["per_sequence"])
    return (
        common
        + 50.0 * (1.0 - _finite(geometry.get("minimum_sequence_height_pearson")))
        + 30.0 * (1.0 - _finite(sequence.get("minimum_eligible_negative_accuracy")))
    )


def _epoch_gates(
    metrics: Mapping[str, Any],
    config: ObjectEventV47TrainConfig,
    *,
    mode: str,
) -> dict[str, bool]:
    if mode == "overfit":
        return _overfit_gates(metrics, config)
    if mode == "screen":
        return _screen_gates(metrics, config)
    raise ValueError(f"Unknown v4.7 mode: {mode}")


def _overfit_gates(
    metrics: Mapping[str, Any], config: ObjectEventV47TrainConfig
) -> dict[str, bool]:
    event = cast(Mapping[str, object], metrics["event"])
    geometry = cast(Mapping[str, object], metrics["geometry"])
    return {
        "expansion_pearson": _finite(event.get("pearson")) >= config.overfit_pearson_gate,
        "balanced_sign": _finite(event.get("balanced_sign_accuracy"))
        >= config.overfit_balanced_sign_gate,
        "height_ratio": _finite(geometry.get("height_log_eta_pearson"))
        >= config.overfit_height_pearson_gate,
        "foreground": _finite(geometry.get("foreground_soft_iou"))
        >= config.overfit_mask_iou_gate,
    }


def _screen_gates(
    metrics: Mapping[str, Any], config: ObjectEventV47TrainConfig
) -> dict[str, bool]:
    event = cast(Mapping[str, object], metrics["event"])
    geometry = cast(Mapping[str, object], metrics["geometry"])
    dependence = cast(Mapping[str, object], metrics["event_dependence"])
    return {
        "height_ratio": _finite(geometry.get("height_log_eta_pearson"))
        >= config.screen_height_pearson_gate,
        "minimum_sequence_height_ratio": _finite(
            geometry.get("minimum_sequence_height_pearson")
        )
        >= config.screen_min_sequence_height_pearson_gate,
        "expansion_pearson": _finite(event.get("pearson"))
        >= config.screen_expansion_pearson_gate,
        "balanced_sign": _finite(event.get("balanced_sign_accuracy"))
        >= config.screen_balanced_sign_gate,
        "negative_accuracy": _finite(event.get("negative_accuracy"))
        >= config.screen_negative_accuracy_gate,
        "foreground": _finite(geometry.get("foreground_soft_iou"))
        >= config.screen_mask_iou_gate,
        "zero_event_dependence": _finite(dependence.get("zero_event_pearson_drop"))
        >= config.zero_event_pearson_drop_gate,
        "shuffled_event_dependence": _finite(
            dependence.get("shuffled_event_pearson_drop")
        )
        >= config.shuffled_event_pearson_drop_gate,
    }


def _optimizer(
    model: ObjectEventTTCV47,
    config: ObjectEventV47TrainConfig,
) -> torch.optim.Optimizer:
    encoder = [p for p in model.geometry_encoder.parameters() if p.requires_grad]
    heads = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("geometry_encoder.")
    ]
    return torch.optim.AdamW(
        [
            {"params": encoder, "lr": config.encoder_learning_rate},
            {"params": heads, "lr": config.decoder_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )


def run(
    *,
    cache_manifest: Path,
    config_path: Path,
    initial_v46_checkpoint: Path,
    output_dir: Path,
    device_name: str,
    mode: str,
    force: bool,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    base_config, geometry_config, train_config, loss_config = _load_config(config_path)
    _seed(train_config.seed)
    device = _resolve_device(device_name)
    train_split, train_manifest = _materialize(
        cache_manifest, "train", input_size=base_config.input_size
    )
    validation_split, validation_manifest = _materialize(
        cache_manifest, "validation", input_size=base_config.input_size
    )
    if mode == "overfit":
        train_split = _balanced_overfit_subset(
            train_split, train_config.overfit_samples, train_config.seed
        )
        validation_split = train_split
        maximum_epochs = train_config.overfit_maximum_epochs
        patience_epochs = maximum_epochs
        minimum_epochs = 1
    else:
        maximum_epochs = train_config.maximum_epochs
        patience_epochs = train_config.patience_epochs
        minimum_epochs = train_config.minimum_epochs

    model = ObjectEventTTCV47(base_config, geometry_config)
    initial_payload = _load_v46_geometry(model, initial_v46_checkpoint)
    model.to(device)
    optimizer = _optimizer(model, train_config)
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and train_config.precision == "fp16"
        else None
    )
    weights = _sampling_weights(train_split)
    best_path = output_dir / "best_observed.pt"
    gate_path = output_dir / "best_gate_passing.pt"
    best_objective = float("inf")
    best_epoch = 0
    best_gate_objective = float("inf")
    best_gate_epoch = 0
    gate_passing_epoch_count = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        indices = torch.multinomial(weights, num_samples=len(train_split), replacement=True)
        components = Counter()
        epoch_loss = 0.0
        examples = 0
        for start in range(0, len(train_split), train_config.batch_size):
            batch_indices = indices[start : start + train_config.batch_size]
            events = train_split.events[batch_indices].to(device=device, dtype=torch.float32)
            delta_t = train_split.delta_t_s[batch_indices].to(device)
            target_ttc = train_split.target_ttc_s[batch_indices].to(device)
            heights = train_split.visible_heights_px[batch_indices].to(device)
            boxes = train_split.boxes_xyxy[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, train_config.precision):
                output = model(events)
                loss_output = object_event_v4_7_loss(
                    output,
                    delta_t,
                    target_ttc,
                    heights,
                    boxes,
                    source_height=train_split.source_height,
                    source_width=train_split.source_width,
                    config=loss_config,
                )
            if scaler is not None:
                scaler.scale(loss_output.total).backward()
                scaler.unscale_(optimizer)
            else:
                loss_output.total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                train_config.max_grad_norm,
            )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            size = int(events.shape[0])
            examples += size
            epoch_loss += float(loss_output.total.detach().cpu()) * size
            for name, value in loss_output.components.items():
                components[name] += float(value.detach().cpu()) * size
            global_step += 1

        _, _, validation_metrics = _evaluate_v47(
            model,
            validation_split,
            train_config=train_config,
            loss_config=loss_config,
            device=device,
            final=False,
            seed=train_config.seed + epoch * 37,
        )
        objective = _selection_objective(validation_metrics, mode=mode)
        epoch_gates = _epoch_gates(validation_metrics, train_config, mode=mode)
        epoch_gate_passed = all(epoch_gates.values())
        if objective < best_objective - 1.0e-6:
            best_objective = objective
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "artifact_type": "object_event_v4_7_best_observed",
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                    "selection_objective": objective,
                    "initial_v46_checkpoint": initial_v46_checkpoint.resolve().as_posix(),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        if epoch_gate_passed:
            gate_passing_epoch_count += 1
            if objective < best_gate_objective - 1.0e-6:
                best_gate_objective = objective
                best_gate_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "artifact_type": "object_event_v4_7_best_gate_passing",
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "validation_metrics": validation_metrics,
                        "validation_gates": epoch_gates,
                        "selection_objective": objective,
                        "initial_v46_checkpoint": initial_v46_checkpoint.resolve().as_posix(),
                    },
                    gate_path,
                )
        event = cast(Mapping[str, object], validation_metrics["event"])
        geometry = cast(Mapping[str, object], validation_metrics["geometry"])
        row = cast(
            dict[str, Any],
            _json_safe(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": epoch_loss / max(examples, 1),
                    "train_components": {
                        name: value / max(examples, 1)
                        for name, value in components.items()
                    },
                    "validation_selection_objective": objective,
                    "validation": validation_metrics,
                    "epoch_gates": epoch_gates,
                    "epoch_gate_passed": epoch_gate_passed,
                    "best_epoch": best_epoch,
                    "best_objective": best_objective,
                    "best_gate_epoch": best_gate_epoch,
                    "best_gate_objective": (
                        best_gate_objective if best_gate_epoch > 0 else None
                    ),
                }
            ),
        )
        history.append(row)
        (output_dir / "history.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in history),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "loss": row["train_loss"],
                    "height_pearson": geometry.get("height_log_eta_pearson"),
                    "min_sequence_height_pearson": geometry.get(
                        "minimum_sequence_height_pearson"
                    ),
                    "expansion_pearson": event.get("pearson"),
                    "balanced_sign": event.get("balanced_sign_accuracy"),
                    "mask_iou": geometry.get("foreground_soft_iou"),
                    "epoch_gate_passed": epoch_gate_passed,
                    "best_epoch": best_epoch,
                    "best_gate_epoch": best_gate_epoch,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if epoch >= minimum_epochs and epochs_without_improvement >= patience_epochs:
            break

    if not best_path.exists():
        raise RuntimeError("v4.7 produced no checkpoint")
    selected_path = gate_path if gate_path.exists() else best_path
    selection_reason = "gate_passing" if gate_path.exists() else "best_observed"
    best_payload = torch.load(selected_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    train_rows, train_per_sequence, train_metrics = _evaluate_v47(
        model,
        train_split,
        train_config=train_config,
        loss_config=loss_config,
        device=device,
        final=True,
        seed=train_config.seed + 7001,
    )
    validation_rows, validation_per_sequence, validation_metrics = _evaluate_v47(
        model,
        validation_split,
        train_config=train_config,
        loss_config=loss_config,
        device=device,
        final=True,
        seed=train_config.seed + 9001,
    )
    train_rows.to_csv(output_dir / "train_predictions.csv", index=False)
    train_per_sequence.to_csv(output_dir / "train_per_sequence.csv", index=False)
    validation_rows.to_csv(output_dir / "validation_predictions.csv", index=False)
    validation_per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)
    gates = _epoch_gates(
        train_metrics if mode == "overfit" else validation_metrics,
        train_config,
        mode=mode,
    )
    passed = all(gates.values())
    summary = cast(
        dict[str, Any],
        _json_safe(
            {
                "artifact_type": "object_event_v4_7_highres_foreground_extent",
                "status": f"{mode}_passed" if passed else f"{mode}_failed",
                "mode": mode,
                "created_at": datetime.now(UTC).isoformat(),
                "started_at": started_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "git_commit": _git_commit(),
                "device": str(device),
                "gpu_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
                "cache_manifest": cache_manifest.resolve().as_posix(),
                "cache_manifest_sha256": _sha256(cache_manifest),
                "config": config_path.resolve().as_posix(),
                "initial_v46_checkpoint": initial_v46_checkpoint.resolve().as_posix(),
                "initial_v46_checkpoint_sha256": _sha256(initial_v46_checkpoint),
                "initial_v46_epoch": int(initial_payload.get("epoch", 0)),
                "base_config": asdict(base_config),
                "geometry_config": asdict(geometry_config),
                "train_config": asdict(train_config),
                "loss_config": asdict(loss_config),
                "train_split": train_manifest,
                "validation_split": validation_manifest,
                "completed_epochs": len(history),
                "best_observed_epoch": best_epoch,
                "best_gate_epoch": best_gate_epoch,
                "gate_passing_epoch_count": gate_passing_epoch_count,
                "selected_checkpoint": selected_path.name,
                "selection_reason": selection_reason,
                "best_epoch": int(best_payload.get("epoch", 0)),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "gates": gates,
                "passed": passed,
                "scientific_contract": {
                    "event_only_inference": True,
                    "boxes_are_training_only_targets": True,
                    "visible_heights_are_training_only_targets": True,
                    "content_dependent_scale_head_removed": True,
                    "ttc_fusion_removed": True,
                    "v46_geometry_encoder_initialisation": True,
                    "validation_is_not_official_eap_test": True,
                    "evttc_not_opened": True,
                    "advance_to_fusion_screen": mode == "screen" and passed,
                },
            }
        ),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if passed:
        torch.save(best_payload, output_dir / "eligible.pt")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-v46-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("overfit", "screen"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            cache_manifest=args.cache_manifest.resolve(),
            config_path=args.config.resolve(),
            initial_v46_checkpoint=args.initial_v46_checkpoint.resolve(),
            output_dir=args.output_dir.resolve(),
            device_name=args.device,
            mode=args.mode,
            force=args.force,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_7_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if bool(result.get("passed")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
