#!/usr/bin/env python3
"""Train Object Event TTC v4.6 learned foreground height-ratio screen."""

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
    MaterializedSplit,
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
from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config  # noqa: E402
from e_jepa_ttc.models.object_event_v4_6 import (  # noqa: E402
    ObjectEventTTCV46,
    ObjectEventV46Config,
)
from e_jepa_ttc.object_event_v4_4 import official_eap_metrics  # noqa: E402
from e_jepa_ttc.training.object_event_v4_6 import (  # noqa: E402
    ObjectEventV46LossConfig,
    boxes_to_feature_masks,
    object_event_v4_6_loss,
)


@dataclass(frozen=True)
class ObjectEventV46TrainConfig:
    batch_size: int = 32
    maximum_epochs: int = 14
    minimum_epochs: int = 5
    patience_epochs: int = 5
    geometry_encoder_learning_rate: float = 5.0e-5
    head_learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 2.0
    precision: str = "fp32"
    seed: int = 7
    shuffle_repeats_during_training: int = 2
    shuffle_repeats_final: int = 8
    bootstrap_repeats_during_training: int = 250
    bootstrap_repeats_final: int = 2000
    overfit_samples: int = 64
    overfit_maximum_epochs: int = 80
    overfit_pearson_gate: float = 0.95
    overfit_balanced_sign_gate: float = 0.95
    overfit_height_pearson_gate: float = 0.95
    overfit_mask_iou_gate: float = 0.70
    screen_relative_mid_improvement_gate: float = 0.05
    screen_pearson_tolerance: float = 0.01
    screen_balanced_sign_tolerance: float = 0.01
    screen_min_sequence_negative_accuracy_gate: float = 0.20
    screen_min_sequence_negative_gain_gate: float = 0.10
    screen_height_pearson_gate: float = 0.35
    screen_mask_iou_gate: float = 0.55
    zero_event_pearson_drop_gate: float = 0.40
    shuffled_event_pearson_drop_gate: float = 0.40
    per_sequence_negative_min_count: int = 20

    def __post_init__(self) -> None:
        positive = (
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
        if min(positive) <= 0:
            raise ValueError("v4.6 integer controls must be positive")
        if self.minimum_epochs > self.maximum_epochs:
            raise ValueError("minimum_epochs exceeds maximum_epochs")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if min(self.geometry_encoder_learning_rate, self.head_learning_rate) <= 0.0:
            raise ValueError("v4.6 learning rates must be positive")


@dataclass
class MaterializedV46Split:
    events: torch.Tensor
    delta_t_s: torch.Tensor
    target_ttc_s: torch.Tensor
    visible_heights_px: torch.Tensor
    boxes_xyxy: torch.Tensor
    sequence_ids: list[str]
    sample_tokens: list[str]
    track_ids: list[str]
    source_height: int
    source_width: int

    def __len__(self) -> int:
        return int(self.events.shape[0])

    def base_view(self) -> MaterializedSplit:
        return MaterializedSplit(
            events=self.events,
            delta_t_s=self.delta_t_s,
            target_ttc_s=self.target_ttc_s,
            sequence_ids=self.sequence_ids,
            sample_tokens=self.sample_tokens,
            track_ids=self.track_ids,
        )

    def subset(self, indices: torch.Tensor) -> "MaterializedV46Split":
        values = indices.tolist()
        return MaterializedV46Split(
            events=self.events[indices],
            delta_t_s=self.delta_t_s[indices],
            target_ttc_s=self.target_ttc_s[indices],
            visible_heights_px=self.visible_heights_px[indices],
            boxes_xyxy=self.boxes_xyxy[indices],
            sequence_ids=[self.sequence_ids[index] for index in values],
            sample_tokens=[self.sample_tokens[index] for index in values],
            track_ids=[self.track_ids[index] for index in values],
            source_height=self.source_height,
            source_width=self.source_width,
        )


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_config(path: Path) -> tuple[
    ObjectEventV41Config,
    ObjectEventV46Config,
    ObjectEventV46TrainConfig,
    ObjectEventV46LossConfig,
]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.6 config must be a mapping")
    return (
        _construct(ObjectEventV41Config, cast(Mapping[str, Any], raw.get("model", {}))),
        _construct(ObjectEventV46Config, cast(Mapping[str, Any], raw.get("geometry", {}))),
        _construct(ObjectEventV46TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))),
        _construct(ObjectEventV46LossConfig, cast(Mapping[str, Any], raw.get("loss", {}))),
    )


def _materialize(
    manifest: Path,
    split: str,
    *,
    input_size: int,
) -> tuple[MaterializedV46Split, dict[str, Any]]:
    dataset = GarlTTCObjectEventV4Dataset(manifest.as_posix(), splits=(split,))
    count = len(dataset)
    events = torch.empty((count, 3, 12, input_size, input_size), dtype=torch.float16)
    delta_t = torch.empty(count, dtype=torch.float32)
    target_ttc = torch.empty(count, dtype=torch.float32)
    heights = torch.empty((count, 2), dtype=torch.float32)
    boxes = torch.empty((count, 3, 4), dtype=torch.float32)
    sequence_ids: list[str] = []
    sample_tokens: list[str] = []
    track_ids: list[str] = []
    source_shape: tuple[int, int] | None = None
    for index in range(count):
        record = dataset[index]
        item = torch.as_tensor(record["event_v4_common_roi"], dtype=torch.float32)
        if item.ndim != 4 or item.shape[:2] != (3, 12):
            raise ValueError(f"Expected [3,12,H,W], got {tuple(item.shape)}")
        current_shape = (int(item.shape[-2]), int(item.shape[-1]))
        if source_shape is None:
            source_shape = current_shape
        elif source_shape != current_shape:
            raise ValueError("v4.6 requires one cache ROI shape")
        if current_shape != (input_size, input_size):
            item = torch.nn.functional.interpolate(item, size=(input_size, input_size), mode="area")
        events[index].copy_(item.to(torch.float16))
        delta_t[index] = float(record["garl_delta_t_s"])
        target_ttc[index] = float(record["ttc_s"])
        heights[index].copy_(torch.as_tensor(record["garl_visible_heights_px"], dtype=torch.float32))
        boxes[index].copy_(torch.as_tensor(record["event_v4_boxes_xyxy"], dtype=torch.float32))
        sequence_ids.append(str(record["sequence_id"]))
        sample_tokens.append(str(record["sample_token"]))
        track_ids.append(str(record["track_id"]))
    if source_shape is None:
        raise ValueError(f"Empty split: {split}")
    if not torch.isfinite(heights).all() or bool((heights <= 0).any()):
        raise ValueError("v4.6 visible-height supervision must be finite and positive")
    signs = ["negative" if float(value) < 0.0 else "positive" for value in target_ttc]
    result = MaterializedV46Split(
        events=events,
        delta_t_s=delta_t,
        target_ttc_s=target_ttc,
        visible_heights_px=heights,
        boxes_xyxy=boxes,
        sequence_ids=sequence_ids,
        sample_tokens=sample_tokens,
        track_ids=track_ids,
        source_height=source_shape[0],
        source_width=source_shape[1],
    )
    return result, {
        "split": split,
        "count": count,
        "event_shape": list(events.shape[1:]),
        "cache_roi_shape": list(source_shape),
        "sequence_counts": dict(sorted(Counter(sequence_ids).items())),
        "sign_counts": dict(sorted(Counter(signs).items())),
        "uses_boxes_as_model_input": False,
        "uses_visible_heights_as_model_input": False,
        "uses_boxes_as_training_target": True,
        "uses_visible_heights_as_training_target": True,
    }


def _balanced_overfit_subset(split: MaterializedV46Split, count: int, seed: int) -> MaterializedV46Split:
    if count % 2 != 0:
        raise ValueError("overfit_samples must be even")
    target = split.target_ttc_s.numpy()
    positive = np.flatnonzero(target > 0.0)
    negative = np.flatnonzero(target < 0.0)
    half = count // 2
    if len(positive) < half or len(negative) < half:
        raise ValueError("Insufficient positive/negative examples for v4.6 overfit")
    rng = np.random.default_rng(seed)
    selected = np.concatenate((rng.choice(positive, half, replace=False), rng.choice(negative, half, replace=False)))
    rng.shuffle(selected)
    return split.subset(torch.from_numpy(selected.astype(np.int64)))


def _sampling_weights(split: MaterializedV46Split) -> torch.Tensor:
    cells = [
        (sequence, "negative" if float(ttc) < 0.0 else "positive")
        for sequence, ttc in zip(split.sequence_ids, split.target_ttc_s, strict=True)
    ]
    counts = Counter(cells)
    raw = np.asarray([1.0 / counts[cell] for cell in cells], dtype=np.float64)
    median = float(np.median(raw))
    if median > 0.0:
        raw = np.minimum(raw, median * 10.0)
    raw /= raw.sum()
    return torch.tensor(raw, dtype=torch.float64)


def _load_checkpoint(model: ObjectEventTTCV46, checkpoint: Path, expected_seed: int) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid v4.2 checkpoint: {checkpoint}")
    seed = int(cast(Mapping[str, Any], payload.get("train_config", {})).get("seed", -1))
    if seed != expected_seed:
        raise ValueError(f"Checkpoint seed mismatch: expected {expected_seed}, got {seed}")
    model.load_base_state_dict(cast(dict[str, torch.Tensor], payload["model_state_dict"]))
    return cast(dict[str, Any], payload)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


@torch.no_grad()
def _geometry_predictions(
    model: ObjectEventTTCV46,
    split: MaterializedV46Split,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    height_log_eta: list[np.ndarray] = []
    height_expansion: list[np.ndarray] = []
    blends: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    soft_ious: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(split), batch_size):
        end = min(start + batch_size, len(split))
        events = split.events[start:end].to(device=device, dtype=torch.float32)
        boxes = split.boxes_xyxy[start:end].to(device=device, dtype=torch.float32)
        output = model(events)
        targets = boxes_to_feature_masks(
            boxes,
            source_height=split.source_height,
            source_width=split.source_width,
            target_height=output.foreground_logits.shape[-2],
            target_width=output.foreground_logits.shape[-1],
        )
        probability = output.foreground_probabilities[:, 1:3]
        intersection = (probability * targets).sum(dim=(-2, -1))
        union = probability.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1)) - intersection
        soft_iou = (intersection + 1.0e-6) / (union + 1.0e-6)
        height_log_eta.append(output.height_log_eta.float().cpu().numpy())
        height_expansion.append(output.height_expansion.float().cpu().numpy())
        blends.append(output.blend.float().cpu().numpy())
        confidences.append(output.geometry_confidence.float().cpu().numpy())
        soft_ious.append(soft_iou.mean(dim=1).float().cpu().numpy())
    return {
        "height_log_eta": np.concatenate(height_log_eta),
        "height_expansion": np.concatenate(height_expansion),
        "blend": np.concatenate(blends),
        "confidence": np.concatenate(confidences),
        "soft_iou": np.concatenate(soft_ious),
    }


def _evaluate_v46(
    model: ObjectEventTTCV46,
    split: MaterializedV46Split,
    *,
    train_config: ObjectEventV46TrainConfig,
    loss_config: ObjectEventV46LossConfig,
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
        shuffle_repeats=(train_config.shuffle_repeats_final if final else train_config.shuffle_repeats_during_training),
        bootstrap_repeats=(train_config.bootstrap_repeats_final if final else train_config.bootstrap_repeats_during_training),
        seed=seed,
    )
    extra = _geometry_predictions(model, split, batch_size=train_config.batch_size, device=device)
    target_expansion = rows["target_expansion"].to_numpy(dtype=np.float64)
    delta_t = rows["delta_t_s"].to_numpy(dtype=np.float64)
    target_ttc = rows["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction = rows["prediction_expansion"].to_numpy(dtype=np.float64)
    target_height_log_eta = (
        torch.log(split.visible_heights_px[:, 0]) - torch.log(split.visible_heights_px[:, 1])
    ).numpy()
    rows["target_height_log_eta"] = target_height_log_eta
    rows["height_log_eta"] = cast(np.ndarray, extra["height_log_eta"])
    rows["height_expansion"] = cast(np.ndarray, extra["height_expansion"])
    rows["geometry_blend"] = cast(np.ndarray, extra["blend"])
    rows["geometry_confidence"] = cast(np.ndarray, extra["confidence"])
    rows["foreground_soft_iou"] = cast(np.ndarray, extra["soft_iou"])
    negative_counts = rows.assign(negative=rows["target_expansion"] < 0).groupby("sequence_id")["negative"].sum().astype(int)
    per_sequence["negative_count"] = per_sequence["sequence_id"].map(negative_counts).fillna(0).astype(int)
    eligible = per_sequence[per_sequence["negative_count"] >= train_config.per_sequence_negative_min_count]
    minimum_negative = float(eligible["negative_accuracy"].min()) if not eligible.empty else 0.0
    augmented = dict(metrics)
    augmented["official_eap"] = official_eap_metrics(
        target_expansion,
        prediction,
        delta_t,
        target_ttc,
        max_abs_expansion=loss_config.max_abs_expansion,
    )
    augmented["geometry"] = {
        "height_log_eta_pearson": _pearson(target_height_log_eta, cast(np.ndarray, extra["height_log_eta"])),
        "height_log_eta_mae": float(np.mean(np.abs(target_height_log_eta - cast(np.ndarray, extra["height_log_eta"])))),
        "height_expansion": _branch_metrics(
            target_expansion,
            cast(np.ndarray, extra["height_expansion"]),
            delta_t,
            ttc_clip=model.config.ttc_clip_seconds,
            min_expansion=model.config.min_abs_expansion_for_ttc,
        ),
        "foreground_soft_iou": float(np.mean(cast(np.ndarray, extra["soft_iou"]))),
        "blend_mean": float(np.mean(cast(np.ndarray, extra["blend"]))),
        "blend_p95": float(np.quantile(cast(np.ndarray, extra["blend"]), 0.95)),
        "confidence_mean": float(np.mean(cast(np.ndarray, extra["confidence"]))),
    }
    augmented["per_sequence"]["minimum_eligible_negative_accuracy"] = minimum_negative
    augmented["per_sequence"]["eligible_negative_sequence_count"] = int(len(eligible))
    return rows, per_sequence, cast(dict[str, Any], _json_safe(augmented))


def _selection_objective(metrics: Mapping[str, Any]) -> float:
    event = cast(Mapping[str, object], metrics["event"])
    official = cast(Mapping[str, object], metrics["official_eap"])
    geometry = cast(Mapping[str, object], metrics["geometry"])
    sequence = cast(Mapping[str, object], metrics["per_sequence"])
    return (
        _finite(official.get("weighted_mid"), 1.0e6)
        + 80.0 * (1.0 - _finite(event.get("pearson")))
        + 60.0 * (1.0 - _finite(event.get("balanced_sign_accuracy")))
        + 60.0 * (1.0 - _finite(sequence.get("minimum_eligible_negative_accuracy")))
        + 20.0 * (1.0 - _finite(geometry.get("height_log_eta_pearson")))
        + 20.0 * (1.0 - _finite(geometry.get("foreground_soft_iou")))
    )


def _screen_gates(
    metrics: Mapping[str, Any],
    baseline: Mapping[str, Any],
    config: ObjectEventV46TrainConfig,
) -> dict[str, bool]:
    event = cast(Mapping[str, object], metrics["event"])
    official = cast(Mapping[str, object], metrics["official_eap"])
    geometry = cast(Mapping[str, object], metrics["geometry"])
    sequence = cast(Mapping[str, object], metrics["per_sequence"])
    dependence = cast(Mapping[str, object], metrics["event_dependence"])
    base_event = cast(Mapping[str, object], baseline["event"])
    base_official = cast(Mapping[str, object], baseline["official_eap"])
    base_sequence = cast(Mapping[str, object], baseline["per_sequence"])
    current_mid = _finite(official.get("weighted_mid"), 1.0e6)
    baseline_mid = _finite(base_official.get("weighted_mid"), 1.0e6)
    relative = (baseline_mid - current_mid) / max(baseline_mid, 1.0e-8)
    current_min_negative = _finite(sequence.get("minimum_eligible_negative_accuracy"))
    base_min_negative = _finite(base_sequence.get("minimum_eligible_negative_accuracy"))
    return {
        "relative_mid_improvement": relative >= config.screen_relative_mid_improvement_gate,
        "pearson_preserved": _finite(event.get("pearson")) >= _finite(base_event.get("pearson")) - config.screen_pearson_tolerance,
        "balanced_sign_preserved": _finite(event.get("balanced_sign_accuracy")) >= _finite(base_event.get("balanced_sign_accuracy")) - config.screen_balanced_sign_tolerance,
        "minimum_sequence_negative_accuracy": current_min_negative >= config.screen_min_sequence_negative_accuracy_gate,
        "minimum_sequence_negative_gain": current_min_negative >= base_min_negative + config.screen_min_sequence_negative_gain_gate,
        "height_ratio_learned": _finite(geometry.get("height_log_eta_pearson")) >= config.screen_height_pearson_gate,
        "foreground_learned": _finite(geometry.get("foreground_soft_iou")) >= config.screen_mask_iou_gate,
        "zero_event_dependence": _finite(dependence.get("zero_event_pearson_drop")) >= config.zero_event_pearson_drop_gate,
        "shuffled_event_dependence": _finite(dependence.get("shuffled_event_pearson_drop")) >= config.shuffled_event_pearson_drop_gate,
    }


def _overfit_gates(metrics: Mapping[str, Any], config: ObjectEventV46TrainConfig) -> dict[str, bool]:
    event = cast(Mapping[str, object], metrics["event"])
    geometry = cast(Mapping[str, object], metrics["geometry"])
    return {
        "pearson": _finite(event.get("pearson")) >= config.overfit_pearson_gate,
        "balanced_sign": _finite(event.get("balanced_sign_accuracy")) >= config.overfit_balanced_sign_gate,
        "height_ratio": _finite(geometry.get("height_log_eta_pearson")) >= config.overfit_height_pearson_gate,
        "foreground": _finite(geometry.get("foreground_soft_iou")) >= config.overfit_mask_iou_gate,
    }


def run(
    *,
    cache_manifest: Path,
    config_path: Path,
    initial_checkpoint: Path,
    output_dir: Path,
    device_name: str,
    mode: str,
    force: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}; pass --force")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    started_at = datetime.now(UTC)
    base_config, geometry_config, train_config, loss_config = _load_config(config_path)
    _seed(train_config.seed)
    device = _resolve_device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    train_split, train_manifest = _materialize(cache_manifest, "train", input_size=base_config.input_size)
    validation_split, validation_manifest = _materialize(cache_manifest, "validation", input_size=base_config.input_size)
    if mode == "overfit":
        train_split = _balanced_overfit_subset(train_split, train_config.overfit_samples, train_config.seed)
        validation_split = train_split
        maximum_epochs = train_config.overfit_maximum_epochs
        minimum_epochs = maximum_epochs
        patience_epochs = maximum_epochs
    elif mode == "screen":
        maximum_epochs = train_config.maximum_epochs
        minimum_epochs = train_config.minimum_epochs
        patience_epochs = train_config.patience_epochs
    else:
        raise ValueError("mode must be overfit or screen")

    model = ObjectEventTTCV46(base_config, geometry_config).to(device)
    initial_payload = _load_checkpoint(model, initial_checkpoint, train_config.seed)
    model.freeze_base()
    model.set_base_only(True)
    _, _, baseline_metrics = _evaluate_v46(
        model,
        validation_split,
        train_config=train_config,
        loss_config=loss_config,
        device=device,
        final=False,
        seed=train_config.seed + 101,
    )
    baseline_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.set_base_only(False)

    encoder_parameters = [parameter for parameter in model.geometry_encoder.parameters() if parameter.requires_grad]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("geometry_encoder.") and not name.startswith("base.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": train_config.geometry_encoder_learning_rate},
            {"params": head_parameters, "lr": train_config.head_learning_rate},
        ],
        weight_decay=train_config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and train_config.precision == "fp16" else None
    weights = _sampling_weights(train_split)
    history: list[dict[str, Any]] = []
    best_objective = _selection_objective(baseline_metrics)
    best_epoch = 0
    best_base_only = True
    best_path = output_dir / "best_observed.pt"
    torch.save(
        {
            "artifact_type": "object_event_v4_6_best_observed",
            "epoch": 0,
            "base_only": True,
            "model_state_dict": baseline_state,
            "validation_metrics": baseline_metrics,
            "selection_objective": best_objective,
            "initial_checkpoint": initial_checkpoint.resolve().as_posix(),
        },
        best_path,
    )
    epochs_without_improvement = 0
    global_step = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        model.base.eval()
        indices = torch.multinomial(weights, num_samples=len(train_split), replacement=True)
        epoch_loss = 0.0
        examples = 0
        components = Counter()
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
                loss_output = object_event_v4_6_loss(
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
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                train_config.max_grad_norm,
            )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            global_step += 1
            size = int(events.shape[0])
            examples += size
            epoch_loss += float(loss_output.total.detach().cpu()) * size
            for name, value in loss_output.components.items():
                components[name] += float(value.detach().cpu()) * size
        _, _, validation_metrics = _evaluate_v46(
            model,
            validation_split,
            train_config=train_config,
            loss_config=loss_config,
            device=device,
            final=False,
            seed=train_config.seed + epoch * 31,
        )
        objective = _selection_objective(validation_metrics)
        improved = objective < best_objective - 1.0e-6
        if improved:
            best_objective = objective
            best_epoch = epoch
            best_base_only = False
            epochs_without_improvement = 0
            torch.save(
                {
                    "artifact_type": "object_event_v4_6_best_observed",
                    "epoch": epoch,
                    "base_only": False,
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                    "selection_objective": objective,
                    "initial_checkpoint": initial_checkpoint.resolve().as_posix(),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        event = cast(Mapping[str, object], validation_metrics["event"])
        official = cast(Mapping[str, object], validation_metrics["official_eap"])
        geometry = cast(Mapping[str, object], validation_metrics["geometry"])
        row = cast(dict[str, Any], _json_safe({
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_loss / max(examples, 1),
            "train_components": {name: value / max(examples, 1) for name, value in components.items()},
            "validation_selection_objective": objective,
            "validation": validation_metrics,
            "best_epoch": best_epoch,
            "best_objective": best_objective,
            "best_base_only": best_base_only,
        }))
        history.append(row)
        (output_dir / "history.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in history),
            encoding="utf-8",
        )
        print(json.dumps({
            "epoch": epoch,
            "loss": row["train_loss"],
            "weighted_mid": official.get("weighted_mid"),
            "pearson": event.get("pearson"),
            "balanced_sign": event.get("balanced_sign_accuracy"),
            "height_pearson": geometry.get("height_log_eta_pearson"),
            "mask_iou": geometry.get("foreground_soft_iou"),
            "best_epoch": best_epoch,
            "best_base_only": best_base_only,
        }, ensure_ascii=False), flush=True)
        if epoch >= minimum_epochs and epochs_without_improvement >= patience_epochs:
            break

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    model.set_base_only(bool(best_payload.get("base_only", False)))
    train_rows, train_per_sequence, train_metrics = _evaluate_v46(
        model,
        train_split,
        train_config=train_config,
        loss_config=loss_config,
        device=device,
        final=True,
        seed=train_config.seed + 7001,
    )
    validation_rows, validation_per_sequence, validation_metrics = _evaluate_v46(
        model,
        validation_split,
        train_config=train_config,
        loss_config=loss_config,
        device=device,
        final=True,
        seed=train_config.seed + 9001,
    )
    train_rows.to_csv(output_dir / "train_predictions.csv", index=False)
    validation_rows.to_csv(output_dir / "validation_predictions.csv", index=False)
    train_per_sequence.to_csv(output_dir / "train_per_sequence.csv", index=False)
    validation_per_sequence.to_csv(output_dir / "validation_per_sequence.csv", index=False)
    if mode == "overfit":
        gates = _overfit_gates(train_metrics, train_config)
    else:
        gates = _screen_gates(validation_metrics, baseline_metrics, train_config)
    passed = all(gates.values()) and not bool(best_payload.get("base_only", False))
    summary = cast(dict[str, Any], _json_safe({
        "artifact_type": "object_event_v4_6_learned_foreground_height_ratio",
        "status": f"{mode}_passed" if passed else f"{mode}_failed",
        "mode": mode,
        "created_at": datetime.now(UTC).isoformat(),
        "started_at": started_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cache_manifest": cache_manifest.resolve().as_posix(),
        "cache_manifest_sha256": _sha256(cache_manifest),
        "config": config_path.resolve().as_posix(),
        "initial_checkpoint": initial_checkpoint.resolve().as_posix(),
        "initial_checkpoint_sha256": _sha256(initial_checkpoint),
        "initial_checkpoint_epoch": int(initial_payload.get("epoch", 0)),
        "base_config": asdict(base_config),
        "geometry_config": asdict(geometry_config),
        "train_config": asdict(train_config),
        "loss_config": asdict(loss_config),
        "train_split": train_manifest,
        "validation_split": validation_manifest,
        "completed_epochs": len(history),
        "best_epoch": int(best_payload.get("epoch", 0)),
        "best_base_only": bool(best_payload.get("base_only", False)),
        "baseline_validation_metrics": baseline_metrics,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "gates": gates,
        "passed": passed,
        "scientific_contract": {
            "event_only_inference": True,
            "receives_boxes_at_inference": False,
            "receives_visible_heights_at_inference": False,
            "boxes_are_training_only_foreground_targets": True,
            "visible_heights_are_training_only_ratio_targets": True,
            "frozen_v4_2_baseline_preserved": True,
            "separate_trainable_geometry_encoder": True,
            "validation_is_not_official_eap_test": True,
            "evttc_not_opened": True,
            "advance_to_multiseed": mode == "screen" and passed,
        },
    }))
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
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("overfit", "screen"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            cache_manifest=args.cache_manifest.resolve(),
            config_path=args.config.resolve(),
            initial_checkpoint=args.initial_checkpoint.resolve(),
            output_dir=args.output_dir.resolve(),
            device_name=args.device,
            mode=args.mode,
            force=args.force,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if bool(result["passed"]) else 2
    except Exception as exc:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_6_operational_failure",
            "status": "operational_failure",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
