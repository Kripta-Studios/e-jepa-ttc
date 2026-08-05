#!/usr/bin/env python3
"""Train the full sequence-held-out Object Event v4.2 event-only screen.

This is the only justified successor to the v4.1 overfit diagnostic.  It uses all
2048 train and 2048 validation rows from the corrected common-coordinate cache,
keeps boxes/motion out of the model, removes the failed global activity branch,
and selects checkpoints only on held-out validation sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.models.object_event_v4_1 import ObjectEventV41Config  # noqa: E402
from e_jepa_ttc.models.object_event_v4_2 import ObjectEventTTCV42  # noqa: E402
from e_jepa_ttc.training.object_event_v4_1 import target_expansion  # noqa: E402
from e_jepa_ttc.training.object_event_v4_2 import (  # noqa: E402
    ObjectEventV42LossConfig,
    object_event_v4_2_loss,
)


@dataclass(frozen=True)
class ObjectEventV42TrainConfig:
    batch_size: int = 32
    num_workers: int = 0
    maximum_epochs: int = 18
    minimum_epochs: int = 6
    patience_epochs: int = 5
    backbone_learning_rate: float = 1.0e-4
    head_learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 2.0
    warmup_epochs: int = 1
    seed: int = 7
    precision: str = "fp32"
    shuffle_repeats_during_training: int = 2
    shuffle_repeats_final: int = 8
    bootstrap_repeats_during_training: int = 250
    bootstrap_repeats_final: int = 2000
    validation_pearson_gate: float = 0.25
    validation_pearson_lower_ci_gate: float = 0.05
    validation_sequence_macro_pearson_gate: float = 0.15
    validation_all_sequences_positive: bool = True
    validation_balanced_sign_gate: float = 0.58
    validation_negative_accuracy_gate: float = 0.55
    validation_expansion_mae_gate: float = 0.025
    validation_saturation_gate: float = 0.10
    zero_event_pearson_drop_gate: float = 0.15
    shuffled_event_pearson_drop_gate: float = 0.10
    shuffled_event_mean_abs_change_gate: float = 0.01

    def __post_init__(self) -> None:
        positive_ints = (
            self.batch_size,
            self.maximum_epochs,
            self.minimum_epochs,
            self.patience_epochs,
            self.warmup_epochs,
            self.shuffle_repeats_during_training,
            self.shuffle_repeats_final,
            self.bootstrap_repeats_during_training,
            self.bootstrap_repeats_final,
            self.num_workers + 1,
        )
        if min(positive_ints) <= 0:
            raise ValueError("v4.2 integer controls must be positive")
        if self.minimum_epochs > self.maximum_epochs:
            raise ValueError("minimum_epochs exceeds maximum_epochs")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if min(self.backbone_learning_rate, self.head_learning_rate) <= 0.0:
            raise ValueError("learning rates must be positive")


@dataclass
class MaterializedSplit:
    events: torch.Tensor
    delta_t_s: torch.Tensor
    target_ttc_s: torch.Tensor
    sequence_ids: list[str]
    sample_tokens: list[str]
    track_ids: list[str]

    def __len__(self) -> int:
        return int(self.events.shape[0])


@dataclass
class EventBatch:
    events: torch.Tensor
    delta_t_s: torch.Tensor
    target_ttc_s: torch.Tensor


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def _load_config(
    path: Path,
) -> tuple[ObjectEventV41Config, ObjectEventV42TrainConfig, ObjectEventV42LossConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("v4.2 config must be a mapping")
    return (
        _construct(ObjectEventV41Config, cast(Mapping[str, Any], raw.get("model", {}))),
        _construct(
            ObjectEventV42TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))
        ),
        _construct(
            ObjectEventV42LossConfig, cast(Mapping[str, Any], raw.get("loss", {}))
        ),
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _json_safe(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _materialize(
    manifest: Path,
    split: str,
    *,
    input_size: int,
) -> tuple[MaterializedSplit, dict[str, Any]]:
    dataset = GarlTTCObjectEventV4Dataset(manifest.as_posix(), splits=(split,))
    count = len(dataset)
    events = torch.empty(
        (count, 3, 12, input_size, input_size), dtype=torch.float16
    )
    delta_t = torch.empty(count, dtype=torch.float32)
    target_ttc = torch.empty(count, dtype=torch.float32)
    sequence_ids: list[str] = []
    sample_tokens: list[str] = []
    track_ids: list[str] = []
    for index in range(count):
        record = dataset[index]
        item = torch.as_tensor(record["event_v4_common_roi"], dtype=torch.float32)
        if item.shape[:2] != (3, 12):
            raise ValueError(f"Expected [3,12,H,W], got {tuple(item.shape)}")
        if item.shape[-2:] != (input_size, input_size):
            item = torch.nn.functional.interpolate(
                item, size=(input_size, input_size), mode="area"
            )
        events[index].copy_(item.to(torch.float16))
        delta_t[index] = float(record["garl_delta_t_s"])
        target_ttc[index] = float(record["ttc_s"])
        sequence_ids.append(str(record["sequence_id"]))
        sample_tokens.append(str(record["sample_token"]))
        track_ids.append(str(record["track_id"]))
    signs = ["negative" if float(value) < 0.0 else "positive" for value in target_ttc]
    sequence_counts = Counter(sequence_ids)
    sign_counts = Counter(signs)
    return (
        MaterializedSplit(
            events=events,
            delta_t_s=delta_t,
            target_ttc_s=target_ttc,
            sequence_ids=sequence_ids,
            sample_tokens=sample_tokens,
            track_ids=track_ids,
        ),
        {
            "split": split,
            "count": count,
            "event_shape": list(events.shape[1:]),
            "storage_dtype": str(events.dtype),
            "sequence_counts": dict(sorted(sequence_counts.items())),
            "sign_counts": dict(sorted(sign_counts.items())),
            "contains_motion_or_boxes": False,
        },
    )


def _sampling_weights(split: MaterializedSplit) -> torch.Tensor:
    cells = [
        (
            sequence,
            "negative" if float(ttc) < 0.0 else "positive",
        )
        for sequence, ttc in zip(split.sequence_ids, split.target_ttc_s, strict=True)
    ]
    counts = Counter(cells)
    raw = np.asarray([1.0 / counts[cell] for cell in cells], dtype=np.float64)
    median = float(np.median(raw))
    if median > 0.0:
        raw = np.minimum(raw, median * 10.0)
    raw /= raw.sum()
    return torch.tensor(raw, dtype=torch.float64)


def _autocast(device: torch.device, precision: str):
    enabled = precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _metric_float(value: object, *, default: float = 0.0) -> float:
    """Return a finite metric value, using a fail-closed default otherwise."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError(
            f"Pearson shape mismatch: target={target.shape}, prediction={prediction.shape}"
        )
    finite = np.isfinite(target) & np.isfinite(prediction)
    target = target[finite]
    prediction = prediction[finite]
    if (
        target.size < 2
        or float(np.std(target)) <= 1.0e-12
        or float(np.std(prediction)) <= 1.0e-12
    ):
        return 0.0
    correlation = float(np.corrcoef(target, prediction)[0, 1])
    return correlation if math.isfinite(correlation) else 0.0


def _balanced_sign(
    target: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, float]:
    positive = target >= 0.0
    negative = target < 0.0
    pos_acc = float(np.mean(prediction[positive] >= 0.0)) if positive.any() else 0.0
    neg_acc = float(np.mean(prediction[negative] < 0.0)) if negative.any() else 0.0
    return pos_acc, neg_acc, 0.5 * (pos_acc + neg_acc)


def _branch_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    delta_t: np.ndarray,
    *,
    ttc_clip: float,
    min_expansion: float,
) -> dict[str, float]:
    sign = np.where(prediction < 0.0, -1.0, 1.0)
    denominator = sign * np.maximum(np.abs(prediction), min_expansion)
    ttc = np.clip(delta_t / denominator, -ttc_clip, ttc_clip)
    pos_acc, neg_acc, balanced = _balanced_sign(target, prediction)
    return {
        "pearson": _pearson(target, prediction),
        "expansion_mae": float(np.mean(np.abs(target - prediction))),
        "prediction_std": float(np.std(prediction)),
        "target_std": float(np.std(target)),
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_sign_accuracy": balanced,
        "ttc_saturation_rate": float(
            np.mean(np.abs(ttc) >= ttc_clip * (1.0 - 1.0e-6))
        ),
    }


def _bootstrap_pearson_ci(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = rng.integers(0, target.size, target.size)
        values[index] = _pearson(target[sample], prediction[sample])
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {"lower_95": float(lower), "median": float(median), "upper_95": float(upper)}


@torch.no_grad()
def _predict(
    model: ObjectEventTTCV42,
    events: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    reverse_predictions: list[np.ndarray] = []
    reversal_errors: list[np.ndarray] = []
    model.eval()
    for start in range(0, events.shape[0], batch_size):
        batch = events[start : start + batch_size].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        output = model(batch)
        predictions.append(output.expansion.float().cpu().numpy())
        reverse_predictions.append(output.reverse_expansion.float().cpu().numpy())
        reversal_errors.append(
            output.reversal_consistency_error.float().cpu().numpy()
        )
    return (
        np.concatenate(predictions),
        np.concatenate(reverse_predictions),
        np.concatenate(reversal_errors),
    )


def _evaluate(
    model: ObjectEventTTCV42,
    split: MaterializedSplit,
    *,
    batch_size: int,
    device: torch.device,
    max_abs_expansion: float,
    shuffle_repeats: int,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prediction, reverse_prediction, reversal_error = _predict(
        model, split.events, batch_size=batch_size, device=device
    )
    zero_prediction, _, _ = _predict(
        model, torch.zeros_like(split.events), batch_size=batch_size, device=device
    )
    target = (
        split.delta_t_s / split.target_ttc_s
    ).clamp(-max_abs_expansion * 0.999, max_abs_expansion * 0.999).numpy()
    delta_t = split.delta_t_s.numpy()

    rng = np.random.default_rng(seed)
    shuffled_predictions: list[np.ndarray] = []
    shuffled_pearsons: list[float] = []
    shuffled_changes: list[float] = []
    for _ in range(shuffle_repeats):
        permutation = rng.permutation(len(split))
        shuffled, _, _ = _predict(
            model,
            split.events[torch.from_numpy(permutation.copy())],
            batch_size=batch_size,
            device=device,
        )
        shuffled_predictions.append(shuffled)
        shuffled_pearsons.append(_pearson(target, shuffled))
        shuffled_changes.append(float(np.mean(np.abs(prediction - shuffled))))
    shuffled_mean = np.mean(np.stack(shuffled_predictions), axis=0)

    event_metrics = _branch_metrics(
        target,
        prediction,
        delta_t,
        ttc_clip=model.config.ttc_clip_seconds,
        min_expansion=model.config.min_abs_expansion_for_ttc,
    )
    zero_metrics = _branch_metrics(
        target,
        zero_prediction,
        delta_t,
        ttc_clip=model.config.ttc_clip_seconds,
        min_expansion=model.config.min_abs_expansion_for_ttc,
    )
    reverse_metrics = _branch_metrics(
        target,
        reverse_prediction,
        delta_t,
        ttc_clip=model.config.ttc_clip_seconds,
        min_expansion=model.config.min_abs_expansion_for_ttc,
    )

    rows = pd.DataFrame(
        {
            "sequence_id": split.sequence_ids,
            "sample_token": split.sample_tokens,
            "track_id": split.track_ids,
            "delta_t_s": delta_t,
            "target_ttc_s": split.target_ttc_s.numpy(),
            "target_expansion": target,
            "prediction_expansion": prediction,
            "reverse_expansion": reverse_prediction,
            "zero_events_expansion": zero_prediction,
            "shuffled_mean_expansion": shuffled_mean,
            "reversal_error": reversal_error,
        }
    )
    per_sequence_rows: list[dict[str, Any]] = []
    for sequence_id, frame in rows.groupby("sequence_id", sort=True):
        per_sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "count": len(frame),
                **_branch_metrics(
                    frame["target_expansion"].to_numpy(dtype=np.float64),
                    frame["prediction_expansion"].to_numpy(dtype=np.float64),
                    frame["delta_t_s"].to_numpy(dtype=np.float64),
                    ttc_clip=model.config.ttc_clip_seconds,
                    min_expansion=model.config.min_abs_expansion_for_ttc,
                ),
            }
        )
    per_sequence = pd.DataFrame.from_records(per_sequence_rows)
    sequence_macro_pearson = float(per_sequence["pearson"].mean())
    sequence_min_pearson = float(per_sequence["pearson"].min())
    shuffle_mean_pearson = float(np.mean(shuffled_pearsons))
    metrics = {
        "event": event_metrics,
        "zero_events": zero_metrics,
        "reverse": reverse_metrics,
        "bootstrap_pearson": _bootstrap_pearson_ci(
            target, prediction, repeats=bootstrap_repeats, seed=seed + 101
        ),
        "per_sequence": {
            "macro_pearson": sequence_macro_pearson,
            "minimum_pearson": sequence_min_pearson,
            "count": len(per_sequence),
        },
        "event_dependence": {
            "zero_event_pearson_drop": event_metrics["pearson"]
            - zero_metrics["pearson"],
            "zero_event_mean_abs_change": float(
                np.mean(np.abs(prediction - zero_prediction))
            ),
            "shuffled_event_pearson_mean": shuffle_mean_pearson,
            "shuffled_event_pearson_std": float(np.std(shuffled_pearsons)),
            "shuffled_event_pearsons": shuffled_pearsons,
            "shuffled_event_pearson_drop": event_metrics["pearson"]
            - shuffle_mean_pearson,
            "shuffled_event_mean_abs_change": float(np.mean(shuffled_changes)),
        },
        "reversal": {
            "mean_abs_consistency_error": float(np.mean(reversal_error)),
            "max_abs_consistency_error": float(np.max(reversal_error)),
            "reverse_target_pearson": _pearson(-target, reverse_prediction),
        },
    }
    return rows, per_sequence, cast(dict[str, Any], _json_safe(metrics))


def _gates(
    metrics: Mapping[str, Any], config: ObjectEventV42TrainConfig
) -> dict[str, bool]:
    event = cast(Mapping[str, object], metrics["event"])
    bootstrap = cast(Mapping[str, object], metrics["bootstrap_pearson"])
    sequence = cast(Mapping[str, object], metrics["per_sequence"])
    dependence = cast(Mapping[str, object], metrics["event_dependence"])
    return {
        "pearson": _metric_float(event.get("pearson"))
        >= config.validation_pearson_gate,
        "pearson_lower_ci": _metric_float(bootstrap.get("lower_95"))
        >= config.validation_pearson_lower_ci_gate,
        "sequence_macro_pearson": _metric_float(sequence.get("macro_pearson"))
        >= config.validation_sequence_macro_pearson_gate,
        "all_sequences_positive": (
            _metric_float(sequence.get("minimum_pearson")) > 0.0
            if config.validation_all_sequences_positive
            else True
        ),
        "balanced_sign": _metric_float(event.get("balanced_sign_accuracy"))
        >= config.validation_balanced_sign_gate,
        "negative_accuracy": _metric_float(event.get("negative_accuracy"))
        >= config.validation_negative_accuracy_gate,
        "expansion_mae": _metric_float(event.get("expansion_mae"), default=float("inf"))
        <= config.validation_expansion_mae_gate,
        "saturation": _metric_float(
            event.get("ttc_saturation_rate"), default=float("inf")
        )
        <= config.validation_saturation_gate,
        "zero_event_dependence": _metric_float(
            dependence.get("zero_event_pearson_drop")
        )
        >= config.zero_event_pearson_drop_gate,
        "shuffled_event_dependence": _metric_float(
            dependence.get("shuffled_event_pearson_drop")
        )
        >= config.shuffled_event_pearson_drop_gate,
        "shuffled_event_change": _metric_float(
            dependence.get("shuffled_event_mean_abs_change")
        )
        >= config.shuffled_event_mean_abs_change_gate,
    }


def _selection_score(metrics: Mapping[str, Any]) -> float:
    event = cast(Mapping[str, float], metrics["event"])
    sequence = cast(Mapping[str, float], metrics["per_sequence"])
    dependence = cast(Mapping[str, float], metrics["event_dependence"])
    return (
        _metric_float(event.get("pearson"))
        + _metric_float(sequence.get("macro_pearson"))
        + _metric_float(event.get("balanced_sign_accuracy"))
        + 0.25 * _metric_float(dependence.get("shuffled_event_pearson_drop"))
        - 10.0 * _metric_float(event.get("expansion_mae"), default=1.0)
        - _metric_float(event.get("ttc_saturation_rate"), default=1.0)
    )


def _gradient_audit(
    model: ObjectEventTTCV42,
    split: MaterializedSplit,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    events = split.events[:batch_size].to(device=device, dtype=torch.float32).requires_grad_(True)
    output = model(events)
    output.expansion.square().mean().backward()
    encoder_sq = 0.0
    head_sq = 0.0
    frozen_gradients = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                frozen_gradients += 1
            continue
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().square().sum().cpu())
        if name.startswith("encoder."):
            encoder_sq += value
        else:
            head_sq += value
    result = {
        "events_mean_abs_gradient": float(events.grad.detach().abs().mean().cpu()),
        "encoder_gradient_l2": math.sqrt(encoder_sq),
        "head_gradient_l2": math.sqrt(head_sq),
        "frozen_activity_parameters_with_gradient": frozen_gradients,
    }
    model.zero_grad(set_to_none=True)
    return result


def run(
    *,
    cache_manifest: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str,
    seed_override: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(UTC)
    model_config, train_config, loss_config = _load_config(config_path)
    if seed_override is not None:
        train_config = ObjectEventV42TrainConfig(
            **{**asdict(train_config), "seed": seed_override}
        )
    _seed(train_config.seed)
    device = _resolve_device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_split, train_manifest = _materialize(
        cache_manifest, "train", input_size=model_config.input_size
    )
    validation_split, validation_manifest = _materialize(
        cache_manifest, "validation", input_size=model_config.input_size
    )
    weights = _sampling_weights(train_split)

    model = ObjectEventTTCV42(model_config).to(device)
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.") and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": train_config.backbone_learning_rate},
            {"params": head_parameters, "lr": train_config.head_learning_rate},
        ],
        weight_decay=train_config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_split) / train_config.batch_size)
    total_steps = steps_per_epoch * train_config.maximum_epochs
    warmup_steps = steps_per_epoch * train_config.warmup_epochs

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / max(warmup_steps, 1), 1.0e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and train_config.precision == "fp16"
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "best_observed.pt"
    last_path = output_dir / "last.pt"
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(1, train_config.maximum_epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(train_config.seed * 1000 + epoch)
        indices = torch.multinomial(
            weights,
            num_samples=len(train_split),
            replacement=True,
            generator=generator,
        )
        component_sums: Counter[str] = Counter()
        epoch_loss = 0.0
        epoch_examples = 0
        gradient_norm_sum = 0.0
        for start in range(0, len(indices), train_config.batch_size):
            batch_indices = indices[start : start + train_config.batch_size]
            events = train_split.events[batch_indices].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            # Horizontal reflection preserves radial expansion while reducing
            # sequence-specific spatial memorisation.
            flip_mask = torch.rand(events.shape[0], device=device) < 0.5
            if bool(flip_mask.any()):
                events[flip_mask] = torch.flip(events[flip_mask], dims=(-1,))
            batch = EventBatch(
                events=events,
                delta_t_s=train_split.delta_t_s[batch_indices].to(
                    device=device, dtype=torch.float32, non_blocking=True
                ),
                target_ttc_s=train_split.target_ttc_s[batch_indices].to(
                    device=device, dtype=torch.float32, non_blocking=True
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, train_config.precision):
                output = model(batch.events)
                loss_output = object_event_v4_2_loss(
                    output,
                    batch.delta_t_s,
                    batch.target_ttc_s,
                    epoch=epoch,
                    config=loss_config,
                )
            if scaler is not None:
                scaler.scale(loss_output.total).backward()
                scaler.unscale_(optimizer)
            else:
                loss_output.total.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    train_config.max_grad_norm,
                )
            )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            global_step += 1
            size = int(events.shape[0])
            epoch_examples += size
            epoch_loss += float(loss_output.total.detach().cpu()) * size
            gradient_norm_sum += gradient_norm
            for name, value in loss_output.components.items():
                component_sums[name] += float(value.detach().cpu()) * size

        _, _, validation_metrics = _evaluate(
            model,
            validation_split,
            batch_size=train_config.batch_size,
            device=device,
            max_abs_expansion=loss_config.max_abs_expansion,
            shuffle_repeats=train_config.shuffle_repeats_during_training,
            bootstrap_repeats=train_config.bootstrap_repeats_during_training,
            seed=train_config.seed + epoch * 17,
        )
        gates = _gates(validation_metrics, train_config)
        score = _selection_score(validation_metrics)
        improved = score > best_score + 1.0e-6
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "artifact_type": "object_event_v4_2_best_observed",
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_config": asdict(model_config),
                    "train_config": asdict(train_config),
                    "loss_config": asdict(loss_config),
                    "model_state_dict": model.state_dict(),
                    "validation_metrics": validation_metrics,
                    "selection_score": score,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "artifact_type": "object_event_v4_2_last",
                "epoch": epoch,
                "global_step": global_step,
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "loss_config": asdict(loss_config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "validation_metrics": validation_metrics,
            },
            last_path,
        )
        event = cast(Mapping[str, float], validation_metrics["event"])
        dependence = cast(Mapping[str, float], validation_metrics["event_dependence"])
        row = cast(
            dict[str, Any],
            _json_safe(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": epoch_loss / max(epoch_examples, 1),
                    "train_components": {
                        name: value / max(epoch_examples, 1)
                        for name, value in component_sums.items()
                    },
                    "gradient_norm_mean": gradient_norm_sum / max(steps_per_epoch, 1),
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                    "validation_selection_score": score,
                    "validation": validation_metrics,
                    "gates": gates,
                    "checkpoint_eligible": all(gates.values()),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            ),
        )
        history.append(row)
        history_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in history),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "loss": row["train_loss"],
                    "validation_pearson": event["pearson"],
                    "validation_balanced_sign": event["balanced_sign_accuracy"],
                    "validation_mae": event["expansion_mae"],
                    "validation_saturation": event["ttc_saturation_rate"],
                    "shuffle_drop": dependence["shuffled_event_pearson_drop"],
                    "gates": gates,
                    "best_epoch": best_epoch,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if (
            epoch >= train_config.minimum_epochs
            and epochs_without_improvement >= train_config.patience_epochs
        ):
            break

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    train_predictions, train_per_sequence, train_metrics = _evaluate(
        model,
        train_split,
        batch_size=train_config.batch_size,
        device=device,
        max_abs_expansion=loss_config.max_abs_expansion,
        shuffle_repeats=train_config.shuffle_repeats_final,
        bootstrap_repeats=train_config.bootstrap_repeats_final,
        seed=train_config.seed + 7001,
    )
    validation_predictions, validation_per_sequence, validation_metrics = _evaluate(
        model,
        validation_split,
        batch_size=train_config.batch_size,
        device=device,
        max_abs_expansion=loss_config.max_abs_expansion,
        shuffle_repeats=train_config.shuffle_repeats_final,
        bootstrap_repeats=train_config.bootstrap_repeats_final,
        seed=train_config.seed + 9001,
    )
    train_predictions.to_csv(output_dir / "train_predictions.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    train_per_sequence.to_csv(output_dir / "train_per_sequence.csv", index=False)
    validation_per_sequence.to_csv(
        output_dir / "validation_per_sequence.csv", index=False
    )
    final_gates = _gates(validation_metrics, train_config)
    screen_passed = all(final_gates.values())
    gradient_audit = _gradient_audit(
        model,
        train_split,
        device=device,
        batch_size=train_config.batch_size,
    )
    if screen_passed:
        torch.save(best_payload, output_dir / "eligible.pt")
    ended_at = datetime.now(UTC)
    summary = cast(
        dict[str, Any],
        _json_safe(
            {
                "artifact_type": "object_event_v4_2_full_event_only_screen",
                "status": "screen_passed" if screen_passed else "screen_failed",
                "created_at": ended_at.isoformat(),
                "started_at": started_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "git_commit": _git_commit(),
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "cache_manifest": cache_manifest.resolve().as_posix(),
                "cache_manifest_sha256": _sha256(cache_manifest),
                "config": config_path.resolve().as_posix(),
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "loss_config": asdict(loss_config),
                "train_split": train_manifest,
                "validation_split": validation_manifest,
                "completed_epochs": len(history),
                "best_epoch": best_epoch,
                "best_selection_score": best_score,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "selection_gates": final_gates,
                "screen_passed": screen_passed,
                "gradient_audit": gradient_audit,
                "scientific_contract": {
                    "event_only": True,
                    "receives_observable_motion": False,
                    "receives_boxes": False,
                    "uses_ttc_domain_loss": False,
                    "uses_activity_shortcut": False,
                    "uses_level_transfer": False,
                    "uses_motion_fusion": False,
                    "checkpoint_selected_on_held_out_sequences": True,
                    "uses_repeated_global_event_shuffles": True,
                    "advance_to_seed_repeats": screen_passed,
                },
            }
        ),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiment/e_jepa_garl_object_event_screen_v4_2.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    try:
        if output_dir.exists():
            if not args.force:
                raise FileExistsError(f"Output exists: {output_dir}; pass --force")
            shutil.rmtree(output_dir)
        result = run(
            cache_manifest=args.cache_manifest.resolve(),
            config_path=args.config.resolve(),
            output_dir=output_dir,
            device_name=args.device,
            seed_override=args.seed,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if bool(result["screen_passed"]) else 2
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_2_operational_failure",
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
