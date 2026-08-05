#!/usr/bin/env python3
"""Run the bounded Object Event v4.1 event-only overfit diagnostic.

The diagnostic reuses the corrected v4 common-coordinate cache. It intentionally
excludes observable motion, fusion and TTC-domain losses. Success means that a
small event-only model can memorize a balanced 64-sample train subset and retain
non-trivial correlation on held-out sequences. Failure blocks further full runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.object_event_v4 import GarlTTCObjectEventV4Dataset  # noqa: E402
from e_jepa_ttc.models.object_event_v4_1 import (  # noqa: E402
    ObjectEventTTCV41,
    ObjectEventV41Config,
)
from e_jepa_ttc.training.object_event_v4_1 import (  # noqa: E402
    ObjectEventV41LossConfig,
    object_event_v4_1_loss,
    target_expansion,
)


@dataclass(frozen=True)
class ObjectEventV41TrainConfig:
    train_samples: int = 64
    validation_samples: int = 256
    batch_size: int = 8
    num_workers: int = 0
    max_steps: int = 320
    minimum_steps: int = 160
    evaluation_interval: int = 20
    learning_rate: float = 1.0e-3
    backbone_learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 2.0
    seed: int = 7
    precision: str = "fp32"
    train_pearson_gate: float = 0.95
    train_balanced_sign_gate: float = 0.95
    train_saturation_gate: float = 0.05
    train_expansion_mae_gate: float = 0.005
    validation_pearson_gate: float = 0.20
    validation_balanced_sign_gate: float = 0.55
    event_dependence_gate: float = 0.05
    required_gate_streak: int = 2

    def __post_init__(self) -> None:
        integers = (
            self.train_samples,
            self.validation_samples,
            self.batch_size,
            self.max_steps,
            self.minimum_steps,
            self.evaluation_interval,
            self.required_gate_streak,
            self.num_workers + 1,
        )
        if min(integers) <= 0:
            raise ValueError("v4.1 integer controls must be positive")
        if self.minimum_steps > self.max_steps:
            raise ValueError("minimum_steps exceeds max_steps")
        if self.train_samples % 2:
            raise ValueError("train_samples must be even for sign balancing")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return cls(**dict(values))


def load_config(path: Path) -> tuple[ObjectEventV41Config, ObjectEventV41TrainConfig, ObjectEventV41LossConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Diagnostic config must be a mapping")
    return (
        _construct(ObjectEventV41Config, cast(Mapping[str, Any], raw.get("model", {}))),
        _construct(ObjectEventV41TrainConfig, cast(Mapping[str, Any], raw.get("train", {}))),
        _construct(ObjectEventV41LossConfig, cast(Mapping[str, Any], raw.get("loss", {}))),
    )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EventOnlyBatch:
    events: torch.Tensor
    delta_t_s: torch.Tensor
    target_ttc_s: torch.Tensor
    sequence_ids: list[str]
    sample_tokens: list[str]
    track_ids: list[str]

    def to(self, device: torch.device) -> "EventOnlyBatch":
        return EventOnlyBatch(
            events=self.events.to(device=device, dtype=torch.float32, non_blocking=True),
            delta_t_s=self.delta_t_s.to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            target_ttc_s=self.target_ttc_s.to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            sequence_ids=self.sequence_ids,
            sample_tokens=self.sample_tokens,
            track_ids=self.track_ids,
        )


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _lightweight_record(record: Mapping[str, Any], input_size: int) -> dict[str, Any]:
    events = torch.as_tensor(record["event_v4_common_roi"], dtype=torch.float32)
    if events.shape[:2] != (3, 12):
        raise ValueError(f"Expected event tensor [3,12,H,W], got {tuple(events.shape)}")
    if events.shape[-2:] != (input_size, input_size):
        events = torch.nn.functional.interpolate(
            events, size=(input_size, input_size), mode="area"
        )
    return {
        "events": events.contiguous(),
        "delta_t_s": float(record["garl_delta_t_s"]),
        "target_ttc_s": float(record["ttc_s"]),
        "sequence_id": str(record["sequence_id"]),
        "sample_token": str(record["sample_token"]),
        "track_id": str(record["track_id"]),
    }


def _balanced_records(
    dataset: GarlTTCObjectEventV4Dataset,
    count: int,
    *,
    seed: int,
    input_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reservoir-sample a balanced subset while reading every shard once.

    The v4 cache shards are large.  Random indices would repeatedly reload
    600-MiB shards through the one-shard cache.  Streaming reservoirs preserve
    a declared random sample and immediately retain only the downsampled event
    tensor plus labels/IDs; motion, boxes and heights are discarded.
    """

    half = count // 2
    rng = random.Random(seed)
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    seen_positive = 0
    seen_negative = 0
    for index in range(len(dataset)):
        record = dataset[index]
        is_negative = float(record["ttc_s"]) < 0.0
        reservoir = negative if is_negative else positive
        seen = seen_negative if is_negative else seen_positive
        seen += 1
        if is_negative:
            seen_negative = seen
        else:
            seen_positive = seen
        if len(reservoir) < half:
            reservoir.append(_lightweight_record(record, input_size))
            continue
        replacement_index = rng.randrange(seen)
        if replacement_index < half:
            reservoir[replacement_index] = _lightweight_record(record, input_size)

    if len(positive) < half or len(negative) < half:
        raise RuntimeError(
            "Cannot form balanced event-only subset: "
            f"positive={seen_positive}, negative={seen_negative}, requested={count}"
        )
    records: list[dict[str, Any]] = []
    for pos, neg in zip(positive, negative, strict=True):
        records.extend((pos, neg))
    sequences = sorted({str(record["sequence_id"]) for record in records})
    return records, {
        "count": len(records),
        "positive": half,
        "negative": half,
        "available_positive": seen_positive,
        "available_negative": seen_negative,
        "sequence_counts": {
            sequence: sum(record["sequence_id"] == sequence for record in records)
            for sequence in sequences
        },
        "materialized_event_shape": list(records[0]["events"].shape),
        "contains_motion_or_boxes": False,
    }


def _collate_event_only(records: list[dict[str, Any]]) -> EventOnlyBatch:
    if not records:
        raise ValueError("Empty v4.1 event-only batch")
    return EventOnlyBatch(
        events=torch.stack([record["events"] for record in records]),
        delta_t_s=torch.tensor(
            [record["delta_t_s"] for record in records], dtype=torch.float32
        ),
        target_ttc_s=torch.tensor(
            [record["target_ttc_s"] for record in records], dtype=torch.float32
        ),
        sequence_ids=[str(record["sequence_id"]) for record in records],
        sample_tokens=[str(record["sample_token"]) for record in records],
        track_ids=[str(record["track_id"]) for record in records],
    )


def _loader(
    records: list[dict[str, Any]], config: ObjectEventV41TrainConfig
) -> DataLoader[EventOnlyBatch]:
    return DataLoader(
        records,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        collate_fn=_collate_event_only,
    )


def _pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.size < 2 or float(np.std(target)) <= 1.0e-12 or float(np.std(prediction)) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(target, prediction)[0, 1])


def _balanced_sign(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    negative = target < 0.0
    positive = ~negative
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
        "ttc_saturation_rate": float(np.mean(np.abs(ttc) >= ttc_clip * (1.0 - 1.0e-6))),
    }


def _autocast(device: torch.device, precision: str):
    enabled = precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.no_grad()
def evaluate(
    model: ObjectEventTTCV41,
    loader: DataLoader[EventOnlyBatch],
    *,
    device: torch.device,
    loss_config: ObjectEventV41LossConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = raw_batch.to(device)
        output = model(batch.events)
        zero = model(torch.zeros_like(batch.events))
        shuffled_events = (
            torch.roll(batch.events, shifts=1, dims=0)
            if batch.events.shape[0] > 1
            else torch.flip(batch.events, dims=(1,))
        )
        shuffled = model(shuffled_events)
        target = target_expansion(
            batch.delta_t_s, batch.target_ttc_s, loss_config.max_abs_expansion
        )
        for index in range(batch.events.shape[0]):
            rows.append(
                {
                    "sequence_id": batch.sequence_ids[index],
                    "sample_token": batch.sample_tokens[index],
                    "track_id": batch.track_ids[index],
                    "delta_t_s": float(batch.delta_t_s[index].cpu()),
                    "target_ttc_s": float(batch.target_ttc_s[index].cpu()),
                    "target_expansion": float(target[index].cpu()),
                    "prediction_expansion": float(output.expansion[index].cpu()),
                    "encoded_expansion": float(output.encoded_expansion[index].cpu()),
                    "activity_expansion": float(output.activity_expansion[index].cpu()),
                    "reverse_expansion": float(output.reverse_expansion[index].cpu()),
                    "zero_events_expansion": float(zero.expansion[index].cpu()),
                    "shuffled_events_expansion": float(shuffled.expansion[index].cpu()),
                    "reversal_error": float(output.reversal_consistency_error[index].cpu()),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    target = frame["target_expansion"].to_numpy(dtype=np.float64)
    delta_t = frame["delta_t_s"].to_numpy(dtype=np.float64)
    branches = {}
    for name, column in {
        "event": "prediction_expansion",
        "encoded": "encoded_expansion",
        "activity": "activity_expansion",
        "reverse": "reverse_expansion",
        "zero_events": "zero_events_expansion",
        "shuffled_events": "shuffled_events_expansion",
    }.items():
        branches[name] = _branch_metrics(
            target,
            frame[column].to_numpy(dtype=np.float64),
            delta_t,
            ttc_clip=model.config.ttc_clip_seconds,
            min_expansion=model.config.min_abs_expansion_for_ttc,
        )
    event_prediction = frame["prediction_expansion"].to_numpy(dtype=np.float64)
    shuffled_prediction = frame["shuffled_events_expansion"].to_numpy(dtype=np.float64)
    zero_prediction = frame["zero_events_expansion"].to_numpy(dtype=np.float64)
    metrics = {
        "branches": branches,
        "event_dependence": {
            "pearson_drop_shuffled_events": branches["event"]["pearson"]
            - branches["shuffled_events"]["pearson"],
            "pearson_drop_zero_events": branches["event"]["pearson"]
            - branches["zero_events"]["pearson"],
            "mean_abs_change_shuffled_events": float(
                np.mean(np.abs(event_prediction - shuffled_prediction))
            ),
            "mean_abs_change_zero_events": float(
                np.mean(np.abs(event_prediction - zero_prediction))
            ),
        },
        "reversal": {
            "mean_abs_consistency_error": float(frame["reversal_error"].mean()),
            "max_abs_consistency_error": float(frame["reversal_error"].max()),
            "reverse_target_pearson": _pearson(
                -target, frame["reverse_expansion"].to_numpy(dtype=np.float64)
            ),
        },
    }
    return frame, cast(dict[str, Any], _json_safe(metrics))


def _train_gates(metrics: Mapping[str, Any], config: ObjectEventV41TrainConfig) -> dict[str, bool]:
    event = cast(Mapping[str, float], cast(Mapping[str, Any], metrics["branches"])["event"])
    return {
        "pearson": float(event["pearson"]) >= config.train_pearson_gate,
        "balanced_sign": float(event["balanced_sign_accuracy"])
        >= config.train_balanced_sign_gate,
        "saturation": float(event["ttc_saturation_rate"])
        <= config.train_saturation_gate,
        "expansion_mae": float(event["expansion_mae"])
        <= config.train_expansion_mae_gate,
    }


def _validation_gates(metrics: Mapping[str, Any], config: ObjectEventV41TrainConfig) -> dict[str, bool]:
    branches = cast(Mapping[str, Any], metrics["branches"])
    event = cast(Mapping[str, float], branches["event"])
    dependence = cast(Mapping[str, float], metrics["event_dependence"])
    return {
        "pearson": float(event["pearson"]) >= config.validation_pearson_gate,
        "balanced_sign": float(event["balanced_sign_accuracy"])
        >= config.validation_balanced_sign_gate,
        "shuffled_event_dependence": float(dependence["pearson_drop_shuffled_events"])
        >= config.event_dependence_gate,
    }


def _gradient_audit(
    model: ObjectEventTTCV41,
    batch: EventOnlyBatch,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    model.zero_grad(set_to_none=True)
    events = batch.events.to(device=device, dtype=torch.float32).requires_grad_(True)
    output = model(events)
    output.expansion.square().mean().backward()
    event_gradient = float(events.grad.detach().abs().mean().cpu())
    encoder_sq = 0.0
    head_sq = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().square().sum().cpu())
        if name.startswith("encoder."):
            encoder_sq += value
        else:
            head_sq += value
    model.zero_grad(set_to_none=True)
    return {
        "events_mean_abs_gradient": event_gradient,
        "encoder_gradient_l2": math.sqrt(encoder_sq),
        "head_gradient_l2": math.sqrt(head_sq),
    }


def run(
    *,
    cache_manifest: Path,
    config_path: Path,
    output_dir: Path,
    device_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = datetime.now(UTC)
    model_config, train_config, loss_config = load_config(config_path)
    _seed(train_config.seed)
    device = _resolve_device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_dataset = GarlTTCObjectEventV4Dataset(
        cache_manifest.as_posix(), splits=("train",)
    )
    validation_dataset = GarlTTCObjectEventV4Dataset(
        cache_manifest.as_posix(), splits=("validation",)
    )
    train_records, train_subset = _balanced_records(
        train_dataset,
        train_config.train_samples,
        seed=train_config.seed,
        input_size=model_config.input_size,
    )
    validation_records, validation_subset = _balanced_records(
        validation_dataset,
        train_config.validation_samples,
        seed=train_config.seed + 1,
        input_size=model_config.input_size,
    )
    train_loader = _loader(train_records, train_config)
    validation_loader = _loader(validation_records, train_config)

    model = ObjectEventTTCV41(model_config).to(device)
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
            {"params": head_parameters, "lr": train_config.learning_rate},
        ],
        weight_decay=train_config.weight_decay,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and train_config.precision == "fp16"
        else None
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    history_path = output_dir / "history.jsonl"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_step = 0
    gate_streak = 0
    step = 0
    running = iter(train_loader)

    while step < train_config.max_steps:
        try:
            raw_batch = next(running)
        except StopIteration:
            running = iter(train_loader)
            raw_batch = next(running)
        step += 1
        batch = raw_batch.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, train_config.precision):
            output = model(batch.events)
            loss_output = object_event_v4_1_loss(
                output,
                batch.delta_t_s,
                batch.target_ttc_s,
                step=step,
                config=loss_config,
            )
        if scaler is not None:
            scaler.scale(loss_output.total).backward()
            scaler.unscale_(optimizer)
        else:
            loss_output.total.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        )
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        should_evaluate = step == 1 or step % train_config.evaluation_interval == 0
        if not should_evaluate:
            continue
        train_frame, train_metrics = evaluate(
            model, train_loader, device=device, loss_config=loss_config
        )
        gates = _train_gates(train_metrics, train_config)
        all_gates = all(gates.values())
        gate_streak = gate_streak + 1 if all_gates else 0
        event = cast(Mapping[str, float], cast(Mapping[str, Any], train_metrics["branches"])["event"])
        score = (
            float(event["pearson"])
            + float(event["balanced_sign_accuracy"])
            - 10.0 * float(event["expansion_mae"])
            - float(event["ttc_saturation_rate"])
        )
        improved = score > best_score
        if improved:
            best_score = score
            best_step = step
            torch.save(
                {
                    "artifact_type": "object_event_v4_1_overfit_checkpoint",
                    "step": step,
                    "model_config": asdict(model_config),
                    "train_config": asdict(train_config),
                    "loss_config": asdict(loss_config),
                    "model_state_dict": model.state_dict(),
                    "train_metrics": train_metrics,
                },
                best_path,
            )
            train_frame.to_csv(output_dir / "best_train_predictions.csv", index=False)
        torch.save(
            {
                "artifact_type": "object_event_v4_1_overfit_checkpoint",
                "step": step,
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "loss_config": asdict(loss_config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
            },
            last_path,
        )
        row = cast(
            dict[str, Any],
            _json_safe(
                {
                    "step": step,
                    "loss": float(loss_output.total.detach().cpu()),
                    "loss_components": {
                        key: float(value.detach().cpu())
                        for key, value in loss_output.components.items()
                    },
                    "gradient_norm": gradient_norm,
                    "train_metrics": train_metrics,
                    "train_gates": gates,
                    "gate_streak": gate_streak,
                    "best_step": best_step,
                    "best_score": best_score,
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
                    "step": step,
                    "loss": row["loss"],
                    "pearson": event["pearson"],
                    "balanced_sign": event["balanced_sign_accuracy"],
                    "mae": event["expansion_mae"],
                    "saturation": event["ttc_saturation_rate"],
                    "gates": gates,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if step >= train_config.minimum_steps and gate_streak >= train_config.required_gate_streak:
            break

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    train_frame, train_metrics = evaluate(
        model, train_loader, device=device, loss_config=loss_config
    )
    validation_frame, validation_metrics = evaluate(
        model, validation_loader, device=device, loss_config=loss_config
    )
    train_frame.to_csv(output_dir / "train_predictions.csv", index=False)
    validation_frame.to_csv(output_dir / "validation_predictions.csv", index=False)
    train_gates = _train_gates(train_metrics, train_config)
    validation_gates = _validation_gates(validation_metrics, train_config)
    first_batch = next(iter(train_loader))
    gradient_audit = _gradient_audit(model, first_batch, device=device)
    overfit_passed = all(train_gates.values())
    screen_passed = overfit_passed and all(validation_gates.values())
    status = "screen_passed" if screen_passed else "overfit_passed" if overfit_passed else "overfit_failed"
    ended_at = datetime.now(UTC)
    summary = cast(
        dict[str, Any],
        _json_safe(
            {
                "artifact_type": "object_event_v4_1_event_only_diagnostic",
                "status": status,
                "created_at": ended_at.isoformat(),
                "started_at": started_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "git_commit": _git_commit(),
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "cache_manifest": cache_manifest.resolve().as_posix(),
                "cache_manifest_sha256": _sha256(cache_manifest),
                "config": config_path.resolve().as_posix(),
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "loss_config": asdict(loss_config),
                "train_subset": train_subset,
                "validation_subset": validation_subset,
                "best_step": best_step,
                "best_score": best_score,
                "completed_steps": step,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "train_gates": train_gates,
                "validation_gates": validation_gates,
                "overfit_passed": overfit_passed,
                "screen_passed": screen_passed,
                "gradient_audit": gradient_audit,
                "scientific_contract": {
                    "event_only": True,
                    "receives_observable_motion": False,
                    "receives_boxes": False,
                    "uses_ttc_domain_loss": False,
                    "hard_codes_reversal_antisymmetry": False,
                    "uses_reversal_as_late_auxiliary": True,
                    "uses_common_coordinate_t0_t1_t2": True,
                    "advance_to_full_training": screen_passed,
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
        default=ROOT / "configs/experiment/e_jepa_garl_object_event_overfit_v4_1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        output_dir = args.output_dir.resolve()
        if output_dir.exists():
            if not args.force:
                raise FileExistsError(f"Output exists: {output_dir}; pass --force to replace it")
            import shutil

            shutil.rmtree(output_dir)
        result = run(
            cache_manifest=args.cache_manifest.resolve(),
            config_path=args.config.resolve(),
            output_dir=output_dir,
            device_name=args.device,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "object_event_v4_1_diagnostic_failure",
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
