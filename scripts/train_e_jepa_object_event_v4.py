#!/usr/bin/env python
"""Train Object Event TTC v4 with event-required validation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.data.object_event_v4 import (  # noqa: E402
    GarlTTCObjectEventV4Dataset,
    ObjectEventV4Batch,
    collate_object_event_v4,
)
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)
from e_jepa_ttc.models.object_event_v4 import (  # noqa: E402
    ObjectEventTTCV4,
    ObjectEventV4Config,
    safe_ttc_from_expansion,
)
from e_jepa_ttc.reproducibility import resolve_device  # noqa: E402
from e_jepa_ttc.training.object_event_v4 import (  # noqa: E402
    ObjectEventV4LossConfig,
    ObjectEventV4ModalityConfig,
    apply_modality_dropout,
    object_event_v4_loss,
)
from e_jepa_ttc.training.tubelet_finetuning import prediction_health  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


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


def _hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_safe(dict(value)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _roc_auc(target_negative: np.ndarray, score_negative: np.ndarray) -> float | None:
    target = np.asarray(target_negative, dtype=bool)
    score = np.asarray(score_negative, dtype=np.float64)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=np.float64)
    return float(
        (ranks[target].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _direction(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    target_negative = target < 0.0
    predicted_negative = prediction < 0.0
    positive_mask = ~target_negative
    negative_mask = target_negative
    positive = float(np.mean(~predicted_negative[positive_mask])) if positive_mask.any() else None
    negative = float(np.mean(predicted_negative[negative_mask])) if negative_mask.any() else None
    valid = [value for value in (positive, negative) if value is not None]
    return {
        "positive_accuracy": positive,
        "negative_accuracy": negative,
        "balanced_accuracy": float(np.mean(valid)) if valid else None,
        "negative_recall": negative,
        "auc": _roc_auc(target_negative, -prediction),
    }


def _ttc_metrics(
    target_ttc: np.ndarray,
    prediction_ttc: np.ndarray,
    sequences: np.ndarray,
) -> dict[str, Any]:
    result = signed_garl_metrics(target_ttc, prediction_ttc)
    error = np.abs(target_ttc - prediction_ttc)
    result["mae_s"] = float(error.mean())
    result["median_ae_s"] = float(np.median(error))
    result["sequence_macro"] = sequence_macro_signed_metrics(
        target_ttc, prediction_ttc, sequences
    )
    return result


@dataclass(frozen=True)
class ObjectEventV4TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_workers: int = 2
    scratch_backbone_learning_rate: float = 1.0e-4
    pretrained_backbone_learning_rate: float = 1.0e-5
    pooling_learning_rate: float = 1.0e-4
    event_head_learning_rate: float = 3.0e-4
    predictor_learning_rate: float = 3.0e-4
    motion_learning_rate: float = 1.0e-4
    fusion_learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    readout_weight_decay: float = 0.0
    pretrained_backbone_warmup_epochs: int = 3
    target_ema_momentum: float = 0.996
    lr_milestones: tuple[int, ...] = (15, 22, 27)
    lr_gamma: float = 0.2
    precision: str = "fp32"
    max_grad_norm: float = 1.0
    selection_start_epoch: int = 5
    minimum_epochs: int = 12
    early_stopping_patience: int = 7
    min_full_expansion_pearson: float = 0.30
    min_event_only_pearson: float = 0.30
    min_event_only_balanced_sign_accuracy: float = 0.60
    min_event_only_negative_recall: float = 0.30
    min_event_dependence_zero_events: float = 0.05
    min_event_dependence_shuffled_events: float = 0.03
    max_reversal_error: float = 1.0e-5
    max_ttc_saturation_rate: float = 0.01
    min_event_gate_mean: float = 0.40
    negative_mid_penalty: float = 0.25
    mae_penalty: float = 5.0
    event_mid_penalty: float = 0.10
    seed: int = 7
    run_scope: str = "bounded_screen"
    require_clean_git: bool = False

    def __post_init__(self) -> None:
        integers = (
            self.epochs,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.num_workers + 1,
            self.selection_start_epoch,
            self.minimum_epochs,
            self.early_stopping_patience + 1,
        )
        if min(integers) <= 0:
            raise ValueError("V4 integer controls must be positive")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs exceeds epochs")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if self.run_scope not in {"bounded_screen", "full_candidate"}:
            raise ValueError("run_scope must be bounded_screen or full_candidate")
        gates = (
            self.min_full_expansion_pearson,
            self.min_event_only_pearson,
            self.min_event_only_balanced_sign_accuracy,
            self.min_event_only_negative_recall,
            self.min_event_dependence_zero_events,
            self.min_event_dependence_shuffled_events,
            self.max_ttc_saturation_rate,
            self.min_event_gate_mean,
        )
        if min(gates) < 0.0 or max(gates) > 1.0:
            raise ValueError("Probability/correlation gates must lie in [0,1]")


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return cast(dict[str, Any], value)


def _resolve(owner: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Expected path reference in {owner}")
    for candidate in (owner.parent / value, ROOT / value):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve {value!r} from {owner}")


def _construct(cls: type[Any], values: Mapping[str, Any]) -> Any:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {unknown}")
    normalized = dict(values)
    if cls is ObjectEventV4TrainConfig and "lr_milestones" in normalized:
        normalized["lr_milestones"] = tuple(normalized["lr_milestones"])
    return cls(**normalized)


def load_spec(
    experiment_path: Path,
) -> tuple[
    ObjectEventV4Config,
    ObjectEventV4TrainConfig,
    ObjectEventV4LossConfig,
    ObjectEventV4ModalityConfig,
    dict[str, Any],
]:
    experiment = _read_yaml(experiment_path)
    model_path = _resolve(experiment_path, experiment["model"])
    train_path = _resolve(experiment_path, experiment["training"])
    model_values = _read_yaml(model_path)
    train_values = _read_yaml(train_path)
    loss_values = train_values.pop("loss")
    modality_values = train_values.pop("modality")
    return (
        _construct(ObjectEventV4Config, model_values),
        _construct(ObjectEventV4TrainConfig, train_values),
        _construct(ObjectEventV4LossConfig, loss_values),
        _construct(ObjectEventV4ModalityConfig, modality_values),
        {
            "experiment": experiment_path.as_posix(),
            "model": model_path.as_posix(),
            "training": train_path.as_posix(),
            "experiment_metadata": {
                key: value
                for key, value in experiment.items()
                if key not in {"model", "training"}
            },
        },
    )


def _autocast(device: torch.device, precision: str) -> torch.autocast:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def _load_pretrained(model: ObjectEventTTCV4, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"used": False, "transferred_keys": []}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != "dense_level_dynamics_jepa_checkpoint_v1"
    ):
        raise ValueError("Expected dense_level_dynamics_jepa_checkpoint_v1")
    state = payload.get("online_encoder_state_dict")
    config = payload.get("online_encoder_config", payload.get("backbone_structural_config"))
    if not isinstance(state, Mapping) or not isinstance(config, Mapping):
        raise ValueError("Pretraining checkpoint lacks encoder state/config")
    report = model.load_adapted_pretrained_backbone(state, config)
    return {
        "used": True,
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        **report,
    }


def _optimizer(
    model: ObjectEventTTCV4,
    config: ObjectEventV4TrainConfig,
    *,
    pretrained: bool,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    backbone: list[torch.nn.Parameter] = []
    pooling: list[torch.nn.Parameter] = []
    for name, parameter in model.encoder.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "query_tokens" or name.startswith("query_attention."):
            pooling.append(parameter)
        else:
            backbone.append(parameter)
    definitions = [
        {
            "name": "backbone",
            "params": backbone,
            "lr": config.pretrained_backbone_learning_rate
            if pretrained
            else config.scratch_backbone_learning_rate,
            "weight_decay": config.weight_decay,
        },
        {
            "name": "pooling",
            "params": pooling,
            "lr": config.pooling_learning_rate,
            "weight_decay": config.readout_weight_decay,
        },
        {
            "name": "event_head",
            "params": list(model.event_order_scorer.parameters()),
            "lr": config.event_head_learning_rate,
            "weight_decay": config.readout_weight_decay,
        },
        {
            "name": "predictor",
            "params": list(model.local_predictor.parameters()),
            "lr": config.predictor_learning_rate,
            "weight_decay": config.weight_decay,
        },
        {
            "name": "motion",
            "params": list(model.motion_encoder.parameters())
            + list(model.motion_head.parameters()),
            "lr": config.motion_learning_rate,
            "weight_decay": config.readout_weight_decay,
        },
        {
            "name": "fusion",
            "params": list(model.event_gate_head.parameters()),
            "lr": config.fusion_learning_rate,
            "weight_decay": config.readout_weight_decay,
        },
    ]
    definitions = [group for group in definitions if group["params"]]
    optimizer = torch.optim.AdamW(definitions)
    manifest = [
        {
            "name": group["name"],
            "base_lr": group["lr"],
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in definitions
    ]
    return optimizer, manifest


def _set_lrs(
    optimizer: torch.optim.Optimizer,
    manifest: list[dict[str, Any]],
    *,
    epoch: int,
    config: ObjectEventV4TrainConfig,
    pretrained: bool,
) -> dict[str, float]:
    decay = config.lr_gamma ** sum(epoch > milestone for milestone in config.lr_milestones)
    result = {}
    for group, metadata in zip(optimizer.param_groups, manifest, strict=True):
        lr = float(metadata["base_lr"]) * decay
        if (
            pretrained
            and metadata["name"] == "backbone"
            and epoch <= config.pretrained_backbone_warmup_epochs
        ):
            lr = 0.0
        group["lr"] = lr
        result[str(metadata["name"])] = lr
    return result


def _loader(
    dataset: GarlTTCObjectEventV4Dataset,
    config: ObjectEventV4TrainConfig,
    *,
    shuffle: bool,
    epoch: int,
) -> DataLoader[ObjectEventV4Batch]:
    generator = torch.Generator().manual_seed(config.seed * 100_003 + epoch)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        generator=generator,
        collate_fn=collate_object_event_v4,
    )


def _train_epoch(
    model: ObjectEventTTCV4,
    loader: DataLoader[ObjectEventV4Batch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    train_config: ObjectEventV4TrainConfig,
    loss_config: ObjectEventV4LossConfig,
    modality_config: ObjectEventV4ModalityConfig,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    components: dict[str, float] = {}
    samples = 0
    optimizer_steps = 0
    motion_drop_count = 0
    event_drop_count = 0
    generator = torch.Generator(device=device).manual_seed(
        train_config.seed * 1_000_003 + epoch
    )
    for step, raw_batch in enumerate(loader, start=1):
        batch = raw_batch.to(device)
        dropped = apply_modality_dropout(
            batch.events,
            batch.observable_motion,
            epoch=epoch,
            config=modality_config,
            generator=generator,
        )
        motion_drop_count += int(dropped.motion_dropped.sum())
        event_drop_count += int(dropped.events_dropped.sum())
        with _autocast(device, train_config.precision):
            output = model(dropped.events, batch.delta_t_s, dropped.observable_motion)
            loss_output = object_event_v4_loss(output, batch, loss_config)
            loss = loss_output.total / train_config.gradient_accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        should_step = (
            step % train_config.gradient_accumulation_steps == 0
            or step == len(loader)
        )
        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model.update_target_encoder(train_config.target_ema_momentum)
            optimizer_steps += 1
        batch_size = batch.events.shape[0]
        samples += batch_size
        total_loss += float(loss_output.total.detach().cpu()) * batch_size
        for name, value in loss_output.components.items():
            components[name] = components.get(name, 0.0) + float(value.detach().cpu()) * batch_size
    return {
        "loss": total_loss / max(samples, 1),
        "components": {key: value / max(samples, 1) for key, value in components.items()},
        "sample_count": samples,
        "optimizer_steps": optimizer_steps,
        "motion_drop_fraction": motion_drop_count / max(samples, 1),
        "event_drop_fraction": event_drop_count / max(samples, 1),
    }


@torch.no_grad()
def _evaluate(
    model: ObjectEventTTCV4,
    loader: DataLoader[ObjectEventV4Batch],
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    reversal_errors: list[np.ndarray] = []
    gate_values: list[np.ndarray] = []
    for raw_batch in loader:
        batch = raw_batch.to(device)
        full = model(**batch.model_inputs())
        zero_events = model(
            torch.zeros_like(batch.events), batch.delta_t_s, batch.observable_motion
        )
        shuffled = model(
            torch.roll(batch.events, shifts=1, dims=0)
            if batch.events.shape[0] > 1
            else torch.flip(batch.events, dims=(1,)),
            batch.delta_t_s,
            batch.observable_motion,
        )
        target_expansion = batch.delta_t_s / batch.target_ttc_s
        reversal_errors.append(full.reversal_error.cpu().numpy())
        gate_values.append(full.event_gate.cpu().numpy())
        for index in range(batch.events.shape[0]):
            rows.append(
                {
                    "sequence_id": batch.sequence_ids[index],
                    "sample_token": batch.sample_tokens[index],
                    "track_id": batch.track_ids[index],
                    "target_ttc_s": float(batch.target_ttc_s[index].cpu()),
                    "target_expansion": float(target_expansion[index].cpu()),
                    "prediction_ttc_s": float(full.ttc_mean_seconds[index].cpu()),
                    "prediction_expansion": float(full.signed_expansion[index].cpu()),
                    "event_ttc_s": float(full.event_ttc_seconds[index].cpu()),
                    "event_expansion": float(full.event_expansion[index].cpu()),
                    "motion_ttc_s": float(full.motion_ttc_seconds[index].cpu()),
                    "motion_expansion": float(full.motion_expansion[index].cpu()),
                    "zero_events_ttc_s": float(zero_events.ttc_mean_seconds[index].cpu()),
                    "zero_events_expansion": float(zero_events.signed_expansion[index].cpu()),
                    "shuffled_events_ttc_s": float(shuffled.ttc_mean_seconds[index].cpu()),
                    "shuffled_events_expansion": float(shuffled.signed_expansion[index].cpu()),
                    "event_gate": float(full.event_gate[index].cpu()),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    target_ttc = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    target_expansion = frame["target_expansion"].to_numpy(dtype=np.float64)
    sequences = frame["sequence_id"].astype(str).to_numpy()

    def branch(prefix: str, ttc_column: str, expansion_column: str) -> dict[str, Any]:
        prediction_ttc = frame[ttc_column].to_numpy(dtype=np.float64)
        prediction_expansion = frame[expansion_column].to_numpy(dtype=np.float64)
        return {
            "name": prefix,
            "metrics": _ttc_metrics(target_ttc, prediction_ttc, sequences),
            "ttc_health": prediction_health(target_ttc, prediction_ttc),
            "expansion_health": prediction_health(target_expansion, prediction_expansion),
            "direction": _direction(target_expansion, prediction_expansion),
            "saturation_rate": float(
                np.mean(np.abs(prediction_ttc) >= model.config.ttc_clip_seconds * (1.0 - 1.0e-6))
            ),
        }

    evaluation = {
        "full": branch("full", "prediction_ttc_s", "prediction_expansion"),
        "event_only": branch("event_only", "event_ttc_s", "event_expansion"),
        "motion_only": branch("motion_only", "motion_ttc_s", "motion_expansion"),
        "zero_events": branch(
            "zero_events", "zero_events_ttc_s", "zero_events_expansion"
        ),
        "shuffled_events": branch(
            "shuffled_events", "shuffled_events_ttc_s", "shuffled_events_expansion"
        ),
        "reversal_error_max": float(np.concatenate(reversal_errors).max()),
        "reversal_error_mean": float(np.concatenate(reversal_errors).mean()),
        "event_gate_mean": float(np.concatenate(gate_values).mean()),
        "event_gate_min": float(np.concatenate(gate_values).min()),
        "event_gate_max": float(np.concatenate(gate_values).max()),
    }
    full_pearson = float(evaluation["full"]["expansion_health"]["pearson"])
    zero_pearson = float(evaluation["zero_events"]["expansion_health"]["pearson"])
    shuffled_pearson = float(
        evaluation["shuffled_events"]["expansion_health"]["pearson"]
    )
    evaluation["event_dependence"] = {
        "pearson_drop_zero_events": full_pearson - zero_pearson,
        "pearson_drop_shuffled_events": full_pearson - shuffled_pearson,
        "mean_abs_change_zero_events": float(
            np.mean(
                np.abs(
                    frame["prediction_expansion"].to_numpy()
                    - frame["zero_events_expansion"].to_numpy()
                )
            )
        ),
        "mean_abs_change_shuffled_events": float(
            np.mean(
                np.abs(
                    frame["prediction_expansion"].to_numpy()
                    - frame["shuffled_events_expansion"].to_numpy()
                )
            )
        ),
    }
    return frame, cast(dict[str, Any], _json_safe(evaluation))


def _selection_score(evaluation: Mapping[str, Any], config: ObjectEventV4TrainConfig) -> float:
    full = cast(Mapping[str, Any], evaluation["full"])
    event = cast(Mapping[str, Any], evaluation["event_only"])
    metrics = cast(Mapping[str, Any], full["metrics"])
    event_metrics = cast(Mapping[str, Any], event["metrics"])
    macro = float(
        cast(Mapping[str, Any], metrics["sequence_macro"])[
            "sequence_macro_paper_MiD_overall"
        ]
    )
    negative = float(cast(Mapping[str, Any], metrics["bins"])["negative"]["mid"])
    event_macro = float(
        cast(Mapping[str, Any], event_metrics["sequence_macro"])[
            "sequence_macro_paper_MiD_overall"
        ]
    )
    return (
        macro
        + config.negative_mid_penalty * negative
        + config.mae_penalty * float(metrics["mae_s"])
        + config.event_mid_penalty * event_macro
    )


def _eligible(
    evaluation: Mapping[str, Any],
    *,
    epoch: int,
    config: ObjectEventV4TrainConfig,
    precontext_fraction: float,
) -> tuple[bool, dict[str, Any]]:
    full = cast(Mapping[str, Any], evaluation["full"])
    event = cast(Mapping[str, Any], evaluation["event_only"])
    dependence = cast(Mapping[str, Any], evaluation["event_dependence"])
    event_direction = cast(Mapping[str, Any], event["direction"])
    gates = {
        "selection_epoch": epoch >= config.selection_start_epoch,
        "precontext_fraction": precontext_fraction >= 0.80,
        "full_expansion_pearson": float(full["expansion_health"]["pearson"])
        >= config.min_full_expansion_pearson,
        "event_only_pearson": float(event["expansion_health"]["pearson"])
        >= config.min_event_only_pearson,
        "event_only_balanced_sign": float(event_direction["balanced_accuracy"])
        >= config.min_event_only_balanced_sign_accuracy,
        "event_only_negative_recall": float(event_direction["negative_recall"])
        >= config.min_event_only_negative_recall,
        "zero_events_degradation": float(dependence["pearson_drop_zero_events"])
        >= config.min_event_dependence_zero_events,
        "shuffled_events_degradation": float(
            dependence["pearson_drop_shuffled_events"]
        )
        >= config.min_event_dependence_shuffled_events,
        "reversal_antisymmetry": float(evaluation["reversal_error_max"])
        <= config.max_reversal_error,
        "saturation": float(full["saturation_rate"])
        <= config.max_ttc_saturation_rate,
        "event_gate": float(evaluation["event_gate_mean"])
        >= config.min_event_gate_mean,
    }
    return all(bool(value) for value in gates.values()), gates


def _atomic_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def train(
    *,
    manifest_path: Path,
    output_dir: Path,
    model_config: ObjectEventV4Config,
    train_config: ObjectEventV4TrainConfig,
    loss_config: ObjectEventV4LossConfig,
    modality_config: ObjectEventV4ModalityConfig,
    provenance: Mapping[str, Any],
    device_name: str,
    pretrained: Path | None,
    resume: bool,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_config.seed)
    dirty = _dirty()
    if train_config.require_clean_git and dirty:
        raise RuntimeError("Full v4 training requires a clean git worktree")

    datasets = {
        split: GarlTTCObjectEventV4Dataset(
            str(manifest_path), splits=(split,)
        )
        for split in ("train", "validation")
    }
    manifest = datasets["train"].manifest
    precontext_fraction = float(
        manifest.get("event_v4_precontext_valid_fraction", 0.0)
    )
    if precontext_fraction < 0.80:
        raise RuntimeError(
            f"V4 cache real-event precontext coverage {precontext_fraction:.6f} is below 0.80"
        )

    model = ObjectEventTTCV4(model_config)
    transfer = _load_pretrained(model, pretrained)
    model.to(device)
    optimizer, optimizer_manifest = _optimizer(
        model, train_config, pretrained=bool(transfer["used"])
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=train_config.precision == "fp16" and device.type == "cuda",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    summary_path = output_dir / "summary.json"
    history_path = output_dir / "history.jsonl"
    best_predictions_path = output_dir / "best_predictions.csv"
    last_predictions_path = output_dir / "last_predictions.csv"

    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    if resume and last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved.get("best_score", math.inf))
        best_epoch = int(saved.get("best_epoch", 0))
        stale = int(saved.get("stale", 0))
        history = list(saved.get("history", []))
        optimizer_steps = int(saved.get("optimizer_steps", 0))

    validation_loader = _loader(
        datasets["validation"], train_config, shuffle=False, epoch=0
    )
    last_evaluation: dict[str, Any] | None = None
    for epoch in range(start_epoch, train_config.epochs + 1):
        learning_rates = _set_lrs(
            optimizer,
            optimizer_manifest,
            epoch=epoch,
            config=train_config,
            pretrained=bool(transfer["used"]),
        )
        train_result = _train_epoch(
            model,
            _loader(datasets["train"], train_config, shuffle=True, epoch=epoch),
            optimizer,
            device=device,
            epoch=epoch,
            train_config=train_config,
            loss_config=loss_config,
            modality_config=modality_config,
            scaler=scaler if scaler.is_enabled() else None,
        )
        optimizer_steps += int(train_result["optimizer_steps"])
        predictions, evaluation = _evaluate(
            model, validation_loader, device=device
        )
        last_evaluation = evaluation
        score = _selection_score(evaluation, train_config)
        eligible, gates = _eligible(
            evaluation,
            epoch=epoch,
            config=train_config,
            precontext_fraction=precontext_fraction,
        )
        improved = eligible and math.isfinite(score) and score < best_score
        if epoch >= train_config.selection_start_epoch:
            stale = 0 if improved else stale + 1
        if improved:
            best_score = score
            best_epoch = epoch
        row = cast(
            dict[str, Any],
            _json_safe(
                {
                    "epoch": epoch,
                    "train": train_result,
                    "learning_rates": learning_rates,
                    "validation_selection_score": score,
                    "validation": evaluation,
                    "selection_gates": gates,
                    "checkpoint_selection_eligible": eligible,
                }
            ),
        )
        history.append(row)
        history_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in history),
            encoding="utf-8",
        )
        checkpoint = {
            "artifact_type": "e_jepa_object_event_ttc_checkpoint_v4",
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "loss_config": asdict(loss_config),
            "modality_config": asdict(modality_config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_manifest": optimizer_manifest,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale": stale,
            "evaluation": evaluation,
            "transfer": transfer,
            "cache_manifest": manifest_path.resolve().as_posix(),
            "history": history,
        }
        _atomic_save(checkpoint, last_path)
        predictions.to_csv(last_predictions_path, index=False)
        if improved:
            inference = dict(checkpoint)
            inference.pop("optimizer_state_dict")
            _atomic_save(inference, best_path)
            predictions.to_csv(best_predictions_path, index=False)
        if epoch >= train_config.minimum_epochs and stale >= train_config.early_stopping_patience:
            break

    if not best_path.is_file():
        gate_report = last_evaluation if last_evaluation is not None else {}
        raise RuntimeError(
            "No v4 checkpoint passed the event-required gates. "
            "This is a valid falsification; inspect last.pt/history.jsonl. "
            f"Last evaluation={json.dumps(_json_safe(gate_report), ensure_ascii=False)}"
        )
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    best_predictions, best_evaluation = _evaluate(
        model, validation_loader, device=device
    )
    best_predictions.to_csv(best_predictions_path, index=False)
    ended_at = datetime.now(UTC)
    summary = cast(
        dict[str, Any],
        _json_safe(
            {
                "artifact_type": "e_jepa_object_event_ttc_training_v4",
                "status": "completed",
                "created_at": ended_at.isoformat(),
                "start_time": started_at.isoformat(),
                "end_time": ended_at.isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "git_commit": _git_commit(),
                "git_dirty": dirty,
                "host": platform.node(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "device": str(device),
                "seed": train_config.seed,
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "loss_config": asdict(loss_config),
                "modality_config": asdict(modality_config),
                "config_hash": _hash(
                    {
                        "model": asdict(model_config),
                        "train": asdict(train_config),
                        "loss": asdict(loss_config),
                        "modality": asdict(modality_config),
                        "provenance": dict(provenance),
                    }
                ),
                "cache_manifest": manifest_path.resolve().as_posix(),
                "cache_manifest_sha256": _sha256(manifest_path),
                "cache_precontext_fraction": precontext_fraction,
                "pretraining": transfer,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_checkpoint": best_path.resolve().as_posix(),
                "best_checkpoint_sha256": _sha256(best_path),
                "best_evaluation": best_evaluation,
                "last_evaluation": last_evaluation,
                "history": history,
                "uses_three_step_common_coordinate_event_roi": True,
                "uses_only_active_event_channels": True,
                "event_predictor_receives_box_motion": False,
                "event_head_receives_box_motion": False,
                "has_independent_event_only_head": True,
                "has_independent_motion_only_head": True,
                "uses_bounded_late_fusion": True,
                "uses_modality_dropout": True,
                "uses_local_token_jepa": True,
                "uses_global_embedding_jepa_v3": False,
                "uses_v3_ordered_swap_loss": False,
                "requires_event_dependence_gates": True,
                "claim_eligible": False,
                "downstream_evaluation_eligible": train_config.run_scope
                == "full_candidate"
                and not dirty,
            }
        ),
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        model_config, train_config, loss_config, modality_config, provenance = load_spec(
            args.config.resolve()
        )
        train_config = replace(
            train_config,
            **{
                key: value
                for key, value in {
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "num_workers": args.workers,
                    "precision": args.precision,
                }.items()
                if value is not None
            },
        )
        result = train(
            manifest_path=args.cache_manifest.resolve(),
            output_dir=args.output_dir.resolve(),
            model_config=model_config,
            train_config=train_config,
            loss_config=loss_config,
            modality_config=modality_config,
            provenance=provenance,
            device_name=args.device,
            pretrained=args.pretrained.resolve() if args.pretrained else None,
            resume=args.resume,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure = {
            "artifact_type": "e_jepa_object_event_ttc_training_failure_v4",
            "status": "failed",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
