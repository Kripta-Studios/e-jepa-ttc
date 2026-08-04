#!/usr/bin/env python
"""Train the stable object-centric inverse-TTC candidate.

This trainer is intentionally separate from ``train_e_jepa_object_lhr.py`` so
all v1 checkpoints and negative results remain reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from functools import partial
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

from e_jepa_ttc.data.garlttc_object_lhr import (  # noqa: E402
    GarlTTCObjectLHRDataset,
    ObjectLHRBatch,
    collate_object_lhr,
)
from e_jepa_ttc.evaluation.garl_ttc_protocol import (  # noqa: E402
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)
from e_jepa_ttc.models.object_expansion import (  # noqa: E402
    ObjectCentricExpansionTTC,
    ObjectExpansionConfig,
)
from e_jepa_ttc.reproducibility import resolve_device  # noqa: E402
from e_jepa_ttc.training.object_expansion import (  # noqa: E402
    ObjectExpansionLossConfig,
    curriculum_phase,
    object_expansion_loss,
    targets_from_batch,
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
    payload = json.dumps(
        _json_safe(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    target = predictions["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction = predictions["prediction_ttc_s"].to_numpy(dtype=np.float64)
    sequence = predictions["sequence_id"].astype(str).to_numpy()
    signed = signed_garl_metrics(target, prediction)
    signed["mae_s"] = float(np.mean(np.abs(target - prediction)))
    signed["median_ae_s"] = float(np.median(np.abs(target - prediction)))
    signed["sign_accuracy"] = float(
        np.mean(np.sign(target) == np.sign(prediction))
    )
    signed["sequence_macro"] = sequence_macro_signed_metrics(
        target,
        prediction,
        sequence,
    )
    return signed


@dataclass(frozen=True)
class ObjectExpansionTrainConfig:
    """Reproducible optimization and checkpoint-gating controls."""

    epochs: int = 30
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_workers: int = 2
    scratch_backbone_learning_rate: float = 1.0e-4
    pretrained_backbone_learning_rate: float = 1.0e-5
    pooling_learning_rate: float = 1.0e-4
    adapter_learning_rate: float = 3.0e-4
    readout_learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    readout_weight_decay: float = 0.0
    pretrained_backbone_warmup_epochs: int = 5
    lr_milestones: tuple[int, ...] = (15, 22, 27)
    lr_gamma: float = 0.2
    precision: str = "fp32"
    max_grad_norm: float = 1.0
    selection_start_epoch: int = 6
    minimum_epochs: int = 12
    early_stopping_patience: int = 6
    collapse_patience: int = 5
    min_inverse_std_ratio: float = 0.05
    min_log_ratio_pearson: float = 0.05
    min_balanced_sign_accuracy: float = 0.45
    max_ttc_saturation_rate: float = 0.05
    seed: int = 7
    run_scope: str = "bounded_screen"
    require_clean_git: bool = False
    event_field: str = "jepa_event_roi"

    def __post_init__(self) -> None:
        integers = (
            self.epochs,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.num_workers + 1,
            self.selection_start_epoch,
            self.minimum_epochs,
            self.early_stopping_patience + 1,
            self.collapse_patience + 1,
        )
        if min(integers) <= 0:
            raise ValueError("Object-expansion integer controls must be positive")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs")
        if self.selection_start_epoch > self.epochs:
            raise ValueError("selection_start_epoch cannot exceed epochs")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if self.run_scope not in {"bounded_screen", "full_candidate"}:
            raise ValueError("run_scope must be bounded_screen or full_candidate")
        if not 0.0 < self.lr_gamma <= 1.0:
            raise ValueError("lr_gamma must lie in (0,1]")
        if any(value < 0 for value in self.lr_milestones):
            raise ValueError("lr_milestones must be non-negative")
        if not 0.0 <= self.max_ttc_saturation_rate <= 1.0:
            raise ValueError("max_ttc_saturation_rate must lie in [0,1]")
        if not 0.0 <= self.min_balanced_sign_accuracy <= 1.0:
            raise ValueError("min_balanced_sign_accuracy must lie in [0,1]")


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


def _inherited(path: Path) -> dict[str, Any]:
    value = _read_yaml(path)
    base = value.pop("base", None)
    if base is None:
        return value
    merged = _inherited(_resolve(path, base))
    merged.update(value)
    return merged


def _tuple_milestones(value: object) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    raise ValueError("lr_milestones must be a YAML list")


def load_spec(
    experiment_path: Path,
) -> tuple[
    ObjectExpansionConfig,
    ObjectExpansionTrainConfig,
    ObjectExpansionLossConfig,
    dict[str, Any],
]:
    experiment = _read_yaml(experiment_path)
    model_path = _resolve(experiment_path, experiment["model"])
    train_path = _resolve(experiment_path, experiment["finetuning"])
    model_value = _inherited(model_path)
    train_value = _inherited(train_path)
    if model_value.get("model") != "e_jepa_object_expansion_v2":
        raise ValueError(
            "Object-expansion trainer requires model=e_jepa_object_expansion_v2"
        )
    model_names = {field.name for field in fields(ObjectExpansionConfig)}
    train_names = {field.name for field in fields(ObjectExpansionTrainConfig)}
    loss_names = {field.name for field in fields(ObjectExpansionLossConfig)}
    if "lr_milestones" in train_value:
        train_value["lr_milestones"] = _tuple_milestones(
            train_value["lr_milestones"]
        )
    return (
        ObjectExpansionConfig(
            **{key: value for key, value in model_value.items() if key in model_names}
        ),
        ObjectExpansionTrainConfig(
            **{key: value for key, value in train_value.items() if key in train_names}
        ),
        ObjectExpansionLossConfig(
            **{key: value for key, value in train_value.items() if key in loss_names}
        ),
        {
            "experiment_path": experiment_path.resolve().as_posix(),
            "model_path": model_path.as_posix(),
            "train_path": train_path.as_posix(),
            "protocol": experiment.get("protocol"),
            "selection": experiment.get("selection"),
        },
    )


def _autocast(device: torch.device, precision: str) -> torch.autocast:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=dtype is not None,
    )


def _load_pretrained(
    model: ObjectCentricExpansionTTC,
    path: Path | None,
) -> dict[str, Any]:
    if path is None:
        return {"used": False, "transferred_keys": []}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type")
        != "dense_level_dynamics_jepa_checkpoint_v1"
    ):
        raise ValueError("Expected dense_level_dynamics_jepa_checkpoint_v1")
    state = payload.get("online_encoder_state_dict")
    config = payload.get(
        "online_encoder_config",
        payload.get("backbone_structural_config"),
    )
    if not isinstance(state, Mapping) or not isinstance(config, Mapping):
        raise ValueError("Pretraining checkpoint lacks exact encoder state/config")
    report = model.load_exact_backbone_state_dict(state, config)
    return {
        "used": True,
        "path": path.resolve().as_posix(),
        "sha256": _sha256(path),
        **report,
    }


def _parameter_groups(
    model: ObjectCentricExpansionTTC,
    config: ObjectExpansionTrainConfig,
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
    adapter = [
        parameter
        for module in (model.endpoint_adapter, model.pair_projector)
        for parameter in module.parameters()
    ]
    readout = [
        parameter
        for module in (model.log_magnitude_head, model.direction_head)
        for parameter in module.parameters()
    ]
    backbone_lr = (
        config.pretrained_backbone_learning_rate
        if pretrained
        else config.scratch_backbone_learning_rate
    )
    definitions = [
        {
            "name": "backbone",
            "params": backbone,
            "lr": backbone_lr,
            "weight_decay": config.weight_decay,
        },
        {
            "name": "pooling",
            "params": pooling,
            "lr": config.pooling_learning_rate,
            "weight_decay": config.readout_weight_decay,
        },
        {
            "name": "adapter",
            "params": adapter,
            "lr": config.adapter_learning_rate,
            "weight_decay": config.readout_weight_decay,
        },
        {
            "name": "readout",
            "params": readout,
            "lr": config.readout_learning_rate,
            "weight_decay": config.readout_weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(definitions)
    manifest = [
        {
            "name": str(group["name"]),
            "parameter_count": sum(
                parameter.numel() for parameter in group["params"]
            ),
            "base_lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in definitions
    ]
    return optimizer, manifest


def _set_epoch_lrs(
    optimizer: torch.optim.Optimizer,
    manifest: list[dict[str, Any]],
    *,
    epoch: int,
    config: ObjectExpansionTrainConfig,
    pretrained: bool,
) -> dict[str, float]:
    reductions = sum(epoch > milestone for milestone in config.lr_milestones)
    multiplier = config.lr_gamma**reductions
    backbone_warmup = (
        pretrained and epoch <= config.pretrained_backbone_warmup_epochs
    )
    result: dict[str, float] = {}
    for group, spec in zip(optimizer.param_groups, manifest, strict=True):
        name = str(spec["name"])
        lr = float(spec["base_lr"]) * multiplier
        if backbone_warmup and name == "backbone":
            lr = 0.0
        group["lr"] = lr
        result[name] = lr
    return result


def _loader(
    dataset: GarlTTCObjectLHRDataset,
    config: ObjectExpansionTrainConfig,
    *,
    shuffle: bool,
    epoch: int,
) -> DataLoader[ObjectLHRBatch]:
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
        generator=generator,
        collate_fn=partial(
            collate_object_lhr,
            event_field=config.event_field,
        ),
    )


def _train_epoch(
    model: ObjectCentricExpansionTTC,
    loader: DataLoader[ObjectLHRBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    train_config: ObjectExpansionTrainConfig,
    loss_config: ObjectExpansionLossConfig,
    scaler: Any | None,
) -> dict[str, Any]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals: list[float] = []
    components: dict[str, list[float]] = {}
    steps = 0
    for batch_index, raw_batch in enumerate(loader):
        batch = raw_batch.to(device)
        with _autocast(device, train_config.precision):
            output = model(**batch.model_inputs())
            loss = object_expansion_loss(
                output,
                batch,
                epoch=epoch,
                config=loss_config,
            )
            backward = loss.total / train_config.gradient_accumulation_steps
        update = (
            (batch_index + 1) % train_config.gradient_accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if scaler is None:
            backward.backward()
            if update:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
        else:
            scaler.scale(backward).backward()
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
        totals.append(float(loss.total.detach().cpu()))
        for name, value in loss.components.items():
            components.setdefault(name, []).append(float(value.detach().cpu()))
    if not totals:
        raise RuntimeError("Object-expansion training produced no batches")
    return {
        "phase": curriculum_phase(epoch, loss_config),
        "loss": float(np.mean(totals)),
        "optimizer_steps": steps,
        "components": {
            name: float(np.mean(values))
            for name, values in components.items()
        },
    }


def _class_accuracy(
    target_negative: np.ndarray,
    predicted_negative: np.ndarray,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    values: list[float] = []
    for name, class_value in (("positive", False), ("negative", True)):
        mask = target_negative == class_value
        accuracy = (
            float(np.mean(predicted_negative[mask] == target_negative[mask]))
            if bool(mask.any())
            else None
        )
        result[f"{name}_accuracy"] = accuracy
        if accuracy is not None:
            values.append(accuracy)
    result["balanced_accuracy"] = float(np.mean(values)) if values else None
    return result


@torch.no_grad()
def _evaluate(
    model: ObjectCentricExpansionTTC,
    loader: DataLoader[ObjectLHRBatch],
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = raw_batch.to(device)
        output = model(**batch.model_inputs())
        targets = targets_from_batch(batch)
        for index in range(len(batch.sample_tokens)):
            rows.append(
                {
                    "sample_token": batch.sample_tokens[index],
                    "sequence_id": batch.sequence_ids[index],
                    "track_id": batch.track_ids[index],
                    "target_ttc_s": float(batch.target_ttc_s[index].cpu()),
                    "prediction_ttc_s": float(
                        output.ttc_mean_seconds[index].cpu()
                    ),
                    "target_signed_inverse_ttc": float(
                        targets.signed_inverse_ttc[index].cpu()
                    ),
                    "prediction_signed_inverse_ttc": float(
                        output.signed_inverse_ttc_hard[index].cpu()
                    ),
                    "prediction_signed_inverse_ttc_soft": float(
                        output.signed_inverse_ttc_soft[index].cpu()
                    ),
                    "target_log_abs_inverse_ttc": float(
                        targets.log_abs_inverse_ttc[index].cpu()
                    ),
                    "prediction_log_abs_inverse_ttc": float(
                        output.log_abs_inverse_ttc[index].cpu()
                    ),
                    "target_log_height_ratio": float(
                        targets.log_height_ratio[index].cpu()
                    ),
                    "prediction_log_height_ratio": float(
                        output.log_height_ratio_soft[index].cpu()
                    ),
                    "target_negative": bool(
                        targets.negative_class[index].cpu()
                    ),
                    "prediction_negative": bool(
                        output.direction_logits[index].argmax().cpu() == 1
                    ),
                    "negative_probability": float(
                        output.negative_probability[index].cpu()
                    ),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise RuntimeError("Object-expansion evaluation produced no rows")
    target_ttc = frame["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction_ttc = frame["prediction_ttc_s"].to_numpy(dtype=np.float64)
    target_inverse = frame["target_signed_inverse_ttc"].to_numpy(
        dtype=np.float64
    )
    prediction_inverse = frame["prediction_signed_inverse_ttc"].to_numpy(
        dtype=np.float64
    )
    target_ratio = frame["target_log_height_ratio"].to_numpy(dtype=np.float64)
    prediction_ratio = frame["prediction_log_height_ratio"].to_numpy(
        dtype=np.float64
    )
    target_negative = frame["target_negative"].to_numpy(dtype=bool)
    prediction_negative = frame["prediction_negative"].to_numpy(dtype=bool)
    clip = model.config.ttc_clip_seconds
    saturation_rate = float(
        np.mean(np.abs(prediction_ttc) >= clip * (1.0 - 1.0e-6))
    )
    return frame, {
        "metrics": _metrics(frame),
        "ttc_health": prediction_health(target_ttc, prediction_ttc),
        "signed_inverse_health": prediction_health(
            target_inverse,
            prediction_inverse,
        ),
        "log_ratio_health": prediction_health(target_ratio, prediction_ratio),
        "log_abs_inverse_mae": float(
            np.mean(
                np.abs(
                    frame["target_log_abs_inverse_ttc"].to_numpy(
                        dtype=np.float64
                    )
                    - frame["prediction_log_abs_inverse_ttc"].to_numpy(
                        dtype=np.float64
                    )
                )
            )
        ),
        "direction": _class_accuracy(
            target_negative,
            prediction_negative,
        ),
        "ttc_saturation_rate": saturation_rate,
    }


def _atomic_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def _eligible(
    evaluation: Mapping[str, Any],
    *,
    epoch: int,
    config: ObjectExpansionTrainConfig,
) -> tuple[bool, dict[str, bool]]:
    inverse_health = cast(Mapping[str, Any], evaluation["signed_inverse_health"])
    ratio_health = cast(Mapping[str, Any], evaluation["log_ratio_health"])
    direction = cast(Mapping[str, Any], evaluation["direction"])
    balanced = direction.get("balanced_accuracy")
    gates = {
        "selection_epoch": epoch >= config.selection_start_epoch,
        "inverse_variation": float(inverse_health["prediction_std_ratio"])
        >= config.min_inverse_std_ratio,
        "ratio_correlation": float(ratio_health["pearson"])
        >= config.min_log_ratio_pearson,
        "balanced_sign": balanced is not None
        and float(balanced) >= config.min_balanced_sign_accuracy,
        "saturation": float(evaluation["ttc_saturation_rate"])
        <= config.max_ttc_saturation_rate,
    }
    return all(gates.values()), gates


def _checkpoint_payload(
    *,
    model: ObjectCentricExpansionTTC,
    optimizer: torch.optim.Optimizer,
    optimizer_manifest: list[dict[str, Any]],
    model_config: ObjectExpansionConfig,
    train_config: ObjectExpansionTrainConfig,
    loss_config: ObjectExpansionLossConfig,
    epoch: int,
    optimizer_steps: int,
    best_score: float,
    best_epoch: int,
    stale: int,
    collapse_streak: int,
    evaluation: Mapping[str, Any],
    transfer: Mapping[str, Any],
    manifest_path: Path,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "artifact_type": "e_jepa_object_expansion_checkpoint_v2",
        "architecture": "e_jepa_object_expansion_v2",
        "modality": "event_only_object_roi",
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "loss_config": asdict(loss_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_manifest": optimizer_manifest,
        "epoch": epoch,
        "optimizer_steps": optimizer_steps,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "stale": stale,
        "collapse_streak": collapse_streak,
        "validation": dict(evaluation),
        "pretraining": dict(transfer),
        "cache_manifest": manifest_path.resolve().as_posix(),
        "cache_manifest_sha256": _sha256(manifest_path),
        "history": history,
    }


def train(
    *,
    manifest_path: Path,
    output_dir: Path,
    model_config: ObjectExpansionConfig,
    train_config: ObjectExpansionTrainConfig,
    loss_config: ObjectExpansionLossConfig,
    provenance: Mapping[str, Any],
    device_name: str,
    pretrained: Path | None,
    resume: bool,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    started = time.perf_counter()
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    device = resolve_device(device_name)
    dirty = _dirty()
    if train_config.require_clean_git and dirty:
        raise RuntimeError("full_candidate requires a clean committed worktree")

    datasets = {
        split: GarlTTCObjectLHRDataset(
            str(manifest_path),
            splits=(split,),
            event_field=train_config.event_field,
        )
        for split in ("train", "validation")
    }
    model = ObjectCentricExpansionTTC(model_config).to(device)
    transfer = _load_pretrained(model, pretrained)
    optimizer, optimizer_manifest = _parameter_groups(
        model,
        train_config,
        pretrained=bool(transfer["used"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and train_config.precision == "fp16"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    summary_path = output_dir / "summary.json"
    best_predictions_path = output_dir / "best_validation_predictions.csv"
    last_predictions_path = output_dir / "last_validation_predictions.csv"
    if not resume:
        for path in (
            last_path,
            best_path,
            summary_path,
            best_predictions_path,
            last_predictions_path,
            output_dir / "FAILURE.json",
        ):
            path.unlink(missing_ok=True)

    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
    stale = 0
    collapse_streak = 0
    optimizer_steps = 0
    history: list[dict[str, Any]] = []
    if resume and last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best_score = float(saved.get("best_score", math.inf))
        best_epoch = int(saved.get("best_epoch", 0))
        stale = int(saved.get("stale", 0))
        collapse_streak = int(saved.get("collapse_streak", 0))
        optimizer_steps = int(saved.get("optimizer_steps", 0))
        history = list(saved.get("history", []))

    validation_loader = _loader(
        datasets["validation"],
        train_config,
        shuffle=False,
        epoch=0,
    )
    last_evaluation: dict[str, Any] | None = None
    last_predictions: pd.DataFrame | None = None

    for epoch in range(start_epoch, train_config.epochs + 1):
        learning_rates = _set_epoch_lrs(
            optimizer,
            optimizer_manifest,
            epoch=epoch,
            config=train_config,
            pretrained=bool(transfer["used"]),
        )
        train_result = _train_epoch(
            model,
            _loader(
                datasets["train"],
                train_config,
                shuffle=True,
                epoch=epoch,
            ),
            optimizer,
            device=device,
            epoch=epoch,
            train_config=train_config,
            loss_config=loss_config,
            scaler=scaler if scaler.is_enabled() else None,
        )
        optimizer_steps += int(train_result["optimizer_steps"])
        predictions, evaluation = _evaluate(
            model,
            validation_loader,
            device=device,
        )
        last_evaluation = evaluation
        last_predictions = predictions
        metrics = cast(Mapping[str, Any], evaluation["metrics"])
        macro = cast(Mapping[str, Any], metrics["sequence_macro"])
        score = float(macro["sequence_macro_paper_MiD_overall"])
        eligible, gates = _eligible(
            evaluation,
            epoch=epoch,
            config=train_config,
        )
        inverse_health = cast(
            Mapping[str, Any],
            evaluation["signed_inverse_health"],
        )
        collapsed = (
            epoch >= train_config.selection_start_epoch
            and float(inverse_health["prediction_std_ratio"])
            < train_config.min_inverse_std_ratio
        )
        collapse_streak = collapse_streak + 1 if collapsed else 0
        improved = eligible and math.isfinite(score) and score < best_score
        if epoch >= train_config.selection_start_epoch:
            stale = 0 if improved else stale + 1
        if improved:
            best_score = score
            best_epoch = epoch

        row = {
            "epoch": epoch,
            "phase": train_result["phase"],
            "train_loss": train_result["loss"],
            "train_components": train_result["components"],
            "optimizer_steps": optimizer_steps,
            "learning_rates": learning_rates,
            "validation_sequence_macro_paper_MiD_overall": score,
            "validation_ttc_health": evaluation["ttc_health"],
            "validation_signed_inverse_health": evaluation[
                "signed_inverse_health"
            ],
            "validation_log_ratio_health": evaluation["log_ratio_health"],
            "validation_direction": evaluation["direction"],
            "validation_ttc_saturation_rate": evaluation[
                "ttc_saturation_rate"
            ],
            "selection_gates": gates,
            "checkpoint_selection_eligible": eligible,
            "collapsed": collapsed,
        }
        history.append(cast(dict[str, Any], _json_safe(row)))
        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            optimizer_manifest=optimizer_manifest,
            model_config=model_config,
            train_config=train_config,
            loss_config=loss_config,
            epoch=epoch,
            optimizer_steps=optimizer_steps,
            best_score=best_score,
            best_epoch=best_epoch,
            stale=stale,
            collapse_streak=collapse_streak,
            evaluation=evaluation,
            transfer=transfer,
            manifest_path=manifest_path,
            history=history,
        )
        _atomic_save(checkpoint, last_path)
        predictions.to_csv(last_predictions_path, index=False)
        if improved:
            inference = dict(checkpoint)
            inference.pop("optimizer_state_dict")
            _atomic_save(inference, best_path)
            predictions.to_csv(best_predictions_path, index=False)

        if (
            epoch >= train_config.minimum_epochs
            and collapse_streak >= train_config.collapse_patience
        ):
            raise RuntimeError(
                "Object-expansion inverse-TTC output remained collapsed; "
                f"std_ratio={inverse_health['prediction_std_ratio']}"
            )
        if (
            epoch >= train_config.minimum_epochs
            and stale >= train_config.early_stopping_patience
        ):
            break

    if not best_path.is_file():
        raise RuntimeError(
            "No Object-expansion checkpoint passed all validation gates"
        )
    if last_evaluation is None or last_predictions is None:
        raise RuntimeError("Training completed without a validation evaluation")

    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    best_predictions, best_evaluation = _evaluate(
        model,
        validation_loader,
        device=device,
    )
    best_predictions.to_csv(best_predictions_path, index=False)

    ended_at = datetime.now(UTC)
    summary = {
        "artifact_type": "e_jepa_object_expansion_training_v2",
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
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "device": str(device),
        "seed": train_config.seed,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "loss_config": asdict(loss_config),
        "config_hash": _hash(
            {
                "model": asdict(model_config),
                "train": asdict(train_config),
                "loss": asdict(loss_config),
                "provenance": dict(provenance),
            }
        ),
        "cache_manifest": manifest_path.resolve().as_posix(),
        "cache_manifest_sha256": _sha256(manifest_path),
        "pretraining": transfer,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_checkpoint": best_path.resolve().as_posix(),
        "best_checkpoint_sha256": _sha256(best_path),
        "best_evaluation": best_evaluation,
        "last_evaluation": last_evaluation,
        "final_evaluation": best_evaluation,
        "history": history,
        "uses_object_roi": True,
        "uses_boxes_for_model_input": False,
        "predicts_absolute_visible_heights": False,
        "predicts_inverse_ttc_magnitude": True,
        "predicts_ttc_direction": True,
        "uses_log_ratio_as_consistency_loss": True,
        "uses_direct_ttc_as_primary_loss": False,
        "reloads_best_checkpoint_for_final_evaluation": True,
        "claim_eligible": False,
        "downstream_evaluation_eligible": (
            train_config.run_scope == "full_candidate" and not dirty
        ),
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return cast(dict[str, Any], _json_safe(summary))


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
        model_config, train_config, loss_config, provenance = load_spec(
            args.config.resolve()
        )
        overrides = {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "precision": args.precision,
        }
        train_config = replace(
            train_config,
            **{
                key: value
                for key, value in overrides.items()
                if value is not None
            },
        )
        result = train(
            manifest_path=args.cache_manifest.resolve(),
            output_dir=args.output_dir.resolve(),
            model_config=model_config,
            train_config=train_config,
            loss_config=loss_config,
            provenance=provenance,
            device_name=args.device,
            pretrained=args.pretrained.resolve() if args.pretrained else None,
            resume=args.resume,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        failure = {
            "artifact_type": "e_jepa_object_expansion_training_failure_v2",
            "status": "failed",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
