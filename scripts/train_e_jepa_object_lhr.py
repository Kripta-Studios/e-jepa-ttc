#!/usr/bin/env python
"""Train object-centric JEPA-LHR from an official GarlTTC cache.

The historical full-frame direct-TTC trainer remains untouched. This trainer
requires one tracked-object ROI per sample, predicts two visible heights, and
derives signed TTC through the analytical LHR equation.
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
from e_jepa_ttc.models.object_lhr import (  # noqa: E402
    ObjectCentricLHR,
    ObjectCentricLHRConfig,
)
from e_jepa_ttc.reproducibility import resolve_device  # noqa: E402
from e_jepa_ttc.training.object_lhr import (  # noqa: E402
    ObjectLHRCurriculumConfig,
    curriculum_phase,
    mask_iou,
    object_lhr_loss,
    target_log_ratio_from_ttc,
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


def _metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    target = predictions["target_ttc_s"].to_numpy(dtype=np.float64)
    prediction = predictions["prediction_ttc_s"].to_numpy(dtype=np.float64)
    sequence = predictions["sequence_id"].astype(str).to_numpy()
    signed = signed_garl_metrics(target, prediction)
    signed["mae_s"] = float(np.mean(np.abs(target - prediction)))
    signed["median_ae_s"] = float(np.median(np.abs(target - prediction)))
    signed["sign_accuracy"] = float(np.mean(np.sign(target) == np.sign(prediction)))
    signed["sequence_macro"] = sequence_macro_signed_metrics(
        target, prediction, sequence
    )
    return signed


@dataclass(frozen=True)
class ObjectLHRTrainConfig:
    """Reproducible supervised optimization controls."""

    epochs: int = 30
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_workers: int = 2
    scratch_backbone_learning_rate: float = 1e-4
    pretrained_backbone_learning_rate: float = 1e-5
    pooling_learning_rate: float = 1e-4
    readout_learning_rate: float = 3e-4
    weight_decay: float = 0.01
    readout_weight_decay: float = 0.0
    pretrained_readout_warmup_epochs: int = 5
    lr_milestones: tuple[int, ...] = (15, 22, 27)
    lr_gamma: float = 0.2
    precision: str = "fp32"
    max_grad_norm: float = 1.0
    minimum_epochs: int = 15
    early_stopping_patience: int = 5
    min_prediction_std_ratio: float = 0.01
    min_log_ratio_std_ratio: float = 0.01
    collapse_patience: int = 5
    seed: int = 7
    run_scope: str = "bounded_screen"
    require_clean_git: bool = False
    event_field: str = "jepa_event_roi"

    def __post_init__(self) -> None:
        positive = (
            self.epochs,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.num_workers + 1,
            self.minimum_epochs,
            self.early_stopping_patience + 1,
            self.collapse_patience + 1,
        )
        if min(positive) <= 0:
            raise ValueError("Object-LHR integer controls must be positive")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs cannot exceed epochs")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if self.run_scope not in {"bounded_screen", "full_candidate"}:
            raise ValueError("run_scope must be bounded_screen or full_candidate")
        if not 0 < self.lr_gamma <= 1:
            raise ValueError("lr_gamma must lie in (0,1]")
        if any(value < 0 for value in self.lr_milestones):
            raise ValueError("lr_milestones must be non-negative")


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
) -> tuple[ObjectCentricLHRConfig, ObjectLHRTrainConfig, ObjectLHRCurriculumConfig, dict[str, Any]]:
    experiment = _read_yaml(experiment_path)
    model_path = _resolve(experiment_path, experiment["model"])
    train_path = _resolve(experiment_path, experiment["finetuning"])
    model_value = _inherited(model_path)
    train_value = _inherited(train_path)
    if model_value.get("model") != "e_jepa_object_lhr_v1":
        raise ValueError("Object-LHR trainer requires model=e_jepa_object_lhr_v1")
    model_names = {field.name for field in fields(ObjectCentricLHRConfig)}
    train_names = {field.name for field in fields(ObjectLHRTrainConfig)}
    curriculum_names = {field.name for field in fields(ObjectLHRCurriculumConfig)}
    if "lr_milestones" in train_value:
        train_value["lr_milestones"] = _tuple_milestones(train_value["lr_milestones"])
    return (
        ObjectCentricLHRConfig(
            **{key: value for key, value in model_value.items() if key in model_names}
        ),
        ObjectLHRTrainConfig(
            **{key: value for key, value in train_value.items() if key in train_names}
        ),
        ObjectLHRCurriculumConfig(
            **{key: value for key, value in train_value.items() if key in curriculum_names}
        ),
        {
            "experiment_path": experiment_path.resolve().as_posix(),
            "model_path": model_path.as_posix(),
            "train_path": train_path.as_posix(),
        },
    )


def _hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_safe(dict(value)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return bool(result.stdout.strip())


def _autocast(device: torch.device, precision: str) -> torch.autocast:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def _load_pretrained(model: ObjectCentricLHR, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"used": False, "transferred_keys": []}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("artifact_type") != "dense_level_dynamics_jepa_checkpoint_v1"
    ):
        raise ValueError("Expected a dense_level_dynamics_jepa_checkpoint_v1 artifact")
    state = payload.get("online_encoder_state_dict")
    config = payload.get("online_encoder_config", payload.get("backbone_structural_config"))
    if not isinstance(state, Mapping) or not isinstance(config, Mapping):
        raise ValueError("Pretraining checkpoint lacks exact encoder state/config")
    report = model.load_exact_backbone_state_dict(state, config)
    return {"used": True, "path": path.resolve().as_posix(), "sha256": _sha256(path), **report}


def _groups(
    model: ObjectCentricLHR,
    config: ObjectLHRTrainConfig,
    *,
    pretrained: bool,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    backbone: list[torch.nn.Parameter] = []
    pooling: list[torch.nn.Parameter] = []
    for name, parameter in model.encoder.named_parameters():
        if name == "query_tokens" or name.startswith("query_attention."):
            pooling.append(parameter)
        elif not name.startswith(("ttc_head.", "collision_head.")):
            backbone.append(parameter)
    task = [
        parameter
        for module in (
            model.height_head,
            model.pair_projector,
            model.direction_head,
            model.mask_decoder,
        )
        if module is not None
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
            "name": "readout",
            "params": task,
            "lr": config.readout_learning_rate,
            "weight_decay": config.readout_weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(definitions)
    manifest = [
        {
            "name": str(group["name"]),
            "parameter_count": sum(parameter.numel() for parameter in group["params"]),
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
    config: ObjectLHRTrainConfig,
    pretrained: bool,
) -> dict[str, float]:
    reductions = sum(epoch > milestone for milestone in config.lr_milestones)
    multiplier = config.lr_gamma**reductions
    warmup = pretrained and epoch <= config.pretrained_readout_warmup_epochs
    result: dict[str, float] = {}
    for group, spec in zip(optimizer.param_groups, manifest, strict=True):
        name = str(spec["name"])
        lr = float(spec["base_lr"]) * multiplier
        if warmup and name == "backbone":
            lr = 0.0
        group["lr"] = lr
        result[name] = lr
    return result


def _loader(
    dataset: GarlTTCObjectLHRDataset,
    config: ObjectLHRTrainConfig,
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
        collate_fn=partial(collate_object_lhr, event_field=config.event_field),
    )


def _train_epoch(
    model: ObjectCentricLHR,
    loader: DataLoader[ObjectLHRBatch],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epoch: int,
    train_config: ObjectLHRTrainConfig,
    curriculum: ObjectLHRCurriculumConfig,
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
            loss = object_lhr_loss(output, batch, epoch=epoch, config=curriculum)
            backward = loss.total / train_config.gradient_accumulation_steps
        update = (
            (batch_index + 1) % train_config.gradient_accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if scaler is None:
            backward.backward()
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
        else:
            scaler.scale(backward).backward()
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
        totals.append(float(loss.total.detach().cpu()))
        for name, value in loss.components.items():
            components.setdefault(name, []).append(float(value.detach().cpu()))
    if not totals:
        raise RuntimeError("Object-LHR training produced no batches")
    return {
        "phase": curriculum_phase(epoch, curriculum),
        "loss": float(np.mean(totals)),
        "optimizer_steps": steps,
        "components": {name: float(np.mean(values)) for name, values in components.items()},
    }


@torch.no_grad()
def _evaluate(
    model: ObjectCentricLHR,
    loader: DataLoader[ObjectLHRBatch],
    *,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    ratio_targets: list[np.ndarray] = []
    ratio_predictions: list[np.ndarray] = []
    height_errors: list[float] = []
    mask_scores: list[float] = []
    for raw_batch in loader:
        batch = raw_batch.to(device)
        output = model(**batch.model_inputs())
        target_log_ratio = target_log_ratio_from_ttc(batch.delta_t_s, batch.target_ttc_s)
        ratio_targets.append(target_log_ratio.cpu().numpy())
        ratio_predictions.append(output.log_height_ratio.cpu().numpy())
        height_errors.append(
            float(
                torch.abs(
                    output.log_visible_heights - torch.log(batch.visible_heights_px)
                )
                .mean()
                .cpu()
            )
        )
        score = mask_iou(output.mask_logits, batch.masks, batch.mask_valid)
        if score is not None:
            mask_scores.append(score)
        for index in range(len(batch.sample_tokens)):
            rows.append(
                {
                    "sample_token": batch.sample_tokens[index],
                    "sequence_id": batch.sequence_ids[index],
                    "track_id": batch.track_ids[index],
                    "target_ttc_s": float(batch.target_ttc_s[index].cpu()),
                    "prediction_ttc_s": float(output.ttc_mean_seconds[index].cpu()),
                    "target_log_height_ratio": float(target_log_ratio[index].cpu()),
                    "prediction_log_height_ratio": float(output.log_height_ratio[index].cpu()),
                    "target_height_t1_px": float(batch.visible_heights_px[index, 0].cpu()),
                    "target_height_t2_px": float(batch.visible_heights_px[index, 1].cpu()),
                    "prediction_height_t1_px": float(output.visible_heights_px[index, 0].cpu()),
                    "prediction_height_t2_px": float(output.visible_heights_px[index, 1].cpu()),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    ttc_health = prediction_health(
        frame["target_ttc_s"].to_numpy(),
        frame["prediction_ttc_s"].to_numpy(),
    )
    target_ratio = np.concatenate(ratio_targets)
    prediction_ratio = np.concatenate(ratio_predictions)
    ratio_health = prediction_health(target_ratio, prediction_ratio)
    return frame, {
        "metrics": _metrics(frame),
        "ttc_health": ttc_health,
        "log_ratio_health": ratio_health,
        "log_height_mae": float(np.mean(height_errors)),
        "mask_iou": float(np.mean(mask_scores)) if mask_scores else None,
    }


def _atomic_save(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def train(
    *,
    manifest_path: Path,
    output_dir: Path,
    model_config: ObjectCentricLHRConfig,
    train_config: ObjectLHRTrainConfig,
    curriculum: ObjectLHRCurriculumConfig,
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
        raise RuntimeError("full_candidate requires a clean committed Git worktree")

    datasets = {
        split: GarlTTCObjectLHRDataset(
            str(manifest_path), splits=(split,), event_field=train_config.event_field
        )
        for split in ("train", "validation")
    }
    model = ObjectCentricLHR(model_config).to(device)
    transfer = _load_pretrained(model, pretrained)
    optimizer, optimizer_manifest = _groups(model, train_config, pretrained=bool(transfer["used"]))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and train_config.precision == "fp16")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "best_validation_predictions.csv"
    if not resume:
        for path in (
            last_path,
            best_path,
            summary_path,
            predictions_path,
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

    validation_loader = _loader(datasets["validation"], train_config, shuffle=False, epoch=0)
    final_evaluation: dict[str, Any] | None = None
    for epoch in range(start_epoch, train_config.epochs + 1):
        lrs = _set_epoch_lrs(
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
            curriculum=curriculum,
            scaler=scaler if scaler.is_enabled() else None,
        )
        optimizer_steps += int(train_result["optimizer_steps"])
        predictions, evaluation = _evaluate(model, validation_loader, device=device)
        final_evaluation = evaluation
        metrics = evaluation["metrics"]
        ttc_health = evaluation["ttc_health"]
        ratio_health = evaluation["log_ratio_health"]
        full_phase = curriculum_phase(epoch, curriculum) == "full_mid"
        collapsed = (
            float(ttc_health["prediction_std_ratio"]) < train_config.min_prediction_std_ratio
            or float(ratio_health["prediction_std_ratio"]) < train_config.min_log_ratio_std_ratio
        )
        collapse_streak = collapse_streak + 1 if full_phase and collapsed else 0
        score = float(metrics["sequence_macro"]["sequence_macro_paper_MiD_overall"])
        eligible = full_phase and not collapsed and math.isfinite(score)
        improved = eligible and score < best_score
        if full_phase:
            stale = 0 if improved else stale + 1
        if improved:
            best_score, best_epoch = score, epoch
        row = {
            "epoch": epoch,
            "phase": train_result["phase"],
            "train_loss": train_result["loss"],
            "train_components": train_result["components"],
            "optimizer_steps": optimizer_steps,
            "learning_rates": lrs,
            "validation_sequence_macro_paper_MiD_overall": score,
            "validation_ttc_health": ttc_health,
            "validation_log_ratio_health": ratio_health,
            "validation_log_height_mae": evaluation["log_height_mae"],
            "validation_mask_iou": evaluation["mask_iou"],
            "collapsed": collapsed,
            "checkpoint_selection_eligible": eligible,
        }
        history.append(cast(dict[str, Any], _json_safe(row)))
        checkpoint = {
            "artifact_type": "e_jepa_object_lhr_checkpoint_v1",
            "architecture": "e_jepa_object_lhr_v1",
            "modality": "event_only_object_roi",
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "curriculum_config": asdict(curriculum),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_manifest": optimizer_manifest,
            "epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "stale": stale,
            "collapse_streak": collapse_streak,
            "validation": evaluation,
            "pretraining": transfer,
            "cache_manifest": manifest_path.resolve().as_posix(),
            "cache_manifest_sha256": _sha256(manifest_path),
            "history": history,
        }
        _atomic_save(checkpoint, last_path)
        if improved:
            inference = dict(checkpoint)
            inference.pop("optimizer_state_dict")
            _atomic_save(inference, best_path)
            predictions.to_csv(predictions_path, index=False)
        if (
            epoch >= train_config.minimum_epochs
            and collapse_streak >= train_config.collapse_patience
        ):
            raise RuntimeError(
                "Object-LHR remained collapsed after the full geometric curriculum; "
                f"ttc_std_ratio={ttc_health['prediction_std_ratio']}, "
                f"log_ratio_std_ratio={ratio_health['prediction_std_ratio']}"
            )
        if (
            epoch >= train_config.minimum_epochs
            and full_phase
            and stale >= train_config.early_stopping_patience
        ):
            break

    if not best_path.is_file():
        raise RuntimeError("No non-collapsed full-curriculum Object-LHR checkpoint was selected")
    ended_at = datetime.now(UTC)
    summary = {
        "artifact_type": "e_jepa_object_lhr_training_v1",
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
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
        "seed": train_config.seed,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "curriculum_config": asdict(curriculum),
        "config_hash": _hash(
            {
                "model": asdict(model_config),
                "train": asdict(train_config),
                "curriculum": asdict(curriculum),
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
        "final_evaluation": final_evaluation,
        "history": history,
        "uses_object_roi": True,
        "uses_boxes_for_model_input": False,
        "predicts_visible_heights": True,
        "derives_ttc_analytically": True,
        "uses_direct_ttc_as_primary_loss": False,
        "claim_eligible": False,
        "downstream_evaluation_eligible": train_config.run_scope == "full_candidate" and not dirty,
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
        model_config, train_config, curriculum, provenance = load_spec(args.config.resolve())
        overrides = {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.workers,
            "precision": args.precision,
        }
        train_config = replace(
            train_config, **{key: value for key, value in overrides.items() if value is not None}
        )
        result = train(
            manifest_path=args.cache_manifest.resolve(),
            output_dir=args.output_dir.resolve(),
            model_config=model_config,
            train_config=train_config,
            curriculum=curriculum,
            provenance=provenance,
            device_name=args.device,
            pretrained=args.pretrained.resolve() if args.pretrained else None,
            resume=args.resume,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_type": "e_jepa_object_lhr_training_failure_v1",
            "status": "failed",
            "created_at": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(failure, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
