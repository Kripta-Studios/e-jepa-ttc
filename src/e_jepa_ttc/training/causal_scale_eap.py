"""Bounded real-data training for the causal event foreground-scale arm."""

from __future__ import annotations

import copy
import math
import os
import random
import time
from collections.abc import Sized
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from e_jepa_ttc.data.object_event_v4 import (
    ObjectEventV4Batch,
    collate_object_event_v4,
    weak_box_masks,
)
from e_jepa_ttc.evaluation.garl_ttc_protocol import (
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)
from e_jepa_ttc.losses.causal_scale_ttc import (
    CausalScaleTTCLossConfig,
    causal_scale_ttc_loss,
)
from e_jepa_ttc.models.causal_scale_ttc import (
    CausalScaleTTC,
    CausalScaleTTCConfig,
    target_log_ratio_from_ttc,
)
from e_jepa_ttc.reproducibility import seed_everything


@dataclass(frozen=True)
class CausalScaleEAPTrainingConfig:
    """Optimizer controls for a validation-only public eAP/Garl-TTC screen."""

    seed: int = 7
    epochs: int = 18
    minimum_epochs: int = 8
    early_stopping_patience: int = 5
    foreground_warmup_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3.0e-4
    minimum_learning_rate: float = 3.0e-5
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    precision: str = "bf16"
    maximum_runtime_hours: float = 6.0
    mask_t0_as_proxy: bool = True

    def __post_init__(self) -> None:
        integers = (
            self.epochs,
            self.minimum_epochs,
            self.early_stopping_patience + 1,
            self.foreground_warmup_epochs + 1,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.num_workers + 1,
        )
        if self.seed < 0 or min(integers) <= 0:
            raise ValueError("invalid causal-scale eAP integer controls")
        if self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs exceeds epochs")
        if self.foreground_warmup_epochs >= self.epochs:
            raise ValueError("foreground warmup must finish before the last epoch")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if not 0.0 <= self.minimum_learning_rate <= self.learning_rate:
            raise ValueError("learning-rate bounds are invalid")
        if min(self.learning_rate, self.grad_clip_norm, self.maximum_runtime_hours) <= 0.0:
            raise ValueError("optimizer/runtime controls must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")


@dataclass
class CausalScaleEAPTrainingResult:
    model: CausalScaleTTC
    history: list[dict[str, Any]]
    best_epoch: int
    best_selection: dict[str, float]
    best_validation: dict[str, Any]
    elapsed_seconds: float


def _autocast(device: torch.device, precision: str) -> torch.autocast:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def _targets(
    batch: ObjectEventV4Batch,
    *,
    mask_t0_as_proxy: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create disclosed weak targets; none are returned as model inputs."""

    endpoint_valid = torch.ones(
        batch.boxes_xyxy.shape[:2], device=batch.boxes_xyxy.device, dtype=torch.bool
    )
    if mask_t0_as_proxy:
        endpoint_valid[:, 0] = False
    masks, mask_valid = weak_box_masks(
        batch.boxes_xyxy,
        height=int(batch.events.shape[-2]),
        width=int(batch.events.shape[-1]),
        endpoint_valid=endpoint_valid,
    )
    delta = batch.delta_t_s[:, None].expand(-1, batch.events.shape[1] - 1)
    target_valid = torch.isfinite(batch.target_ttc_s) & (batch.target_ttc_s != 0.0)
    return delta, masks, mask_valid & target_valid[:, None]


def _loss(
    model: CausalScaleTTC,
    batch: ObjectEventV4Batch,
    loss_config: CausalScaleTTCLossConfig,
    *,
    mask_t0_as_proxy: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Any]:
    delta, masks, mask_valid = _targets(batch, mask_t0_as_proxy=mask_t0_as_proxy)
    output = model(batch.events, delta)
    result = causal_scale_ttc_loss(
        output,
        target_ttc_seconds=batch.target_ttc_s,
        delta_t_s=delta,
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_valid=torch.ones_like(batch.target_ttc_s, dtype=torch.bool),
        target_masks=masks,
        mask_valid=mask_valid,
        config=loss_config,
    )
    return result.total, result.components, output


def _selection(metrics: dict[str, Any]) -> dict[str, float]:
    macro = float(metrics["sequence_macro"]["sequence_macro_paper_MiD_overall"])
    failure = float(metrics["signed"]["failure_rate_pct"])
    if not math.isfinite(macro) or not math.isfinite(failure):
        raise FloatingPointError("validation selection metrics are non-finite")
    return {"sequence_macro_MiD": macro, "failure_rate_pct": failure}


def _is_better(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    if incumbent is None:
        return True
    return (candidate["sequence_macro_MiD"], candidate["failure_rate_pct"]) < (
        incumbent["sequence_macro_MiD"],
        incumbent["failure_rate_pct"],
    )


def _loader(
    dataset: Dataset[dict[str, Any]],
    config: CausalScaleEAPTrainingConfig,
    *,
    train: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[ObjectEventV4Batch]:
    if train and generator is None:
        generator = torch.Generator().manual_seed(config.seed)
    return cast(
        DataLoader[ObjectEventV4Batch],
        DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=train,
            generator=generator if train else None,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
            collate_fn=collate_object_event_v4,
        ),
    )


def train_one_real_epoch(
    model: CausalScaleTTC,
    loader: DataLoader[ObjectEventV4Batch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: CausalScaleEAPTrainingConfig,
    loss_config: CausalScaleTTCLossConfig,
) -> dict[str, float]:
    """Train one example-weighted epoch with bounded gradient accumulation."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {}
    examples = 0
    steps = len(loader)
    for step, host_batch in enumerate(loader, start=1):
        batch = host_batch.to(device)
        with _autocast(device, config.precision):
            total, components, _ = _loss(
                model, batch, loss_config, mask_t0_as_proxy=config.mask_t0_as_proxy
            )
            scaled = total / config.gradient_accumulation_steps
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("non-finite real causal-scale loss")
        scaled.backward()
        if step % config.gradient_accumulation_steps == 0 or step == steps:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        count = int(batch.events.shape[0])
        examples += count
        totals["total"] = (
            totals.get("total", 0.0) + float(total.detach().cpu()) * count
        )
        for key, value in components.items():
            totals[key] = (
                totals.get(key, 0.0) + float(value.detach().cpu()) * count
            )
    if examples == 0:
        raise ValueError("training loader is empty")
    return {key: value / examples for key, value in totals.items()}


@torch.inference_mode()
def evaluate_real_causal_scale(
    model: CausalScaleTTC,
    loader: DataLoader[ObjectEventV4Batch],
    device: torch.device,
    config: CausalScaleEAPTrainingConfig,
    loss_config: CausalScaleTTCLossConfig,
) -> dict[str, Any]:
    """Evaluate the public validation split with official signed metrics."""

    model.eval()
    truth: list[torch.Tensor] = []
    prediction: list[torch.Tensor] = []
    known: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    ratio_targets: list[torch.Tensor] = []
    weak_iou: list[torch.Tensor] = []
    sequences: list[str] = []
    tokens: list[str] = []
    losses: list[tuple[float, int]] = []
    for host_batch in loader:
        batch = host_batch.to(device)
        with _autocast(device, config.precision):
            total, _, output = _loss(
                model, batch, loss_config, mask_t0_as_proxy=config.mask_t0_as_proxy
            )
        delta, masks, mask_valid = _targets(
            batch, mask_t0_as_proxy=config.mask_t0_as_proxy
        )
        target_ratio, valid_ratio = target_log_ratio_from_ttc(
            batch.target_ttc_s, delta[:, -1]
        )
        predicted_masks = torch.sigmoid(output.foreground_logits) >= 0.5
        selected_masks = mask_valid
        intersection = (predicted_masks & masks.bool()).sum(dim=(-3, -2, -1)).float()
        union = (predicted_masks | masks.bool()).sum(dim=(-3, -2, -1)).float().clamp_min(1)
        weak_iou.append((intersection[selected_masks] / union[selected_masks]).cpu())
        count = int(batch.events.shape[0])
        losses.append((float(total.detach().cpu()), count))
        truth.append(batch.target_ttc_s.cpu())
        current_prediction = output.ttc_mean_seconds.float()
        current_prediction = torch.where(
            output.known_mask, current_prediction, torch.full_like(current_prediction, float("nan"))
        )
        prediction.append(current_prediction.cpu())
        known.append(output.known_mask.cpu())
        ratios.append(output.log_height_ratio[:, -1][valid_ratio].float().cpu())
        ratio_targets.append(target_ratio[valid_ratio].float().cpu())
        sequences.extend(batch.sequence_ids)
        tokens.extend(batch.sample_tokens)
    target_np = torch.cat(truth).numpy().astype(np.float64)
    prediction_np = torch.cat(prediction).numpy().astype(np.float64)
    ratio = torch.cat(ratios)
    ratio_target = torch.cat(ratio_targets)
    pearson = (
        float(torch.corrcoef(torch.stack((ratio_target, ratio)))[0, 1])
        if (
            ratio.numel() > 1
            and ratio.std(unbiased=False) > 0
            and ratio_target.std(unbiased=False) > 0
        )
        else float("nan")
    )
    signed = signed_garl_metrics(target_np, prediction_np)
    sequence_macro = sequence_macro_signed_metrics(
        target_np, prediction_np, np.asarray(sequences)
    )
    return {
        "num_samples": int(target_np.size),
        "loss": sum(value * count for value, count in losses) / sum(count for _, count in losses),
        "signed": signed,
        "sequence_macro": sequence_macro,
        "known_coverage": float(torch.cat(known).float().mean()),
        "weak_bbox_iou": float(torch.cat(weak_iou).mean()),
        "log_ratio_mae": float((ratio - ratio_target).abs().mean()),
        "log_ratio_pearson": pearson,
        "sample_tokens": tokens,
        "target_ttc_s": target_np.tolist(),
        "prediction_ttc_s": prediction_np.tolist(),
        "sequence_ids": sequences,
    }


def train_real_causal_scale(
    model_config: CausalScaleTTCConfig,
    training_config: CausalScaleEAPTrainingConfig,
    loss_config: CausalScaleTTCLossConfig,
    train_dataset: Dataset[dict[str, Any]],
    validation_dataset: Dataset[dict[str, Any]],
    device: torch.device,
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> CausalScaleEAPTrainingResult:
    """Train under a hard wall-clock guard and select on validation only."""

    if not isinstance(train_dataset, Sized) or not isinstance(validation_dataset, Sized):
        raise TypeError("real causal-scale datasets must expose length")
    seed_everything(training_config.seed, deterministic=True)
    generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = _loader(
        train_dataset, training_config, train=True, generator=generator
    )
    validation_loader = _loader(validation_dataset, training_config, train=False)
    model = CausalScaleTTC(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs,
        eta_min=training_config.minimum_learning_rate,
    )
    foreground_only = replace(
        loss_config,
        log_ratio_nll_weight=0.0,
        log_ratio_huber_weight=0.0,
        log_ratio_tail_weight=0.0,
        risk_weight=0.0,
        auxiliary_inverse_ttc_weight=0.0,
        residual_regularization_weight=0.0,
        temporal_consistency_weight=0.0,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_selection: dict[str, float] | None = None
    best_validation: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    prior_elapsed = 0.0
    start_epoch = 1
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "last.pt" if checkpoint_dir is not None else None
    best_path = checkpoint_dir / "best.pt" if checkpoint_dir is not None else None
    if resume:
        if last_path is None or not last_path.is_file():
            raise FileNotFoundError("resume requested but last.pt is unavailable")
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if saved.get("artifact_type") != "causal_scale_eap_resume_state_v1":
            raise ValueError("last.pt has an incompatible artifact type")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        best_state = saved.get("best_model_state_dict")
        best_selection = saved.get("best_selection")
        best_validation = saved.get("best_validation")
        best_epoch = int(saved.get("best_epoch", 0))
        stale = int(saved.get("stale", 0))
        history = list(saved.get("history", []))
        prior_elapsed = float(saved.get("elapsed_seconds", 0.0))
        start_epoch = int(saved["epoch"]) + 1
        generator.set_state(saved["loader_generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        random.setstate(saved["python_random_state"])
        np.random.set_state(saved["numpy_random_state"])
        if device.type == "cuda" and saved.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
    started = time.perf_counter()
    remaining_seconds = max(
        0.0, training_config.maximum_runtime_hours * 3600.0 - prior_elapsed
    )
    deadline = started + remaining_seconds

    def save_state(path: Path, epoch: int, elapsed: float) -> None:
        payload = {
            "artifact_type": "causal_scale_eap_resume_state_v1",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_model_state_dict": best_state,
            "best_selection": best_selection,
            "best_validation": best_validation,
            "best_epoch": best_epoch,
            "stale": stale,
            "history": history,
            "elapsed_seconds": elapsed,
            "loader_generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
        }
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    last_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, training_config.epochs + 1):
        if time.perf_counter() >= deadline:
            break
        epoch_started = time.perf_counter()
        active_loss = (
            foreground_only
            if epoch <= training_config.foreground_warmup_epochs
            else loss_config
        )
        train_metrics = train_one_real_epoch(
            model, train_loader, optimizer, device, training_config, active_loss
        )
        validation = evaluate_real_causal_scale(
            model, validation_loader, device, training_config, loss_config
        )
        selection = _selection(validation)
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "foreground_warmup": epoch <= training_config.foreground_warmup_epochs,
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "selection": selection,
            "train": train_metrics,
            "validation": {
                key: value
                for key, value in validation.items()
                if key not in {"sample_tokens", "target_ttc_s", "prediction_ttc_s", "sequence_ids"}
            },
        }
        history.append(record)
        eligible = epoch > training_config.foreground_warmup_epochs
        if eligible and _is_better(selection, best_selection):
            best_state = copy.deepcopy(model.state_dict())
            best_selection = selection
            best_validation = validation
            best_epoch = epoch
            stale = 0
        elif eligible:
            stale += 1
        scheduler.step()
        last_completed_epoch = epoch
        elapsed = prior_elapsed + time.perf_counter() - started
        if last_path is not None:
            save_state(last_path, epoch, elapsed)
        if best_path is not None and best_epoch == epoch:
            save_state(best_path, epoch, elapsed)
        if (
            epoch >= training_config.minimum_epochs
            and stale >= training_config.early_stopping_patience
        ):
            break
    if best_state is None or best_selection is None or best_validation is None:
        raise RuntimeError("real causal-scale training produced no selectable checkpoint")
    model.load_state_dict(best_state)
    elapsed_total = prior_elapsed + time.perf_counter() - started
    if last_path is not None and last_completed_epoch < start_epoch:
        raise RuntimeError("runtime budget expired before a resumable epoch completed")
    return CausalScaleEAPTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_selection=best_selection,
        best_validation=best_validation,
        elapsed_seconds=elapsed_total,
    )


def checkpoint_payload(
    result: CausalScaleEAPTrainingResult,
    training_config: CausalScaleEAPTrainingConfig,
    loss_config: CausalScaleTTCLossConfig,
) -> dict[str, Any]:
    return {
        "artifact_type": "causal_scale_eap_public_validation_checkpoint_v1",
        "model_config": result.model.checkpoint_config(),
        "training_config": asdict(training_config),
        "loss_config": asdict(loss_config),
        "best_epoch": result.best_epoch,
        "best_selection": result.best_selection,
        "model_state_dict": result.model.state_dict(),
    }


__all__ = [
    "CausalScaleEAPTrainingConfig",
    "CausalScaleEAPTrainingResult",
    "checkpoint_payload",
    "evaluate_real_causal_scale",
    "train_one_real_epoch",
    "train_real_causal_scale",
]
