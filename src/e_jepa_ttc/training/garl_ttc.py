"""EvTTC trainer for the matched local Garl-TTC replica."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset, ShardLocalSampler
from e_jepa_ttc.evaluation.object_ttc import (
    garl_ttc_metrics,
    grouped_ttc_selection_components,
    object_ttc_metrics,
)
from e_jepa_ttc.models.garl_ttc_replica import GarlTTCConfig, GarlTTCOutput, GarlTTCReplica
from e_jepa_ttc.reproducibility import cuda_device_name, resolve_device
from e_jepa_ttc.training.object_geo_trainer import (
    OGETrainerConfig,
    _atomic_save,
    _autocast,
    _git_commit,
    _hash_file,
    _indices,
    _loader,
    _selection_hash,
    _set_seed,
    _shutdown_loader,
    _source_tree_hash,
    _state_dict_hash,
    _tensor,
)
from e_jepa_ttc.utils.io import write_structured


def _garl_forward(
    model: GarlTTCReplica,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> GarlTTCOutput:
    if "garl_delta_t_s" not in batch:
        raise ValueError("Source-audited Garl cache requires garl_delta_t_s.")
    elapsed_s = (
        _tensor(
            batch,
            "garl_delta_t_s",
            device,
            dtype=torch.float32,
        )
        .reshape(-1)
        .clamp_min(1e-6)
    )
    rgb_pair = (
        _tensor(batch, "garl_rgb_pair", device, dtype=torch.float32)
        if "garl_rgb_pair" in batch
        else None
    )
    if rgb_pair is not None and rgb_pair.max() > 1.5:
        rgb_pair = rgb_pair / 255.0
    if "garl_event_roi" not in batch:
        raise ValueError("Paper-aligned Garl requires garl_event_roi from cache format v2.")
    return model(
        _tensor(batch, "garl_event_roi", device, dtype=torch.float32),
        elapsed_s,
        rgb_pair=rgb_pair,
    )


def _loss(
    output: GarlTTCOutput,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    *,
    epoch: int,
) -> dict[str, torch.Tensor]:
    truth = _tensor(batch, "ttc_s", device, dtype=torch.float32).reshape(-1)
    zero = truth.new_zeros(())
    losses: dict[str, torch.Tensor] = {
        "ttc": zero,
        "visible_height": zero,
        "mid": zero,
        "mask_t1_focal": zero,
        "mask_t2_focal": zero,
    }
    if output.predicted_heights is None:
        losses["ttc"] = functional.smooth_l1_loss(output.ttc_seconds, truth)
    else:
        if "garl_visible_heights_px" not in batch:
            raise ValueError("Garl LHR requires the declared EvTTC visible-height adapter.")
        target_heights = _tensor(
            batch,
            "garl_visible_heights_px",
            device,
            dtype=torch.float32,
        )
        losses["visible_height"] = functional.smooth_l1_loss(
            output.predicted_heights,
            target_heights,
        )
        if epoch > 5:
            delta_t = _tensor(
                batch,
                "garl_delta_t_s",
                device,
                dtype=torch.float32,
            ).reshape(-1)
            target_ratio = 1.0 - delta_t / truth
            valid = (
                (output.predicted_height_ratio > 0)
                & (target_ratio > 0)
                & torch.isfinite(output.predicted_height_ratio)
            )
            if valid.any():
                losses["mid"] = (
                    output.predicted_height_ratio[valid].log() - target_ratio[valid].log()
                ).abs().mean() * 1e4
    if output.foreground_logits is not None:
        if "garl_foreground_mask" not in batch:
            raise ValueError("Foreground Garl requires a cached polygon/SAM target.")
        target = _tensor(
            batch,
            "garl_foreground_mask",
            device,
        ).long()
        if target.ndim != 4 or target.shape[1] != 2:
            raise ValueError("Garl foreground target must contain both endpoint masks.")
        first_target = functional.one_hot(target[:, 0], num_classes=2).permute(0, 3, 1, 2)
        last_target = functional.one_hot(target[:, 1], num_classes=2).permute(0, 3, 1, 2)
        losses["mask_t1_focal"] = 500.0 * _sigmoid_focal_loss(
            output.foreground_logits[:, :2],
            first_target.to(output.foreground_logits.dtype),
        )
        losses["mask_t2_focal"] = 500.0 * _sigmoid_focal_loss(
            output.foreground_logits[:, 2:],
            last_target.to(output.foreground_logits.dtype),
        )
    losses["total"] = sum(losses.values())
    return losses


def _sigmoid_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    cross_entropy = functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    probability_target = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_target = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_target * cross_entropy * (1.0 - probability_target).pow(gamma)).mean()


def _evaluate_garl(
    model: GarlTTCReplica,
    loader: DataLoader[dict[str, torch.Tensor | list[str]]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    true: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    mask_iou: list[np.ndarray] = []
    delta_t: list[np.ndarray] = []
    sequences: list[str] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            output = _garl_forward(model, batch, device)
            true.append(_tensor(batch, "ttc_s", device).reshape(-1).cpu().numpy())
            predicted.append(output.ttc_seconds.cpu().numpy())
            delta_t.append(
                _tensor(
                    batch,
                    "garl_delta_t_s",
                    device,
                    dtype=torch.float32,
                )
                .reshape(-1)
                .cpu()
                .numpy()
            )
            if output.foreground_logits is not None:
                target_mask = (
                    _tensor(
                        batch,
                        "garl_foreground_mask",
                        device,
                        dtype=torch.float32,
                    )
                    >= 0.5
                )
                predicted_mask = torch.stack(
                    (
                        output.foreground_logits[:, :2].argmax(dim=1),
                        output.foreground_logits[:, 2:].argmax(dim=1),
                    ),
                    dim=1,
                ).bool()
                intersection = (target_mask & predicted_mask).sum(dim=(-1, -2)).float()
                union = (target_mask | predicted_mask).sum(dim=(-1, -2)).float()
                mask_iou.append((intersection / union.clamp_min(1.0)).cpu().numpy())
            values = batch["sequence_id"]
            if not isinstance(values, list):
                raise TypeError("sequence_id must collate to a list.")
            sequences.extend(str(value) for value in values)
    arrays = {
        "ttc_true": np.concatenate(true),
        "ttc_pred": np.concatenate(predicted),
        "sequence_id": np.asarray(sequences),
        "delta_t_s": np.concatenate(delta_t),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - started
    metrics = object_ttc_metrics(arrays["ttc_true"], arrays["ttc_pred"])
    representative_delta = float(np.median(arrays["delta_t_s"]))
    metrics["garl_source_metrics"] = garl_ttc_metrics(
        arrays["ttc_true"],
        arrays["ttc_pred"],
        delta_t_s=representative_delta,
    )
    metrics["delta_t_s_median"] = representative_delta
    metrics["delta_t_s_min"] = float(np.min(arrays["delta_t_s"]))
    metrics["delta_t_s_max"] = float(np.max(arrays["delta_t_s"]))
    metrics.update(
        grouped_ttc_selection_components(
            arrays["ttc_true"],
            arrays["ttc_pred"],
            arrays["sequence_id"],
        )
    )
    per_sequence = [
        float(
            np.mean(
                np.abs(
                    arrays["ttc_true"][arrays["sequence_id"] == sequence]
                    - arrays["ttc_pred"][arrays["sequence_id"] == sequence]
                )
            )
        )
        for sequence in np.unique(arrays["sequence_id"])
    ]
    metrics["sequence_macro_mae_s"] = float(np.mean(per_sequence))
    metrics["worst_sequence_mae_s"] = float(np.max(per_sequence))
    metrics["evaluation_seconds"] = evaluation_seconds
    metrics["milliseconds_per_window"] = (
        1000.0 * evaluation_seconds / max(arrays["ttc_true"].shape[0], 1)
    )
    if mask_iou:
        iou = np.concatenate(mask_iou)
        arrays["mask_iou"] = iou
        metrics["mask_iou_mean"] = float(iou.mean())
    return metrics, arrays


def _load_local_lhr_branches(
    model: GarlTTCReplica,
    *,
    rgb_branch_checkpoint: str | Path | None,
    event_branch_checkpoint: str | Path | None,
    device: torch.device,
) -> dict[str, Any]:
    """Initialize late fusion from locally trained unimodal LHR branches."""

    requires_branches = (
        model.config.modality == "rgbe"
        and model.config.fusion == "late"
        and model.config.objective == "height_ratio"
    )
    paths = {
        "rgb": Path(rgb_branch_checkpoint) if rgb_branch_checkpoint is not None else None,
        "event": (Path(event_branch_checkpoint) if event_branch_checkpoint is not None else None),
    }
    if requires_branches and any(path is None for path in paths.values()):
        raise ValueError(
            "Source-audited late-fusion LHR requires local G3 RGB and G4 event checkpoints."
        )
    result: dict[str, Any] = {
        "required_by_source_protocol": requires_branches,
        "rgb": None,
        "event": None,
    }
    for branch, path in paths.items():
        if path is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Local Garl {branch} branch checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"Garl branch checkpoint has no model_state_dict: {path}")
        prefix = f"{branch}_encoder."
        branch_state = {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        target = getattr(model, f"{branch}_encoder")
        if target is None or not branch_state:
            raise ValueError(f"Checkpoint {path} does not contain a usable {branch} encoder.")
        target.load_state_dict(branch_state, strict=True)
        result[branch] = {
            "path": path.as_posix(),
            "sha256": _hash_file(path),
            "source_epoch": checkpoint.get("epoch"),
            "loaded_prefix": prefix,
        }
    return result


def train_garl_ttc(
    *,
    cache_manifest_path: str | Path,
    output_dir: str | Path,
    model_config: GarlTTCConfig,
    trainer_config: OGETrainerConfig | None = None,
    device_name: str = "auto",
    rgb_branch_checkpoint: str | Path | None = None,
    event_branch_checkpoint: str | Path | None = None,
    resume: bool = False,
    dry_run_fingerprint: bool = False,
) -> dict[str, Any] | str:
    """Train a direct/LHR modality arm with identical validation selection."""

    trainer = trainer_config or OGETrainerConfig()
    assert_no_sealed_benchmark_paths((cache_manifest_path, output_dir))
    _set_seed(trainer.seed)
    device = resolve_device(device_name)
    train_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("train",))
    validation_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("validation",))
    train_indices = _indices(len(train_dataset), trainer.max_train_samples)
    validation_indices = _indices(len(validation_dataset), trainer.max_validation_samples)
    train_loader = _loader(
        train_dataset,
        train_indices,
        batch_size=trainer.batch_size,
        num_workers=trainer.num_workers,
        device=device,
        train=True,
        seed=trainer.seed,
    )
    validation_loader = _loader(
        validation_dataset,
        validation_indices,
        batch_size=trainer.batch_size * 2,
        num_workers=max(0, trainer.num_workers // 2),
        device=device,
        train=False,
        seed=trainer.seed,
    )
    model = GarlTTCReplica(model_config).to(device)
    branch_initialization = _load_local_lhr_branches(
        model,
        rgb_branch_checkpoint=rgb_branch_checkpoint,
        event_branch_checkpoint=event_branch_checkpoint,
        device=device,
    )
    initial_model_sha256 = _state_dict_hash(model)
    sample_selection_sha256 = _selection_hash(train_indices, validation_indices)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer.learning_rate,
        weight_decay=trainer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=(10, 20, 30, 40),
        gamma=0.5,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and trainer.precision == "fp16"
        else None
    )
    fingerprint_payload = {
        "git_commit": _git_commit(),
        "source_tree_sha256": _source_tree_hash(),
        "device": str(device),
        "gpu_name": cuda_device_name(device),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "cache_manifest_sha256": _hash_file(cache_manifest_path),
        "model_config": asdict(model_config),
        "trainer_config": asdict(trainer),
        "branch_initialization": branch_initialization,
        "initial_model_sha256": initial_model_sha256,
        "sample_selection_sha256": sample_selection_sha256,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if dry_run_fingerprint:
        train_dataset.close()
        validation_dataset.close()
        return fingerprint
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    last_path = output / "last.pt"
    weights_only_path = output / "weights_only.pt"
    resume_path = output / "resume.pt"
    history_path = output / "history.jsonl"
    best_value = float("inf")
    best_epoch = -1
    no_improvement = 0
    start_epoch = 1
    if resume:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["run_fingerprint"] != fingerprint:
            raise ValueError("Resume fingerprint differs from this Garl arm.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_value = float(state["best_value"])
        best_epoch = int(state["best_epoch"])
        no_improvement = int(state["no_improvement"])
    elif resume_path.exists():
        raise FileExistsError(f"{resume_path} exists; pass resume=True or use a new directory.")

    def checkpoint(epoch: int, role: str) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "trainer_config": asdict(trainer),
            "epoch": epoch,
            "role": role,
            "selected_by": "validation_sequence_macro_selection_score",
            "run_fingerprint": fingerprint,
        }

    completed = start_epoch - 1
    stopped_early = False
    started = time.perf_counter()
    with history_path.open("a" if resume else "w", encoding="utf-8") as history_file:
        for epoch in range(start_epoch, trainer.epochs + 1):
            model.train()
            if isinstance(train_loader.sampler, ShardLocalSampler):
                train_loader.sampler.set_epoch(epoch - 1)
            optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            for step, batch in enumerate(train_loader, start=1):
                with _autocast(device, trainer.precision):
                    losses = _loss(
                        _garl_forward(model, batch, device),
                        batch,
                        device,
                        epoch=epoch,
                    )
                    scaled_loss = losses["total"] / trainer.gradient_accumulation
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                if step % trainer.gradient_accumulation == 0 or step == len(train_loader):
                    if scaler is None:
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    else:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                for name, loss_value in losses.items():
                    totals[name] = totals.get(name, 0.0) + float(loss_value.detach())
            scheduler.step()
            validation, _ = _evaluate_garl(model, validation_loader, device)
            value = float(validation["sequence_macro_selection_score"])
            history_file.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train": {
                            name: total / max(len(train_loader), 1)
                            for name, total in totals.items()
                        },
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "validation": validation,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            history_file.flush()
            threshold = (
                best_value * (1.0 - trainer.early_stopping_min_delta_relative)
                if np.isfinite(best_value)
                else float("inf")
            )
            if value < threshold:
                best_value = value
                best_epoch = epoch
                no_improvement = 0
                _atomic_save(checkpoint(epoch, "best"), best_path)
            else:
                no_improvement += 1
            completed = epoch
            resume_state = checkpoint(epoch, "resume")
            resume_state.update(
                {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_value": best_value,
                    "best_epoch": best_epoch,
                    "no_improvement": no_improvement,
                }
            )
            _atomic_save(resume_state, resume_path)
            if (
                trainer.early_stopping_patience > 0
                and epoch >= trainer.early_stopping_min_epochs
                and no_improvement >= trainer.early_stopping_patience
            ):
                stopped_early = True
                break
    _atomic_save(checkpoint(completed, "last"), last_path)
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    _atomic_save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "epoch": best_epoch,
            "role": "weights_only",
            "selected_by": "validation_sequence_macro_selection_score",
            "run_fingerprint": fingerprint,
        },
        weights_only_path,
    )
    resume_path.unlink(missing_ok=True)
    validation, predictions = _evaluate_garl(model, validation_loader, device)
    np.savez_compressed(output / "validation_predictions.npz", **predictions)
    summary: dict[str, Any] = {
        "architecture": asdict(model_config),
        "trainer": asdict(trainer),
        "run_fingerprint": fingerprint,
        "git_commit": _git_commit(),
        "cache_manifest": Path(cache_manifest_path).as_posix(),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "sample_selection_sha256": sample_selection_sha256,
        "initial_model_sha256": initial_model_sha256,
        "branch_initialization": branch_initialization,
        "optimizer": "Adam",
        "scheduler": {
            "type": "MultiStepLR",
            "milestones": [10, 20, 30, 40],
            "gamma": 0.5,
        },
        "effective_batch_size": trainer.batch_size * trainer.gradient_accumulation,
        "train_batches_per_epoch": len(train_loader),
        "optimizer_steps_per_epoch": math.ceil(len(train_loader) / trainer.gradient_accumulation),
        "completed_optimizer_steps": (
            math.ceil(len(train_loader) / trainer.gradient_accumulation) * completed
        ),
        "maximum_optimizer_steps": (
            math.ceil(len(train_loader) / trainer.gradient_accumulation) * trainer.epochs
        ),
        "epochs_completed": completed,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
        "weights_only_checkpoint": weights_only_path.as_posix(),
        "resume_checkpoint": None,
        "checkpoint_policy": ["best", "last", "weights_only"],
        "validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "benchmark10_opened": False,
    }
    write_structured(output / "summary.json", summary)
    _shutdown_loader(train_loader)
    _shutdown_loader(validation_loader)
    train_dataset.close()
    validation_dataset.close()
    return summary


__all__ = ["train_garl_ttc"]
