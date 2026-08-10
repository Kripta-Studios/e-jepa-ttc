"""Bounded real-data training for the causal event foreground-scale arm."""

from __future__ import annotations

import copy
import math
import os
import random
import time
from collections.abc import Iterator, Sized
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from e_jepa_ttc.data.object_event_v4 import (
    BoxGeometryTargets,
    ObjectEventV4Batch,
    box_geometry_targets,
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
    foreground_supervision: Literal[
        "weak_box", "bbox_geometry", "bbox_geometry_sam_teacher"
    ] = "weak_box"
    teacher_cache_artifact_sha256: str | None = None

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
        allowed = {"weak_box", "bbox_geometry", "bbox_geometry_sam_teacher"}
        if self.foreground_supervision not in allowed:
            raise ValueError(f"foreground_supervision must be one of {sorted(allowed)}")
        uses_teacher = self.foreground_supervision == "bbox_geometry_sam_teacher"
        if uses_teacher != bool(self.teacher_cache_artifact_sha256):
            raise ValueError("SAM supervision and teacher cache identity must be declared together")


@dataclass
class CausalScaleEAPTrainingResult:
    model: CausalScaleTTC
    history: list[dict[str, Any]]
    best_epoch: int
    best_selection: dict[str, float]
    best_validation: dict[str, Any]
    elapsed_seconds: float


@dataclass(frozen=True)
class CausalScaleEAPTargets:
    """Training-only targets kept outside the model input contract."""

    delta_t_s: torch.Tensor
    target_valid: torch.Tensor
    target_masks: torch.Tensor | None
    mask_valid: torch.Tensor | None
    geometry: BoxGeometryTargets | None


def _autocast(device: torch.device, precision: str) -> torch.autocast:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def _targets(
    batch: ObjectEventV4Batch,
    *,
    mask_t0_as_proxy: bool,
    foreground_supervision: Literal[
        "weak_box", "bbox_geometry", "bbox_geometry_sam_teacher"
    ],
) -> CausalScaleEAPTargets:
    """Create disclosed weak targets; none are returned as model inputs."""

    endpoint_valid = torch.ones(
        batch.boxes_xyxy.shape[:2], device=batch.boxes_xyxy.device, dtype=torch.bool
    )
    if mask_t0_as_proxy:
        endpoint_valid[:, 0] = False
    delta = batch.delta_t_s[:, None].expand(-1, batch.events.shape[1] - 1)
    target_valid = torch.isfinite(batch.target_ttc_s) & (batch.target_ttc_s != 0.0)
    endpoint_valid = endpoint_valid & target_valid[:, None]
    height = int(batch.events.shape[-2])
    width = int(batch.events.shape[-1])
    if foreground_supervision == "weak_box":
        masks, mask_valid = weak_box_masks(
            batch.boxes_xyxy,
            height=height,
            width=width,
            endpoint_valid=endpoint_valid,
        )
        return CausalScaleEAPTargets(
            delta_t_s=delta,
            target_valid=target_valid,
            target_masks=masks,
            mask_valid=mask_valid,
            geometry=None,
        )
    if foreground_supervision in {"bbox_geometry", "bbox_geometry_sam_teacher"}:
        geometry = box_geometry_targets(
            batch.boxes_xyxy,
            height=height,
            width=width,
            endpoint_valid=endpoint_valid,
        )
        target_masks = None
        mask_valid = None
        if foreground_supervision == "bbox_geometry_sam_teacher":
            if (batch.sam_teacher_masks is None) != (batch.sam_teacher_mask_valid is None):
                raise ValueError("SAM teacher masks and validity must be provided together")
            if batch.sam_teacher_masks is not None and batch.sam_teacher_mask_valid is not None:
                zeros = torch.zeros_like(batch.sam_teacher_masks[:, :1])
                target_masks = torch.cat((zeros, batch.sam_teacher_masks), dim=1)
                invalid_t0 = torch.zeros_like(batch.sam_teacher_mask_valid[:, :1])
                mask_valid = torch.cat((invalid_t0, batch.sam_teacher_mask_valid), dim=1)
        return CausalScaleEAPTargets(
            delta_t_s=delta,
            target_valid=target_valid,
            target_masks=target_masks,
            mask_valid=mask_valid,
            geometry=geometry,
        )
    raise ValueError(f"unsupported foreground supervision: {foreground_supervision}")


def _loss(
    model: CausalScaleTTC,
    batch: ObjectEventV4Batch,
    loss_config: CausalScaleTTCLossConfig,
    *,
    mask_t0_as_proxy: bool,
    foreground_supervision: Literal[
        "weak_box", "bbox_geometry", "bbox_geometry_sam_teacher"
    ],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Any]:
    targets = _targets(
        batch,
        mask_t0_as_proxy=mask_t0_as_proxy,
        foreground_supervision=foreground_supervision,
    )
    output = model(batch.events, targets.delta_t_s)
    result = causal_scale_ttc_loss(
        output,
        target_ttc_seconds=batch.target_ttc_s,
        delta_t_s=targets.delta_t_s,
        risk_thresholds_s=model.config.risk_thresholds_s,
        target_valid=targets.target_valid,
        target_masks=targets.target_masks,
        mask_valid=targets.mask_valid,
        target_geometry=targets.geometry,
        config=loss_config,
    )
    return result.total, result.components, output


def _foreground_only_loss_config(
    loss_config: CausalScaleTTCLossConfig,
) -> CausalScaleTTCLossConfig:
    """Disable temporal/TTC objectives during the endpoint-geometry warm-up."""

    return replace(
        loss_config,
        log_ratio_nll_weight=0.0,
        log_ratio_huber_weight=0.0,
        log_ratio_tail_weight=0.0,
        risk_weight=0.0,
        auxiliary_inverse_ttc_weight=0.0,
        residual_regularization_weight=0.0,
        temporal_consistency_weight=0.0,
        foreground_pair_ratio_weight=0.0,
    )


def _selection(metrics: dict[str, Any]) -> dict[str, float]:
    macro = float(metrics["sequence_macro"]["sequence_macro_paper_MiD_overall"])
    failure = float(metrics["signed"]["failure_rate_pct"])
    per_sequence = metrics["sequence_macro"].get("per_sequence")
    if not isinstance(per_sequence, dict) or not per_sequence:
        raise ValueError("validation selection requires per-sequence metrics")
    finite_sequence_count = sum(
        math.isfinite(float(value.get("paper_MiD_overall", float("nan"))))
        for value in per_sequence.values()
        if isinstance(value, dict)
    )
    complete_sequence_coverage = float(finite_sequence_count == len(per_sequence))
    if not math.isfinite(macro) or not math.isfinite(failure):
        raise FloatingPointError("validation selection metrics are non-finite")
    return {
        "sequence_macro_MiD": macro,
        "failure_rate_pct": failure,
        "finite_sequence_count": float(finite_sequence_count),
        "sequence_count": float(len(per_sequence)),
        "complete_sequence_coverage": complete_sequence_coverage,
    }


def _is_better(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    if candidate.get("complete_sequence_coverage", 1.0) != 1.0:
        return False
    if incumbent is None:
        return True
    return (candidate["sequence_macro_MiD"], candidate["failure_rate_pct"]) < (
        incumbent["sequence_macro_MiD"],
        incumbent["failure_rate_pct"],
    )


def _relationship(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    if target.shape != prediction.shape:
        raise ValueError("diagnostic target and prediction shapes differ")
    valid = np.isfinite(target) & np.isfinite(prediction)
    x = target[valid].astype(np.float64, copy=False)
    y = prediction[valid].astype(np.float64, copy=False)
    target_std = float(np.std(x)) if x.size else float("nan")
    prediction_std = float(np.std(y)) if y.size else float("nan")
    pearson = (
        float(np.corrcoef(x, y)[0, 1])
        if x.size > 1 and target_std > 0.0 and prediction_std > 0.0
        else float("nan")
    )
    centered = x - x.mean() if x.size else x
    denominator = float(np.dot(centered, centered))
    slope = (
        float(np.dot(centered, y - y.mean()) / denominator)
        if x.size > 1 and denominator > 0.0
        else float("nan")
    )
    sign_valid = (x != 0.0) & (y != 0.0)
    return {
        "count": int(x.size),
        "pearson": pearson,
        "slope": slope,
        "mae": float(np.mean(np.abs(y - x))) if x.size else float("nan"),
        "sign_accuracy": (
            float(np.mean(np.sign(y[sign_valid]) == np.sign(x[sign_valid])))
            if np.any(sign_valid)
            else float("nan")
        ),
        "target_std": target_std,
        "prediction_std": prediction_std,
        "prediction_target_std_ratio": (
            prediction_std / target_std
            if math.isfinite(target_std) and target_std > 0.0
            else float("nan")
        ),
    }


def _relationship_by_sequence(
    target: np.ndarray,
    prediction: np.ndarray,
    sequences: np.ndarray,
) -> dict[str, Any]:
    if target.shape != prediction.shape or target.shape != sequences.shape:
        raise ValueError("sequence diagnostic arrays must share shape")
    per_sequence = {
        str(sequence): _relationship(
            target[sequences == sequence], prediction[sequences == sequence]
        )
        for sequence in sorted(set(sequences.astype(str).tolist()))
    }
    metric_names = (
        "pearson",
        "slope",
        "mae",
        "sign_accuracy",
        "prediction_target_std_ratio",
    )
    macro: dict[str, float | int] = {"sequence_count": len(per_sequence)}
    for name in metric_names:
        values = np.asarray(
            [float(metrics[name]) for metrics in per_sequence.values()], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        macro[name] = float(np.mean(finite)) if finite.size else float("nan")
        macro[f"{name}_sequence_count"] = int(finite.size)
    return {
        "global": _relationship(target, prediction),
        "macro_by_sequence": macro,
        "per_sequence": per_sequence,
    }


class ShardGroupedRandomSampler(Sampler[int]):
    """Shuffle shards and their records while keeping each shard I/O-contiguous."""

    def __init__(
        self,
        groups: tuple[tuple[int, ...], ...],
        *,
        dataset_size: int,
        generator: torch.Generator,
    ) -> None:
        flattened = [index for group in groups for index in group]
        if not groups or any(not group for group in groups):
            raise ValueError("shard sampler groups must be non-empty")
        if sorted(flattened) != list(range(dataset_size)):
            raise ValueError("shard sampler groups must partition the dataset exactly")
        self.groups = groups
        self.dataset_size = dataset_size
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        shard_order = torch.randperm(
            len(self.groups), generator=self.generator
        ).tolist()
        for shard_index in shard_order:
            group = self.groups[shard_index]
            record_order = torch.randperm(
                len(group), generator=self.generator
            ).tolist()
            for record_index in record_order:
                yield group[record_index]

    def __len__(self) -> int:
        return self.dataset_size


def _loader(
    dataset: Dataset[dict[str, Any]],
    config: CausalScaleEAPTrainingConfig,
    *,
    train: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[ObjectEventV4Batch]:
    if train and generator is None:
        generator = torch.Generator().manual_seed(config.seed)
    sampler: Sampler[int] | None = None
    shuffle = train
    group_provider = getattr(dataset, "shard_index_groups", None)
    if train and generator is not None and callable(group_provider):
        if not isinstance(dataset, Sized):
            raise TypeError("shard-grouped datasets must expose length")
        groups = cast(tuple[tuple[int, ...], ...], group_provider())
        sampler = ShardGroupedRandomSampler(
            groups,
            dataset_size=len(dataset),
            generator=generator,
        )
        shuffle = False
    return cast(
        DataLoader[ObjectEventV4Batch],
        DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
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
                model,
                batch,
                loss_config,
                mask_t0_as_proxy=config.mask_t0_as_proxy,
                foreground_supervision=config.foreground_supervision,
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
    endpoint_geometry: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "log_height_target",
            "log_height_prediction",
            "log_width_target",
            "log_width_prediction",
            "centroid_x_target",
            "centroid_x_prediction",
            "centroid_y_target",
            "centroid_y_prediction",
        )
    }
    pair_geometry: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "height_target",
            "height_prediction",
            "width_target",
            "width_prediction",
            "isotropic_target",
            "isotropic_prediction",
        )
    }
    physical_geometry: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "target",
            "height_prediction",
            "width_prediction",
            "isotropic_prediction",
        )
    }
    endpoint_geometry_sequences: list[str] = []
    pair_geometry_sequences: list[str] = []
    physical_geometry_sequences: list[str] = []
    sequences: list[str] = []
    tokens: list[str] = []
    losses: list[tuple[float, int]] = []
    for host_batch in loader:
        batch = host_batch.to(device)
        with _autocast(device, config.precision):
            total, _, output = _loss(
                model,
                batch,
                loss_config,
                mask_t0_as_proxy=config.mask_t0_as_proxy,
                foreground_supervision=config.foreground_supervision,
            )
        targets = _targets(
            batch,
            mask_t0_as_proxy=config.mask_t0_as_proxy,
            foreground_supervision=config.foreground_supervision,
        )
        target_ratio, valid_ratio = target_log_ratio_from_ttc(
            batch.target_ttc_s, targets.delta_t_s[:, -1]
        )
        if targets.target_masks is not None and targets.mask_valid is not None:
            predicted_masks = torch.sigmoid(output.foreground_logits) >= 0.5
            selected_masks = targets.mask_valid
            intersection = (
                predicted_masks & targets.target_masks.bool()
            ).sum(dim=(-3, -2, -1)).float()
            union = (
                predicted_masks | targets.target_masks.bool()
            ).sum(dim=(-3, -2, -1)).float().clamp_min(1)
            weak_iou.append(
                (intersection[selected_masks] / union[selected_masks]).cpu()
            )
        if targets.geometry is not None:
            geometry = targets.geometry
            endpoint_valid = geometry.valid.bool()
            endpoint_geometry["log_height_target"].append(
                geometry.height_normalized[endpoint_valid].clamp_min(1.0e-6).log().cpu()
            )
            endpoint_geometry["log_height_prediction"].append(
                output.visible_height_normalized[endpoint_valid]
                .clamp_min(1.0e-6)
                .log()
                .float()
                .cpu()
            )
            endpoint_geometry["log_width_target"].append(
                geometry.width_normalized[endpoint_valid].clamp_min(1.0e-6).log().cpu()
            )
            endpoint_geometry["log_width_prediction"].append(
                output.visible_width_normalized[endpoint_valid]
                .clamp_min(1.0e-6)
                .log()
                .float()
                .cpu()
            )
            endpoint_geometry["centroid_x_target"].append(
                geometry.centroid_x_normalized[endpoint_valid].cpu()
            )
            endpoint_geometry["centroid_x_prediction"].append(
                output.diagnostics["foreground_centroid_x"][endpoint_valid].float().cpu()
            )
            endpoint_geometry["centroid_y_target"].append(
                geometry.centroid_y_normalized[endpoint_valid].cpu()
            )
            endpoint_geometry["centroid_y_prediction"].append(
                output.diagnostics["foreground_centroid_y"][endpoint_valid].float().cpu()
            )
            endpoint_valid_cpu = endpoint_valid.cpu()
            for sequence, row_valid in zip(
                batch.sequence_ids, endpoint_valid_cpu, strict=True
            ):
                endpoint_geometry_sequences.extend(
                    [sequence] * int(row_valid.sum().item())
                )

            pair_valid = endpoint_valid[:, -2] & endpoint_valid[:, -1]
            target_height_ratio = (
                geometry.height_normalized[:, -1].clamp_min(1.0e-6).log()
                - geometry.height_normalized[:, -2].clamp_min(1.0e-6).log()
            )
            target_width_ratio = (
                geometry.width_normalized[:, -1].clamp_min(1.0e-6).log()
                - geometry.width_normalized[:, -2].clamp_min(1.0e-6).log()
            )
            predicted_height_ratio = (
                output.visible_height_normalized[:, -1].clamp_min(1.0e-6).log()
                - output.visible_height_normalized[:, -2].clamp_min(1.0e-6).log()
            )
            predicted_width_ratio = (
                output.visible_width_normalized[:, -1].clamp_min(1.0e-6).log()
                - output.visible_width_normalized[:, -2].clamp_min(1.0e-6).log()
            )
            target_isotropic_ratio = 0.5 * (target_height_ratio + target_width_ratio)
            predicted_isotropic_ratio = 0.5 * (
                predicted_height_ratio + predicted_width_ratio
            )
            pair_values = {
                "height_target": target_height_ratio,
                "height_prediction": predicted_height_ratio,
                "width_target": target_width_ratio,
                "width_prediction": predicted_width_ratio,
                "isotropic_target": target_isotropic_ratio,
                "isotropic_prediction": predicted_isotropic_ratio,
            }
            for key, value in pair_values.items():
                pair_geometry[key].append(value[pair_valid].float().cpu())
            for sequence, valid in zip(batch.sequence_ids, pair_valid.cpu(), strict=True):
                if bool(valid):
                    pair_geometry_sequences.append(sequence)

            physical_valid = pair_valid & valid_ratio
            physical_values = {
                "target": target_ratio,
                "height_prediction": predicted_height_ratio,
                "width_prediction": predicted_width_ratio,
                "isotropic_prediction": predicted_isotropic_ratio,
            }
            for key, value in physical_values.items():
                physical_geometry[key].append(value[physical_valid].float().cpu())
            for sequence, valid in zip(
                batch.sequence_ids, physical_valid.cpu(), strict=True
            ):
                if bool(valid):
                    physical_geometry_sequences.append(sequence)
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
    geometry_diagnostics: dict[str, Any] | None = None
    if endpoint_geometry["log_height_target"]:
        endpoint_np = {
            key: torch.cat(values).numpy().astype(np.float64)
            for key, values in endpoint_geometry.items()
        }
        pair_np = {
            key: torch.cat(values).numpy().astype(np.float64)
            for key, values in pair_geometry.items()
        }
        physical_np = {
            key: torch.cat(values).numpy().astype(np.float64)
            for key, values in physical_geometry.items()
        }
        endpoint_sequence_np = np.asarray(endpoint_geometry_sequences)
        pair_sequence_np = np.asarray(pair_geometry_sequences)
        physical_sequence_np = np.asarray(physical_geometry_sequences)
        geometry_diagnostics = {
            "absolute_log_height": _relationship_by_sequence(
                endpoint_np["log_height_target"],
                endpoint_np["log_height_prediction"],
                endpoint_sequence_np,
            ),
            "absolute_log_width": _relationship_by_sequence(
                endpoint_np["log_width_target"],
                endpoint_np["log_width_prediction"],
                endpoint_sequence_np,
            ),
            "centroid_x": _relationship_by_sequence(
                endpoint_np["centroid_x_target"],
                endpoint_np["centroid_x_prediction"],
                endpoint_sequence_np,
            ),
            "centroid_y": _relationship_by_sequence(
                endpoint_np["centroid_y_target"],
                endpoint_np["centroid_y_prediction"],
                endpoint_sequence_np,
            ),
            "delta_log_height_vs_bbox": _relationship_by_sequence(
                pair_np["height_target"],
                pair_np["height_prediction"],
                pair_sequence_np,
            ),
            "delta_log_width_vs_bbox": _relationship_by_sequence(
                pair_np["width_target"],
                pair_np["width_prediction"],
                pair_sequence_np,
            ),
            "delta_log_isotropic_vs_bbox": _relationship_by_sequence(
                pair_np["isotropic_target"],
                pair_np["isotropic_prediction"],
                pair_sequence_np,
            ),
            "delta_log_height_vs_physical": _relationship_by_sequence(
                physical_np["target"],
                physical_np["height_prediction"],
                physical_sequence_np,
            ),
            "delta_log_width_vs_physical": _relationship_by_sequence(
                physical_np["target"],
                physical_np["width_prediction"],
                physical_sequence_np,
            ),
            "delta_log_isotropic_vs_physical": _relationship_by_sequence(
                physical_np["target"],
                physical_np["isotropic_prediction"],
                physical_sequence_np,
            ),
            "r_iso_is_diagnostic_only": True,
            "bbox_used_as_model_input": False,
        }
    return {
        "num_samples": int(target_np.size),
        "loss": sum(value * count for value, count in losses) / sum(count for _, count in losses),
        "signed": signed,
        "sequence_macro": sequence_macro,
        "known_coverage": float(torch.cat(known).float().mean()),
        "weak_bbox_iou": (
            float(torch.cat(weak_iou).mean()) if weak_iou else float("nan")
        ),
        "weak_bbox_iou_count": sum(int(value.numel()) for value in weak_iou),
        "log_ratio_mae": float((ratio - ratio_target).abs().mean()),
        "log_ratio_pearson": pearson,
        "geometry_diagnostics": geometry_diagnostics,
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
    stop_after_epoch: int | None = None,
) -> CausalScaleEAPTrainingResult:
    """Train under a hard wall-clock guard and select on validation only."""

    if not isinstance(train_dataset, Sized) or not isinstance(validation_dataset, Sized):
        raise TypeError("real causal-scale datasets must expose length")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= training_config.epochs:
        raise ValueError("stop_after_epoch must lie within the configured epoch range")
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
    foreground_only = _foreground_only_loss_config(loss_config)
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
        expected_contract = {
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "loss_config": asdict(loss_config),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
        }
        actual_contract = {
            key: saved.get(key) for key in expected_contract
        }
        if actual_contract != expected_contract:
            raise ValueError("resume state differs from the current config/data contract")
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
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "loss_config": asdict(loss_config),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
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
        selectable = eligible and selection["complete_sequence_coverage"] == 1.0
        if selectable and _is_better(selection, best_selection):
            best_state = copy.deepcopy(model.state_dict())
            best_selection = selection
            best_validation = validation
            best_epoch = epoch
            stale = 0
        elif selectable:
            stale += 1
        scheduler.step()
        last_completed_epoch = epoch
        elapsed = prior_elapsed + time.perf_counter() - started
        if last_path is not None:
            save_state(last_path, epoch, elapsed)
        if best_path is not None and best_epoch == epoch:
            save_state(best_path, epoch, elapsed)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            break
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
    "ShardGroupedRandomSampler",
    "checkpoint_payload",
    "evaluate_real_causal_scale",
    "train_one_real_epoch",
    "train_real_causal_scale",
]
