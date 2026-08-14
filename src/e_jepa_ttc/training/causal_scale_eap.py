"""Bounded real-data training for the causal event foreground-scale arm."""

from __future__ import annotations

import copy
import hashlib
import json
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
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Sampler

from e_jepa_ttc.data.object_event_v4 import (
    BoxGeometryTargets,
    ObjectEventV4Batch,
    box_geometry_targets,
    collate_object_event_v4,
    weak_box_masks,
)
from e_jepa_ttc.distillation.dinov3_relational import (
    local_relational_distillation_loss,
    local_relational_temporal_delta_loss,
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
    finite_ttc_from_inverse,
    target_log_ratio_from_ttc,
)
from e_jepa_ttc.reproducibility import seed_everything


def _safe_progress_value(value: object) -> object:
    """Convert aggregate telemetry to strict JSON without model state."""

    if isinstance(value, np.generic):
        return _safe_progress_value(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _safe_progress_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_progress_value(item) for item in value]
    return value


def _module_tensor_sha256(module: nn.Module) -> str:
    """Hash ordered tensor content without serialization metadata."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        for value in (name, str(cpu.dtype), repr(tuple(cpu.shape))):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        raw = cpu.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _apply_encoder_freeze(
    model: CausalScaleTTC,
    *,
    freeze_encoder: bool,
    freeze_encoder_stages: int,
) -> list[str]:
    """Freeze the full encoder or the exact leading ``features`` slice."""

    if freeze_encoder and freeze_encoder_stages:
        raise ValueError("full and partial encoder freezing are mutually exclusive")
    if freeze_encoder_stages < 0:
        raise ValueError("freeze_encoder_stages must be non-negative")
    if freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    elif freeze_encoder_stages:
        encoder_stages = list(model.encoder.features.children())
        if freeze_encoder_stages > len(encoder_stages):
            raise ValueError("freeze_encoder_stages exceeds encoder.features length")
        for stage in encoder_stages[:freeze_encoder_stages]:
            for parameter in stage.parameters():
                parameter.requires_grad_(False)
    return sorted(
        name for name, parameter in model.named_parameters() if not parameter.requires_grad
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    prefetch_factor: int = 2
    precision: str = "bf16"
    maximum_runtime_hours: float = 6.0
    mask_t0_as_proxy: bool = True
    foreground_supervision: Literal["weak_box", "bbox_geometry", "bbox_geometry_sam_teacher"] = (
        "weak_box"
    )
    teacher_cache_artifact_sha256: str | None = None
    representation_supervision: Literal[
        "none",
        "dinov3_local_relational",
        "dinov3_local_relational_temporal_delta",
    ] = "none"
    representation_teacher_cache_artifact_sha256: str | None = None
    representation_distillation_weight: float = 0.0
    representation_temporal_delta_weight: float = 0.0
    representation_temporal_delta_calibration_artifact_sha256: str | None = None
    initialization_checkpoint: str | None = None
    initialization_checkpoint_sha256: str | None = None
    initialization_mode: Literal["none", "shape_compatible"] = "none"
    freeze_encoder: bool = False
    freeze_encoder_stages: int = 0
    soft_geometry_teacher_checkpoint: str | None = None
    soft_geometry_teacher_checkpoint_sha256: str | None = None
    soft_dense_cosine_weight: float = 0.0
    soft_geometry_weight: float = 0.0

    def __post_init__(self) -> None:
        integers = (
            self.epochs,
            self.minimum_epochs,
            self.early_stopping_patience + 1,
            self.foreground_warmup_epochs + 1,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.num_workers + 1,
            self.prefetch_factor,
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
        # --- A4: Representation distillation cross-validation ---
        if self.representation_supervision not in {
            "none",
            "dinov3_local_relational",
            "dinov3_local_relational_temporal_delta",
        }:
            raise ValueError(
                f"unsupported representation_supervision: {self.representation_supervision}"
            )

        uses_dino = self.representation_supervision != "none"
        if uses_dino:
            if not bool(self.representation_teacher_cache_artifact_sha256):
                raise ValueError("representation_teacher_cache_artifact_sha256 must be provided")
            if not math.isfinite(self.representation_distillation_weight):
                raise ValueError("representation_distillation_weight must be finite")
            if self.representation_distillation_weight <= 0.0:
                raise ValueError("representation_distillation_weight must be > 0.0")
        else:
            if self.representation_distillation_weight != 0.0:
                raise ValueError(
                    "representation_distillation_weight must be 0.0 "
                    "when representation_supervision is 'none'"
                )

        if not math.isfinite(self.representation_temporal_delta_weight):
            raise ValueError("representation_temporal_delta_weight must be finite")
        uses_temporal_delta = (
            self.representation_supervision == "dinov3_local_relational_temporal_delta"
        )
        if uses_temporal_delta:
            if self.representation_temporal_delta_weight <= 0.0:
                raise ValueError(
                    "representation_temporal_delta_weight must be > 0.0 "
                    "for temporal-delta supervision"
                )
            if not bool(self.representation_temporal_delta_calibration_artifact_sha256):
                raise ValueError(
                    "A4D temporal-delta supervision requires the signed "
                    "calibration artifact identity"
                )
        else:
            if self.representation_temporal_delta_weight != 0.0:
                raise ValueError(
                    "representation_temporal_delta_weight must be 0.0 unless "
                    "representation_supervision is "
                    "'dinov3_local_relational_temporal_delta'"
                )
            if self.representation_temporal_delta_calibration_artifact_sha256 is not None:
                raise ValueError("temporal-delta calibration identity must be absent outside A4D")

        uses_initialization = self.initialization_mode != "none"
        if uses_initialization:
            if self.initialization_mode != "shape_compatible":
                raise ValueError("unsupported initialization_mode")
            if not self.initialization_checkpoint or not self.initialization_checkpoint_sha256:
                raise ValueError(
                    "shape-compatible initialization requires checkpoint path and SHA256"
                )
            if len(self.initialization_checkpoint_sha256) != 64:
                raise ValueError("initialization_checkpoint_sha256 must be a SHA256 hex digest")
        else:
            if (
                self.initialization_checkpoint is not None
                or self.initialization_checkpoint_sha256 is not None
            ):
                raise ValueError(
                    "initialization checkpoint metadata must be absent when "
                    "initialization_mode=none"
                )
        if self.freeze_encoder:
            if not uses_initialization:
                raise ValueError("freeze_encoder requires a frozen initialization checkpoint")
            if self.foreground_warmup_epochs != 0:
                raise ValueError(
                    "freeze_encoder requires foreground_warmup_epochs=0 because "
                    "foreground-only warmup has no trainable path"
                )
        if self.freeze_encoder_stages < 0:
            raise ValueError("freeze_encoder_stages must be non-negative")
        uses_soft_teacher = self.soft_geometry_teacher_checkpoint is not None
        if uses_soft_teacher != bool(self.soft_geometry_teacher_checkpoint_sha256):
            raise ValueError("soft geometry teacher path and SHA256 must be declared together")
        soft_weights = (self.soft_dense_cosine_weight, self.soft_geometry_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in soft_weights):
            raise ValueError("soft geometry weights must be finite and non-negative")
        if uses_soft_teacher != all(value > 0.0 for value in soft_weights):
            raise ValueError("soft geometry teacher requires both losses with positive weights")
        if self.freeze_encoder and self.freeze_encoder_stages:
            raise ValueError("full and partial encoder freezing are mutually exclusive")


@dataclass
class CausalScaleEAPTrainingResult:
    model: CausalScaleTTC
    history: list[dict[str, Any]]
    best_epoch: int
    best_selection: dict[str, float]
    best_validation: dict[str, Any]
    elapsed_seconds: float
    initialization: dict[str, Any]


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
    foreground_supervision: Literal["weak_box", "bbox_geometry", "bbox_geometry_sam_teacher"],
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
    foreground_supervision: Literal["weak_box", "bbox_geometry", "bbox_geometry_sam_teacher"],
    representation_supervision: Literal[
        "none",
        "dinov3_local_relational",
        "dinov3_local_relational_temporal_delta",
    ] = "none",
    representation_distillation_weight: float = 0.0,
    representation_temporal_delta_weight: float = 0.0,
    soft_geometry_teacher: CausalScaleTTC | None = None,
    soft_dense_cosine_weight: float = 0.0,
    soft_geometry_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Any]:
    targets = _targets(
        batch,
        mask_t0_as_proxy=mask_t0_as_proxy,
        foreground_supervision=foreground_supervision,
    )
    need_dense = representation_supervision != "none" or soft_geometry_teacher is not None
    output = model(
        batch.events,
        targets.delta_t_s,
        return_dense_features=need_dense,
    )
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
    total = result.total
    components = dict(result.components)

    if soft_geometry_teacher is not None:
        soft_geometry_teacher.eval()
        with torch.no_grad():
            teacher_output = soft_geometry_teacher(
                batch.events,
                targets.delta_t_s,
                return_dense_features=True,
            )
        if output.endpoint_dense_features is None or teacher_output.endpoint_dense_features is None:
            raise RuntimeError("soft geometry distillation requires dense endpoint features")
        student_dense = functional.normalize(output.endpoint_dense_features.float(), dim=2)
        teacher_dense = functional.normalize(teacher_output.endpoint_dense_features.float(), dim=2)
        if student_dense.shape != teacher_dense.shape:
            raise ValueError("soft geometry teacher and student dense features differ in shape")
        dense_cosine = (1.0 - (student_dense * teacher_dense).sum(dim=2)).mean()
        student_geometry = torch.stack(
            (
                output.visible_height_normalized.clamp_min(1.0e-6).log(),
                output.visible_width_normalized.clamp_min(1.0e-6).log(),
                output.diagnostics["foreground_centroid_x"],
                output.diagnostics["foreground_centroid_y"],
            ),
            dim=-1,
        )
        teacher_geometry = torch.stack(
            (
                teacher_output.visible_height_normalized.clamp_min(1.0e-6).log(),
                teacher_output.visible_width_normalized.clamp_min(1.0e-6).log(),
                teacher_output.diagnostics["foreground_centroid_x"],
                teacher_output.diagnostics["foreground_centroid_y"],
            ),
            dim=-1,
        )
        geometry_distillation = functional.smooth_l1_loss(
            student_geometry.float(),
            teacher_geometry.float(),
            beta=0.02,
        )
        weighted_dense = soft_dense_cosine_weight * dense_cosine
        weighted_geometry = soft_geometry_weight * geometry_distillation
        total = total + weighted_dense + weighted_geometry
        components["soft_dense_cosine_raw"] = dense_cosine.detach()
        components["soft_dense_cosine_weighted"] = weighted_dense.detach()
        components["soft_geometry_raw"] = geometry_distillation.detach()
        components["soft_geometry_weighted"] = weighted_geometry.detach()

    # --- A4: DINOv3 relational distillation (train-only) ---
    if (
        representation_supervision
        in {
            "dinov3_local_relational",
            "dinov3_local_relational_temporal_delta",
        }
        and batch.dinov3_relation_targets is not None
        and batch.dinov3_relation_valid is not None
    ):
        if output.endpoint_dense_features is None:
            raise RuntimeError(
                "DINO relational distillation requires dense features but "
                "the model did not return them"
            )
        # Use t1/t2 endpoints (indices 1 and 2 in the T-step sequence)
        # The model produces [B,T,C,H,W]; we select the observation endpoints.
        dense = output.endpoint_dense_features
        if dense.shape[1] < 3:
            raise ValueError(
                f"DINO distillation requires at least 3 temporal endpoints, got {dense.shape[1]}"
            )
        # Select t1, t2 (not t0 which is the proxy reference)
        student_features = dense[:, 1:3]  # [B, 2, C, H, W]

        relational_loss = local_relational_distillation_loss(
            student_features,
            batch.dinov3_relation_targets,
            batch.dinov3_relation_valid,
        )
        weighted_relational = representation_distillation_weight * relational_loss
        total = total + weighted_relational
        components["dinov3_relational_raw"] = relational_loss.detach()
        components["dinov3_relational_weighted"] = weighted_relational.detach()

        # --- A4D: temporal change in the same local relation maps ---
        if representation_supervision == "dinov3_local_relational_temporal_delta":
            temporal_delta_loss = local_relational_temporal_delta_loss(
                student_features,
                batch.dinov3_relation_targets,
                batch.dinov3_relation_valid,
            )
            weighted_temporal_delta = representation_temporal_delta_weight * temporal_delta_loss
            total = total + weighted_temporal_delta
            components["dinov3_relational_temporal_delta_raw"] = temporal_delta_loss.detach()
            components["dinov3_relational_temporal_delta_weighted"] = (
                weighted_temporal_delta.detach()
            )

        # --- Train-only fg/bg diagnostic (not used for selection) ---
        _record_relational_fg_bg_diagnostic(
            student_features,
            batch.dinov3_relation_targets,
            batch.dinov3_relation_valid,
            batch.boxes_xyxy,
            source_height=batch.events.shape[-2],
            source_width=batch.events.shape[-1],
            feat_h=dense.shape[-2],
            feat_w=dense.shape[-1],
            components=components,
        )
    elif representation_supervision in {
        "dinov3_local_relational",
        "dinov3_local_relational_temporal_delta",
    }:
        raise ValueError(
            "DINO relational supervision is active but batch lacks "
            "dinov3_relation_targets/dinov3_relation_valid"
        )

    return total, components, output


def _record_relational_fg_bg_diagnostic(
    student_features: torch.Tensor,
    teacher_relations: torch.Tensor,
    teacher_valid: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    source_height: int,
    source_width: int,
    feat_h: int,
    feat_w: int,
    components: dict[str, torch.Tensor],
) -> None:
    """Record inside/outside bbox relational loss split as diagnostics.

    A4 deliberately distills the complete target-specific common ROI without
    bbox-masking.  These diagnostics let us understand whether background/other
    objects inside the ROI dilute the teacher gradient, but they are NEVER
    used for selection or training.
    """

    from e_jepa_ttc.distillation.dinov3_relational import (
        local_cosine_relation_maps,
    )

    with torch.no_grad():
        student_rels = local_cosine_relation_maps(student_features)
        combined_valid = teacher_valid.bool() & student_rels.valid
        error_map = (
            student_rels.values - teacher_relations.float()
        ).abs() * combined_valid.float()  # [B, 2, K, H, W]

        # Build a coarse fg mask at feature resolution from the t1/t2 boxes
        # boxes_xyxy is [B, T, 4]; we need endpoints 1,2
        batch_size = student_features.shape[0]
        fg_mask = torch.zeros(
            batch_size,
            2,
            feat_h,
            feat_w,
            dtype=torch.bool,
            device=student_features.device,
        )
        if min(source_height, source_width, feat_h, feat_w) <= 0:
            raise ValueError("relational fg/bg diagnostic dimensions must be positive")
        scale_x = feat_w / float(source_width)
        scale_y = feat_h / float(source_height)
        for b in range(batch_size):
            for ep_idx, t_idx in enumerate([1, 2]):
                if t_idx >= boxes_xyxy.shape[1]:
                    continue
                box = boxes_xyxy[b, t_idx].float()  # common-ROI pixel coordinates
                x1 = int(torch.floor(box[0] * scale_x).clamp(0, feat_w).item())
                y1 = int(torch.floor(box[1] * scale_y).clamp(0, feat_h).item())
                x2 = int(torch.ceil(box[2] * scale_x).clamp(0, feat_w).item())
                y2 = int(torch.ceil(box[3] * scale_y).clamp(0, feat_h).item())
                if x2 > x1 and y2 > y1:
                    fg_mask[b, ep_idx, y1:y2, x1:x2] = True

        # Expand fg_mask to match error_map: [B,2,K,H,W]
        fg_expanded = fg_mask.unsqueeze(2).expand_as(error_map)
        valid_fg = combined_valid & fg_expanded
        valid_bg = combined_valid & ~fg_expanded

        fg_loss = (
            error_map[valid_fg].mean() if valid_fg.any() else error_map.new_tensor(float("nan"))
        )
        bg_loss = (
            error_map[valid_bg].mean() if valid_bg.any() else error_map.new_tensor(float("nan"))
        )
        fg_fraction = (
            valid_fg.float().sum() / combined_valid.float().sum()
            if combined_valid.any()
            else error_map.new_tensor(float("nan"))
        )
        components["dinov3_relational_fg_loss"] = fg_loss
        components["dinov3_relational_bg_loss"] = bg_loss
        components["dinov3_relational_fg_fraction"] = fg_fraction


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
        shard_order = torch.randperm(len(self.groups), generator=self.generator).tolist()
        for shard_index in shard_order:
            group = self.groups[shard_index]
            record_order = torch.randperm(len(group), generator=self.generator).tolist()
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
            prefetch_factor=(config.prefetch_factor if config.num_workers > 0 else None),
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
    soft_geometry_teacher: CausalScaleTTC | None = None,
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
                representation_supervision=config.representation_supervision,
                representation_distillation_weight=config.representation_distillation_weight,
                representation_temporal_delta_weight=(config.representation_temporal_delta_weight),
                soft_geometry_teacher=soft_geometry_teacher,
                soft_dense_cosine_weight=config.soft_dense_cosine_weight,
                soft_geometry_weight=config.soft_geometry_weight,
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
        totals["total"] = totals.get("total", 0.0) + float(total.detach().cpu()) * count
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * count
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
    *,
    use_auxiliary_dev_metadata: bool = True,
) -> dict[str, Any]:
    """Evaluate signed TTC; optionally add post-selection geometry diagnostics."""

    model.eval()
    truth: list[torch.Tensor] = []
    prediction: list[torch.Tensor] = []
    point_prediction: list[torch.Tensor] = []
    auxiliary_prediction: list[torch.Tensor] = []
    guard_margins: list[torch.Tensor] = []
    ttc_log_variances: list[torch.Tensor] = []
    event_counts: list[torch.Tensor] = []
    event_rates: list[torch.Tensor] = []
    motion_magnitudes: list[torch.Tensor] = []
    known: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    ratio_targets: list[torch.Tensor] = []
    ratio_components: dict[str, list[torch.Tensor]] = {
        "analytic": [],
        "residual": [],
        "final_pair": [],
    }
    ratio_component_sequences: list[str] = []
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
    transport_geometry: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "target",
            "divergence_x",
            "divergence_y",
            "divergence_isotropic",
            "foreground_divergence_x",
            "foreground_divergence_y",
            "foreground_divergence_isotropic",
        )
    }
    transport_quality: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "confidence_margin",
            "entropy",
            "cycle_error",
            "foreground_confidence_margin",
            "foreground_entropy",
            "foreground_cycle_error",
            "flow_magnitude",
            "foreground_flow_magnitude",
        )
    }
    endpoint_geometry_sequences: list[str] = []
    pair_geometry_sequences: list[str] = []
    physical_geometry_sequences: list[str] = []
    transport_geometry_sequences: list[str] = []
    transport_quality_sequences: list[str] = []
    sequences: list[str] = []
    tokens: list[str] = []
    tracks: list[str] = []
    losses: list[tuple[float, int]] = []
    for host_batch in loader:
        batch = host_batch.to(device)
        # --- A4: Validation must never see DINO teacher fields ---
        if batch.dinov3_relation_targets is not None or batch.dinov3_relation_valid is not None:
            raise ValueError(
                "DINO teacher fields must not appear in validation batches. "
                "The DINOv3RelationalTeacherDataset wrapper must only be "
                "applied to the train dataset."
            )
        if use_auxiliary_dev_metadata:
            with _autocast(device, config.precision):
                total, _, output = _loss(
                    model,
                    batch,
                    loss_config,
                    mask_t0_as_proxy=config.mask_t0_as_proxy,
                    foreground_supervision=config.foreground_supervision,
                    # Representation supervision is always absent on dev.
                )
            targets = _targets(
                batch,
                mask_t0_as_proxy=config.mask_t0_as_proxy,
                foreground_supervision=config.foreground_supervision,
            )
        else:
            delta = batch.delta_t_s[:, None].expand(-1, batch.events.shape[1] - 1)
            target_valid = torch.isfinite(batch.target_ttc_s) & (batch.target_ttc_s != 0.0)
            targets = CausalScaleEAPTargets(
                delta_t_s=delta,
                target_valid=target_valid,
                target_masks=None,
                mask_valid=None,
                geometry=None,
            )
            with _autocast(device, config.precision):
                output = model(batch.events, delta, return_dense_features=False)
                ttc_only = causal_scale_ttc_loss(
                    output,
                    target_ttc_seconds=batch.target_ttc_s,
                    delta_t_s=delta,
                    risk_thresholds_s=model.config.risk_thresholds_s,
                    target_valid=target_valid,
                    target_masks=None,
                    mask_valid=None,
                    target_geometry=None,
                    config=loss_config,
                )
                total = ttc_only.total
        target_ratio, valid_ratio = target_log_ratio_from_ttc(
            batch.target_ttc_s, targets.delta_t_s[:, -1]
        )
        if targets.target_masks is not None and targets.mask_valid is not None:
            predicted_masks = torch.sigmoid(output.foreground_logits) >= 0.5
            selected_masks = targets.mask_valid
            intersection = (
                (predicted_masks & targets.target_masks.bool()).sum(dim=(-3, -2, -1)).float()
            )
            union = (
                (predicted_masks | targets.target_masks.bool())
                .sum(dim=(-3, -2, -1))
                .float()
                .clamp_min(1)
            )
            weak_iou.append((intersection[selected_masks] / union[selected_masks]).cpu())
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
            for sequence, row_valid in zip(batch.sequence_ids, endpoint_valid_cpu, strict=True):
                endpoint_geometry_sequences.extend([sequence] * int(row_valid.sum().item()))

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
            predicted_isotropic_ratio = 0.5 * (predicted_height_ratio + predicted_width_ratio)
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
            for sequence, valid in zip(batch.sequence_ids, physical_valid.cpu(), strict=True):
                if bool(valid):
                    physical_geometry_sequences.append(sequence)

        if "transport_divergence_isotropic" in output.diagnostics:
            current_pair = -1
            transport_valid = valid_ratio
            transport_geometry["target"].append(target_ratio[transport_valid].float().cpu())
            for name in (
                "divergence_x",
                "divergence_y",
                "divergence_isotropic",
                "foreground_divergence_x",
                "foreground_divergence_y",
                "foreground_divergence_isotropic",
            ):
                transport_geometry[name].append(
                    output.diagnostics[f"transport_{name}"][:, current_pair][transport_valid]
                    .float()
                    .cpu()
                )
            for name in transport_quality:
                transport_quality[name].append(
                    output.diagnostics[f"transport_{name}"][:, current_pair].float().cpu()
                )
            for sequence, valid in zip(batch.sequence_ids, transport_valid.cpu(), strict=True):
                if bool(valid):
                    transport_geometry_sequences.append(sequence)
            transport_quality_sequences.extend(batch.sequence_ids)
        count = int(batch.events.shape[0])
        losses.append((float(total.detach().cpu()), count))
        truth.append(batch.target_ttc_s.cpu())
        current_prediction = output.ttc_mean_seconds.float()
        point_prediction.append(current_prediction.cpu())
        auxiliary_prediction.append(
            finite_ttc_from_inverse(
                output.auxiliary_inverse_ttc[:, -1].float(),
                minimum_abs_inverse_ttc=1.0 / model.config.ttc_clip_seconds,
                clip_seconds=model.config.ttc_clip_seconds,
            ).cpu()
        )
        guard_margins.append(
            torch.minimum(
                output.log_height_ratio[:, -1].abs() / 0.002,
                output.sensor_support[:, -1] / 0.0001,
            )
            .float()
            .cpu()
        )
        ttc_log_variances.append(output.ttc_log_variance.float().cpu())
        event_counts.append(batch.events[:, -1, -2].float().mean(dim=(-2, -1)).cpu())
        event_rates.append(batch.events[:, -1, -1].float().mean(dim=(-2, -1)).cpu())
        motion_magnitudes.append(
            output.diagnostics.get(
                "transport_flow_magnitude",
                torch.full(
                    (batch.events.shape[0], batch.events.shape[1] - 1),
                    float("nan"),
                    device=batch.events.device,
                ),
            )[:, -1]
            .float()
            .cpu()
        )
        current_prediction = torch.where(
            output.known_mask, current_prediction, torch.full_like(current_prediction, float("nan"))
        )
        prediction.append(current_prediction.cpu())
        known.append(output.known_mask.cpu())
        ratios.append(output.log_height_ratio[:, -1][valid_ratio].float().cpu())
        ratio_targets.append(target_ratio[valid_ratio].float().cpu())
        ratio_components["analytic"].append(
            output.analytic_log_height_ratio[:, -1][valid_ratio].float().cpu()
        )
        ratio_components["residual"].append(
            output.residual_log_height_ratio[:, -1][valid_ratio].float().cpu()
        )
        ratio_components["final_pair"].append(
            output.pair_log_height_ratio[:, -1][valid_ratio].float().cpu()
        )
        for sequence, valid in zip(batch.sequence_ids, valid_ratio.cpu(), strict=True):
            if bool(valid):
                ratio_component_sequences.append(sequence)
        sequences.extend(batch.sequence_ids)
        tokens.extend(batch.sample_tokens)
        tracks.extend(batch.track_ids)
    target_np = torch.cat(truth).numpy().astype(np.float64)
    prediction_np = torch.cat(prediction).numpy().astype(np.float64)
    point_prediction_np = torch.cat(point_prediction).numpy().astype(np.float64)
    auxiliary_prediction_np = torch.cat(auxiliary_prediction).numpy().astype(np.float64)
    guard_margin_np = torch.cat(guard_margins).numpy().astype(np.float64)
    ttc_log_variance_np = torch.cat(ttc_log_variances).numpy().astype(np.float64)
    event_count_np = torch.cat(event_counts).numpy().astype(np.float64)
    event_rate_np = torch.cat(event_rates).numpy().astype(np.float64)
    motion_magnitude_np = torch.cat(motion_magnitudes).numpy().astype(np.float64)
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
    component_np = {
        key: torch.cat(values).numpy().astype(np.float64)
        for key, values in ratio_components.items()
    }
    component_sequences_np = np.asarray(ratio_component_sequences)
    ratio_component_diagnostics = {
        key: _relationship_by_sequence(
            ratio_target.numpy().astype(np.float64), value, component_sequences_np
        )
        for key, value in component_np.items()
    }
    residual_values = component_np["residual"]
    residual_bound = float(model.config.max_abs_log_ratio_residual)
    ratio_component_diagnostics["residual_saturation"] = {
        "absolute_mean": float(np.mean(np.abs(residual_values))),
        "absolute_median": float(np.median(np.abs(residual_values))),
        "fraction_above_80pct_bound": float(
            np.mean(np.abs(residual_values) >= 0.8 * residual_bound)
        )
        if residual_bound > 0.0
        else 0.0,
        "fraction_above_95pct_bound": float(
            np.mean(np.abs(residual_values) >= 0.95 * residual_bound)
        )
        if residual_bound > 0.0
        else 0.0,
        "bound": residual_bound,
    }
    signed = signed_garl_metrics(target_np, prediction_np)
    signed_point = signed_garl_metrics(target_np, point_prediction_np)
    sequence_macro = sequence_macro_signed_metrics(target_np, prediction_np, np.asarray(sequences))
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
    transport_diagnostics: dict[str, Any] | None = None
    if transport_geometry["target"]:
        transport_np = {
            key: torch.cat(values).numpy().astype(np.float64)
            for key, values in transport_geometry.items()
        }
        transport_sequence_np = np.asarray(transport_geometry_sequences)
        quality_np = {
            key: torch.cat(values).numpy().astype(np.float64)
            for key, values in transport_quality.items()
        }
        quality_sequence_np = np.asarray(transport_quality_sequences)
        transport_diagnostics = {
            "against_physical_log_ratio": {
                name: _relationship_by_sequence(
                    transport_np["target"],
                    transport_np[name],
                    transport_sequence_np,
                )
                for name in transport_geometry
                if name != "target"
            },
            "quality": {
                name: {
                    "global_mean": float(np.mean(values)),
                    "global_median": float(np.median(values)),
                    "per_sequence_mean": {
                        str(sequence): float(np.mean(values[quality_sequence_np == sequence]))
                        for sequence in sorted(set(quality_sequence_np.astype(str).tolist()))
                    },
                }
                for name, values in quality_np.items()
            },
            "event_only_inference": True,
            "bbox_used_for_transport": False,
        }

    return {
        "num_samples": int(target_np.size),
        "loss": sum(value * count for value, count in losses) / sum(count for _, count in losses),
        "signed": signed,
        "signed_point": signed_point,
        "sequence_macro": sequence_macro,
        "known_coverage": float(torch.cat(known).float().mean()),
        "weak_bbox_iou": (float(torch.cat(weak_iou).mean()) if weak_iou else float("nan")),
        "weak_bbox_iou_count": sum(int(value.numel()) for value in weak_iou),
        "log_ratio_mae": float((ratio - ratio_target).abs().mean()),
        "log_ratio_pearson": pearson,
        "ratio_component_diagnostics": ratio_component_diagnostics,
        "geometry_diagnostics": geometry_diagnostics,
        "transport_diagnostics": transport_diagnostics,
        "sample_tokens": tokens,
        "target_ttc_s": target_np.tolist(),
        "prediction_ttc_s": prediction_np.tolist(),
        "point_prediction_ttc_s": point_prediction_np.tolist(),
        "auxiliary_prediction_ttc_s": auxiliary_prediction_np.tolist(),
        "known_mask": torch.cat(known).numpy().astype(bool).tolist(),
        "guard_margin": guard_margin_np.tolist(),
        "ttc_log_variance": ttc_log_variance_np.tolist(),
        "ttc_variance": np.exp(ttc_log_variance_np).tolist(),
        "event_count_log1p": event_count_np.tolist(),
        "event_rate_log1p": event_rate_np.tolist(),
        "transport_flow_magnitude": motion_magnitude_np.tolist(),
        "sequence_ids": sequences,
        "track_ids": tracks,
    }


def _shape_compatible_initialize(
    model: CausalScaleTTC,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load coherent A4 modules whose complete tensor schemas still match A5.

    If any tensor under a top-level module changes shape (for example the A5
    transport-expanded residual or pair projector), the *entire* module is left
    at its preregistered A5 initialization.  This avoids incoherent hybrids such
    as loading a trained output layer on top of a newly random input layer.
    """

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if payload.get("artifact_type") not in {
        "causal_scale_eap_public_validation_checkpoint_v1",
        "causal_scale_eap_grouped_dev_checkpoint_v1",
    }:
        raise ValueError("initialization checkpoint has an incompatible artifact type")
    source_state = payload.get("model_state_dict")
    if not isinstance(source_state, dict):
        raise ValueError("initialization checkpoint is missing model_state_dict")
    current = model.state_dict()

    def group(name: str) -> str:
        return name.split(".", 1)[0]

    mismatched: list[str] = []
    unexpected: list[str] = []
    blocked_groups: set[str] = set()
    for name, value in source_state.items():
        if name not in current:
            unexpected.append(str(name))
            blocked_groups.add(group(str(name)))
            continue
        if not isinstance(value, torch.Tensor) or value.shape != current[name].shape:
            mismatched.append(str(name))
            blocked_groups.add(group(str(name)))

    compatible: dict[str, torch.Tensor] = {}
    for name, value in source_state.items():
        name = str(name)
        if group(name) in blocked_groups:
            continue
        if (
            name in current
            and isinstance(value, torch.Tensor)
            and value.shape == current[name].shape
        ):
            compatible[name] = value
    model.load_state_dict(compatible, strict=False)
    transport_encoder_initialized_from_primary = False
    if model.transport_encoder is not None:
        model.transport_encoder.load_state_dict(model.encoder.state_dict(), strict=True)
        transport_encoder_initialized_from_primary = True
    missing = sorted(set(current) - set(compatible))
    encoder_keys = [name for name in current if name.startswith("encoder.")]
    loaded_encoder_keys = [name for name in compatible if name.startswith("encoder.")]
    if encoder_keys and len(loaded_encoder_keys) != len(encoder_keys):
        raise ValueError(
            "shape-compatible initialization did not recover the complete endpoint encoder"
        )
    return {
        "mode": "shape_compatible",
        "source_artifact_type": payload.get("artifact_type"),
        "source_model_config": payload.get("model_config"),
        "loaded_tensor_count": len(compatible),
        "missing_tensor_count": len(missing),
        "mismatched_tensor_count": len(mismatched),
        "unexpected_tensor_count": len(unexpected),
        "blocked_top_level_groups": sorted(blocked_groups),
        "mismatched_tensors": sorted(mismatched),
        "unexpected_tensors": sorted(unexpected),
        "complete_encoder_loaded": True,
        "transport_encoder_initialized_from_primary": transport_encoder_initialized_from_primary,
    }


def _load_soft_geometry_teacher(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    device: torch.device,
) -> tuple[CausalScaleTTC, dict[str, Any]]:
    """Load a frozen fold-local A4 teacher for train-only distillation."""

    observed_sha256 = _file_sha256(checkpoint_path)
    if observed_sha256 != expected_sha256:
        raise ValueError("soft geometry teacher checkpoint SHA256 differs")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(raw_config, dict) or not isinstance(state, dict):
        raise ValueError("soft geometry teacher checkpoint is malformed")
    teacher = CausalScaleTTC(CausalScaleTTCConfig(**raw_config))
    teacher.load_state_dict(state, strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.to(device)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("soft geometry teacher is not frozen")
    return teacher, {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": observed_sha256,
        "training": False,
        "frozen": True,
        "parameter_count": sum(parameter.numel() for parameter in teacher.parameters()),
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
    train_loader = _loader(train_dataset, training_config, train=True, generator=generator)
    validation_loader = _loader(validation_dataset, training_config, train=False)
    model = CausalScaleTTC(model_config).to(device)
    soft_geometry_teacher: CausalScaleTTC | None = None
    soft_teacher_metadata: dict[str, Any] | None = None
    if training_config.soft_geometry_teacher_checkpoint is not None:
        expected_teacher_sha = training_config.soft_geometry_teacher_checkpoint_sha256
        if expected_teacher_sha is None:
            raise ValueError("soft geometry teacher SHA256 is unavailable")
        soft_geometry_teacher, soft_teacher_metadata = _load_soft_geometry_teacher(
            Path(training_config.soft_geometry_teacher_checkpoint),
            expected_sha256=expected_teacher_sha,
            device=device,
        )
    initialization: dict[str, Any] = {
        "mode": "none",
        "freeze_encoder": False,
        "frozen_parameter_count": 0,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if training_config.initialization_mode == "shape_compatible":
        if training_config.initialization_checkpoint is None:
            raise ValueError("shape-compatible initialization checkpoint is unavailable")
        initialization = _shape_compatible_initialize(
            model, training_config.initialization_checkpoint
        )
        initialization["checkpoint"] = training_config.initialization_checkpoint
        initialization["checkpoint_sha256"] = training_config.initialization_checkpoint_sha256
    frozen_parameter_names = _apply_encoder_freeze(
        model,
        freeze_encoder=training_config.freeze_encoder,
        freeze_encoder_stages=training_config.freeze_encoder_stages,
    )
    if training_config.freeze_encoder:
        initialization["freeze_encoder"] = True
    if training_config.freeze_encoder_stages:
        initialization["freeze_encoder_stages"] = training_config.freeze_encoder_stages
    initialization["soft_geometry_teacher"] = soft_teacher_metadata
    initialization["soft_geometry_teacher_excluded_from_optimizer"] = (
        soft_geometry_teacher is not None
    )
    frozen_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameter_count <= 0:
        raise RuntimeError("causal-scale run has no trainable parameters")
    initialization["frozen_parameter_count"] = frozen_parameter_count
    initialization["trainable_parameter_count"] = trainable_parameter_count
    optimizer_parameter_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    initialization["frozen_parameter_names"] = frozen_parameter_names
    initialization["optimizer_parameter_names"] = optimizer_parameter_names
    initialization["frozen_optimizer_overlap"] = sorted(
        set(frozen_parameter_names) & set(optimizer_parameter_names)
    )
    initial_primary_encoder_sha256 = _module_tensor_sha256(model.encoder)
    initialization["initial_primary_encoder_sha256"] = initial_primary_encoder_sha256
    initial_transport_encoder_sha256 = (
        _module_tensor_sha256(model.transport_encoder)
        if model.transport_encoder is not None
        else None
    )
    initialization["initial_transport_encoder_sha256"] = initial_transport_encoder_sha256
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
    progress_path = checkpoint_dir / "progress.json" if checkpoint_dir is not None else None
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
        actual_contract = {key: saved.get(key) for key in expected_contract}
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
    remaining_seconds = max(0.0, training_config.maximum_runtime_hours * 3600.0 - prior_elapsed)
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

    def save_progress(*, status: str, epoch: int, elapsed: float) -> None:
        if progress_path is None:
            return
        payload = _safe_progress_value(
            {
                "artifact_type": "causal_scale_eap_safe_progress_v1",
                "status": status,
                "epoch": epoch,
                "configured_epochs": training_config.epochs,
                "best_epoch": best_epoch,
                "stale_epochs": stale,
                "elapsed_seconds": elapsed,
                "latest_selection": history[-1].get("selection") if history else None,
                "best_selection": best_selection,
            }
        )
        temporary = progress_path.with_name(f".{progress_path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, progress_path)

    last_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, training_config.epochs + 1):
        if time.perf_counter() >= deadline:
            break
        epoch_started = time.perf_counter()
        active_loss = (
            foreground_only if epoch <= training_config.foreground_warmup_epochs else loss_config
        )
        train_metrics = train_one_real_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training_config,
            active_loss,
            soft_geometry_teacher,
        )
        validation = evaluate_real_causal_scale(
            model,
            validation_loader,
            device,
            training_config,
            loss_config,
            use_auxiliary_dev_metadata=False,
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
                if key
                not in {
                    "sample_tokens",
                    "sequence_ids",
                    "track_ids",
                    "target_ttc_s",
                    "prediction_ttc_s",
                    "point_prediction_ttc_s",
                    "auxiliary_prediction_ttc_s",
                    "known_mask",
                    "guard_margin",
                    "ttc_log_variance",
                    "ttc_variance",
                    "event_count_log1p",
                    "event_rate_log1p",
                    "transport_flow_magnitude",
                }
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
        save_progress(status="running", epoch=epoch, elapsed=elapsed)
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
    final_primary_encoder_sha256 = _module_tensor_sha256(model.encoder)
    initialization["final_primary_encoder_sha256"] = final_primary_encoder_sha256
    initialization["primary_encoder_exact_initial"] = (
        final_primary_encoder_sha256 == initial_primary_encoder_sha256
    )
    if model.transport_encoder is not None:
        final_transport_encoder_sha256 = _module_tensor_sha256(model.transport_encoder)
        initialization["final_transport_encoder_sha256"] = final_transport_encoder_sha256
        initialization["transport_encoder_changed_from_initial"] = (
            final_transport_encoder_sha256 != initial_transport_encoder_sha256
        )
    post_selection_validation = evaluate_real_causal_scale(
        model,
        validation_loader,
        device,
        training_config,
        loss_config,
        use_auxiliary_dev_metadata=True,
    )
    if _selection(post_selection_validation) != best_selection:
        raise RuntimeError("post-selection geometry evaluation changed TTC selection metrics")
    best_validation = post_selection_validation
    elapsed_total = prior_elapsed + time.perf_counter() - started
    save_progress(status="completed", epoch=last_completed_epoch, elapsed=elapsed_total)
    if last_path is not None and last_completed_epoch < start_epoch:
        raise RuntimeError("runtime budget expired before a resumable epoch completed")
    return CausalScaleEAPTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_selection=best_selection,
        best_validation=best_validation,
        elapsed_seconds=elapsed_total,
        initialization=initialization,
    )


def checkpoint_payload(
    result: CausalScaleEAPTrainingResult,
    training_config: CausalScaleEAPTrainingConfig,
    loss_config: CausalScaleTTCLossConfig,
    *,
    artifact_type: str = "causal_scale_eap_public_validation_checkpoint_v1",
) -> dict[str, Any]:
    if artifact_type not in {
        "causal_scale_eap_public_validation_checkpoint_v1",
        "causal_scale_eap_grouped_dev_checkpoint_v1",
    }:
        raise ValueError(f"unsupported causal-scale checkpoint artifact type: {artifact_type}")
    return {
        "artifact_type": artifact_type,
        "model_config": result.model.checkpoint_config(),
        "training_config": asdict(training_config),
        "loss_config": asdict(loss_config),
        "best_epoch": result.best_epoch,
        "best_selection": result.best_selection,
        "initialization": result.initialization,
        "model_state_dict": result.model.state_dict(),
    }


__all__ = [
    "CausalScaleEAPTrainingConfig",
    "CausalScaleEAPTrainingResult",
    "ShardGroupedRandomSampler",
    "checkpoint_payload",
    "_shape_compatible_initialize",
    "evaluate_real_causal_scale",
    "train_one_real_epoch",
    "train_real_causal_scale",
]
