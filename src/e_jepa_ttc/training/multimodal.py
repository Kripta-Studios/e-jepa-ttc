"""Label-free DINOv3-to-Event-JEPA feature distillation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset, ShardLocalSampler
from e_jepa_ttc.models.multimodal import DINOv3FeatureTeacher
from e_jepa_ttc.models.object_jepa import ObjectCentricEventJEPA, ObjectJEPAConfig
from e_jepa_ttc.training.object_jepa import _device, _jepa_forward, _set_seed, _tensor_batch
from e_jepa_ttc.utils.io import write_structured


def _student_teacher_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if student.shape != teacher.shape or valid.shape != student.shape[:-1]:
        msg = "Projected student, teacher and validity shapes are incompatible."
        raise ValueError(msg)
    if not torch.any(valid):
        msg = "DINOv3 distillation batch has no valid object tokens."
        raise ValueError(msg)
    student_normalized = functional.normalize(student[valid], dim=-1)
    teacher_normalized = functional.normalize(teacher[valid].detach(), dim=-1)
    return (1.0 - (student_normalized * teacher_normalized).sum(dim=-1)).mean()


def distill_object_event_jepa_from_dinov3(
    *,
    cache_manifest_path: str | Path,
    event_checkpoint_path: str | Path,
    output_dir: str | Path,
    teacher_model_name: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    distillation_weight: float = 0.25,
    ema_start: float = 0.99,
    ema_end: float = 0.9999,
    seed: int = 42,
    device_name: str = "auto",
    teacher: nn.Module | None = None,
) -> dict[str, Any]:
    """Continue Event-JEPA with its world loss plus frozen DINOv3 targets."""

    if epochs <= 0 or batch_size <= 0 or distillation_weight < 0:
        msg = "Epochs/batch size must be positive and distillation weight non-negative."
        raise ValueError(msg)
    _set_seed(seed)
    device = _device(device_name)
    checkpoint = torch.load(event_checkpoint_path, map_location="cpu", weights_only=False)
    config = ObjectJEPAConfig(**checkpoint["model_config"])
    if not config.pre_cropped_events:
        msg = "DINOv3 object distillation requires a pre-cropped event checkpoint."
        raise ValueError(msg)
    model = ObjectCentricEventJEPA(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher_model = teacher or DINOv3FeatureTeacher(teacher_model_name)
    teacher_model = teacher_model.to(device).eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    train_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("train",))
    validation_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("validation",))
    probe = train_dataset[0]
    if "context_rgb" not in probe:
        msg = "The selected object cache has no context_rgb tensors."
        raise ValueError(msg)
    probe_rgb = probe["context_rgb"]
    if not isinstance(probe_rgb, torch.Tensor):
        msg = "context_rgb must be a tensor."
        raise TypeError(msg)
    with torch.inference_mode():
        teacher_probe = teacher_model(probe_rgb[None].to(device))
    if teacher_probe.ndim != 3:
        msg = "The RGB teacher must return [B,T,D] features."
        raise ValueError(msg)
    teacher_dim = int(teacher_probe.shape[-1])
    projection = nn.Linear(config.embedding_dim, teacher_dim).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=ShardLocalSampler(train_dataset, seed=seed),
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(projection.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    best_path = output / "object_jepa_dinov3_best.pt"
    best_validation = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    total_steps = max(1, epochs * len(train_loader))
    global_step = 0
    start_time = time.perf_counter()
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, epochs + 1):
            model.train()
            projection.train()
            train_total = 0.0
            train_distillation = 0.0
            train_count = 0
            divergence_sum = 0.0
            divergence_count = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    jepa_losses, valid_count = _jepa_forward(model, batch, device)
                    event = model.context_encoder(
                        _tensor_batch(batch, "context_events", device, dtype=torch.float32),
                        _tensor_batch(batch, "context_boxes", device, dtype=torch.float32),
                        _tensor_batch(batch, "context_object_mask", device).bool(),
                        sampling_boxes_xyxy=_tensor_batch(
                            batch,
                            "context_sampling_boxes",
                            device,
                            dtype=torch.float32,
                        ),
                        ego_actions=_tensor_batch(
                            batch,
                            "context_ego_actions",
                            device,
                            dtype=torch.float32,
                        ),
                        ego_action_mask=_tensor_batch(
                            batch,
                            "context_ego_action_mask",
                            device,
                        ).bool(),
                    )
                    with torch.no_grad():
                        teacher_features = teacher_model(
                            _tensor_batch(batch, "context_rgb", device)
                        )[:, -1, None, :]
                    distillation = _student_teacher_distillation_loss(
                        projection(event.object_tokens),
                        teacher_features,
                        event.object_mask,
                    )
                    total = jepa_losses["total"] + distillation_weight * distillation
                if scaler is None:
                    total.backward()
                    nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                else:
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                progress = global_step / max(total_steps - 1, 1)
                momentum = ema_end - (ema_end - ema_start) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
                divergence = model.update_target_encoder(momentum)
                divergence_sum += divergence
                divergence_count += 1
                global_step += 1
                train_total += float(total.detach()) * valid_count
                train_distillation += float(distillation.detach()) * valid_count
                train_count += valid_count
            validation = _evaluate_distillation(
                model,
                projection,
                teacher_model,
                validation_loader,
                device=device,
                distillation_weight=distillation_weight,
            )
            row = {
                "epoch": epoch,
                "train_total": train_total / max(train_count, 1),
                "train_distillation": train_distillation / max(train_count, 1),
                "validation": validation,
                "ema_momentum": momentum,
                "target_encoder_divergence_l2": divergence_sum / max(divergence_count, 1),
            }
            history.append(row)
            history_file.write(json.dumps(row, sort_keys=True) + "\n")
            history_file.flush()
            if validation["total"] < best_validation:
                best_validation = validation["total"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(config),
                        "distillation_projection_state_dict": projection.state_dict(),
                        "teacher_model_name": teacher_model_name,
                        "teacher_feature_dim": teacher_dim,
                        "epoch": epoch,
                        "seed": seed,
                        "checkpoint_role": "best",
                        "selected_by": "validation_jepa_plus_dinov3_distillation",
                        "uses_ttc_labels": False,
                    },
                    best_path,
                )
    summary: dict[str, Any] = {
        "method": "object_event_jepa_dinov3_distillation",
        "teacher_model_name": teacher_model_name,
        "teacher_frozen": True,
        "teacher_feature_dim": teacher_dim,
        "uses_ttc_labels": False,
        "cache_manifest": str(cache_manifest_path),
        "source_event_checkpoint": str(event_checkpoint_path),
        "best_checkpoint": best_path.as_posix(),
        "best_epoch": best_epoch,
        "best_validation_total": best_validation,
        "distillation_weight": distillation_weight,
        "seed": seed,
        "device": str(device),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "elapsed_seconds": time.perf_counter() - start_time,
        "history": history,
    }
    write_structured(output / "summary.json", summary)
    train_dataset.close()
    validation_dataset.close()
    return summary


@torch.no_grad()
def _evaluate_distillation(
    model: ObjectCentricEventJEPA,
    projection: nn.Linear,
    teacher: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor | list[str]]],
    *,
    device: torch.device,
    distillation_weight: float,
) -> dict[str, float]:
    model.eval()
    projection.eval()
    total_sum = 0.0
    jepa_sum = 0.0
    distillation_sum = 0.0
    denominator = 0
    for batch in loader:
        jepa_losses, valid_count = _jepa_forward(model, batch, device)
        event = model.context_encoder(
            _tensor_batch(batch, "context_events", device, dtype=torch.float32),
            _tensor_batch(batch, "context_boxes", device, dtype=torch.float32),
            _tensor_batch(batch, "context_object_mask", device).bool(),
            sampling_boxes_xyxy=_tensor_batch(
                batch,
                "context_sampling_boxes",
                device,
                dtype=torch.float32,
            ),
            ego_actions=_tensor_batch(
                batch,
                "context_ego_actions",
                device,
                dtype=torch.float32,
            ),
            ego_action_mask=_tensor_batch(
                batch,
                "context_ego_action_mask",
                device,
            ).bool(),
        )
        teacher_features = teacher(_tensor_batch(batch, "context_rgb", device))[:, -1, None, :]
        distillation = _student_teacher_distillation_loss(
            projection(event.object_tokens),
            teacher_features,
            event.object_mask,
        )
        total = jepa_losses["total"] + distillation_weight * distillation
        total_sum += float(total) * valid_count
        jepa_sum += float(jepa_losses["total"]) * valid_count
        distillation_sum += float(distillation) * valid_count
        denominator += valid_count
    if denominator == 0:
        msg = "DINOv3 validation contains no valid targets."
        raise ValueError(msg)
    return {
        "total": total_sum / denominator,
        "jepa_total": jepa_sum / denominator,
        "distillation": distillation_sum / denominator,
    }


__all__ = ["distill_object_event_jepa_from_dinov3"]
