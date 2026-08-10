#!/usr/bin/env python
"""A5 transport preflight: zero-training, train-only correspondence/gradient audit.

This diagnostic never opens validation/test and never takes an optimizer step.  It
compares fixed-coordinate DINO relation deltas (p->p) with local transported
relation matching (p->q) at radii 1/2/4, audits foreground/background behaviour,
student event-feature correspondence quality, and encoder-gradient interactions
between A4, A4D, geometry, and TTC objectives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.data.dinov3_relational_teacher_cache import DINOv3RelationalTeacherDataset  # noqa: E402
from e_jepa_ttc.data.object_event_v4 import (  # noqa: E402
    GarlTTCObjectEventV4Dataset,
    collate_object_event_v4,
    weak_box_masks,
)
from e_jepa_ttc.distillation.dinov3_relational import (  # noqa: E402
    local_relational_distillation_loss,
    local_relational_temporal_delta_loss,
)
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig, causal_scale_ttc_loss  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig  # noqa: E402
from e_jepa_ttc.models.local_transport import (  # noqa: E402
    local_correlation_match,
    transport_physical_features,
)
from e_jepa_ttc.reproducibility import resolve_device, seed_everything  # noqa: E402
from e_jepa_ttc.training.causal_scale_eap import _targets  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4d_dinov3_temporal_delta_v1.yaml"
DEFAULT_CHECKPOINT = ROOT / "artifacts/runs/causal_scale_eap_screen_a4d_dinov3_temporal_delta_seed7/model_best.pt"
DEFAULT_CONTRACT_CONFIG = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a5_corr_v1.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _candidate_offsets(radius: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    return dx.reshape(-1), dy.reshape(-1)


def _teacher_relation_match(
    relation_t1: torch.Tensor,
    relation_t2: torch.Tensor,
    valid_t1: torch.Tensor,
    valid_t2: torch.Tensor,
    *,
    radius: int,
    temperature: float = 0.02,
    minimum_relation_count: int = 3,
) -> dict[str, torch.Tensor]:
    """Local p->q matching using the cached six-dimensional DINO relation vector."""
    if relation_t1.shape != relation_t2.shape or relation_t1.ndim != 4:
        raise ValueError("teacher relation endpoints must share [B,K,H,W]")
    if valid_t1.shape != relation_t1.shape or valid_t2.shape != relation_t2.shape:
        raise ValueError("teacher relation validity must match values")
    batch, relation_count, height, width = relation_t1.shape
    kernel = 2 * radius + 1
    candidate_count = kernel * kernel
    t2_patches = functional.unfold(
        relation_t2.float(), kernel_size=kernel, padding=radius
    ).reshape(batch, relation_count, candidate_count, height, width)
    v2_patches = functional.unfold(
        valid_t2.float(), kernel_size=kernel, padding=radius
    ).reshape(batch, relation_count, candidate_count, height, width) > 0.5
    common = valid_t1[:, :, None].bool() & v2_patches
    counts = common.sum(dim=1)
    error = ((t2_patches - relation_t1.float()[:, :, None]).abs() * common.float()).sum(dim=1)
    error = error / counts.clamp_min(1).float()
    candidate_valid = counts >= int(minimum_relation_count)
    large = torch.finfo(error.dtype).max / 1024.0
    masked_error = error.masked_fill(~candidate_valid, large)
    best_error, best_index = masked_error.min(dim=1)
    top2 = masked_error.topk(k=2, dim=1, largest=False).values
    margin = (top2[:, 1] - top2[:, 0]).clamp_min(0.0)

    logits = (-masked_error / float(temperature)).masked_fill(~candidate_valid, torch.finfo(error.dtype).min)
    probability = torch.softmax(logits, dim=1) * candidate_valid.float()
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    dx_values, dy_values = _candidate_offsets(radius, error.device, error.dtype)
    dx = (probability * dx_values[None, :, None, None]).sum(dim=1)
    dy = (probability * dy_values[None, :, None, None]).sum(dim=1)
    valid_count = candidate_valid.sum(dim=1).clamp_min(2).float()
    entropy = -(probability.clamp_min(1.0e-12).log() * probability).sum(dim=1) / valid_count.log()

    fixed_common = valid_t1.bool() & valid_t2.bool()
    fixed_count = fixed_common.sum(dim=1)
    fixed_error = ((relation_t2.float() - relation_t1.float()).abs() * fixed_common.float()).sum(dim=1)
    fixed_error = fixed_error / fixed_count.clamp_min(1).float()
    fixed_valid = fixed_count >= int(minimum_relation_count)
    local_valid = candidate_valid.any(dim=1)
    return {
        "fixed_error": fixed_error,
        "fixed_valid": fixed_valid,
        "best_error": best_error,
        "local_valid": local_valid,
        "dx": dx,
        "dy": dy,
        "margin": margin,
        "entropy": entropy.clamp(0.0, 1.0),
        "best_index": best_index,
    }



def _cycle_error_map(
    forward: dict[str, torch.Tensor],
    reverse: dict[str, torch.Tensor],
    *,
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward/reverse cycle error for teacher p->q relation matching.

    The reverse displacement is bilinearly sampled at the forward destination.
    This mirrors the event-feature cycle diagnostic and makes it harder for a
    low p->q error caused by ambiguous local matches to masquerade as transport.
    """

    dx = forward["dx"]
    dy = forward["dy"]
    reverse_dx = reverse["dx"]
    reverse_dy = reverse["dy"]
    batch, height, width = dx.shape
    dtype = dx.dtype
    device = dx.device
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    sample_grid = torch.stack(
        (
            grid_x[None].expand(batch, -1, -1)
            + (2.0 / float(max(width - 1, 1))) * dx,
            grid_y[None].expand(batch, -1, -1)
            + (2.0 / float(max(height - 1, 1))) * dy,
        ),
        dim=-1,
    )
    reverse_field = torch.stack((reverse_dx, reverse_dy), dim=1)
    sampled = functional.grid_sample(
        reverse_field,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    reverse_valid = functional.grid_sample(
        reverse["local_valid"][:, None].float(),
        sample_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0] > 0.5
    valid = forward["local_valid"] & reverse_valid
    cycle = torch.sqrt(
        (dx + sampled[:, 0]).square() + (dy + sampled[:, 1]).square() + 1.0e-12
    ) / float(max(radius, 1))
    return cycle, valid

def _feature_bbox_mask(boxes: torch.Tensor, source_h: int, source_w: int, feat_h: int, feat_w: int) -> torch.Tensor:
    scale = boxes.new_tensor([feat_w / source_w, feat_h / source_h, feat_w / source_w, feat_h / source_h])
    scaled = boxes * scale
    masks, valid = weak_box_masks(scaled[:, None], height=feat_h, width=feat_w)
    return masks[:, 0, 0].bool() & valid[:, 0, None, None]


def _mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask & torch.isfinite(values)]
    return float(selected.mean().item()) if selected.numel() else float("nan")


def _median(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask & torch.isfinite(values)]
    return float(selected.median().item()) if selected.numel() else float("nan")


def _grad_vector(loss: torch.Tensor, named_parameters: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        [p for _, p in named_parameters],
        retain_graph=True,
        allow_unused=True,
    )
    parts = [
        (torch.zeros_like(p).reshape(-1) if g is None else g.detach().float().reshape(-1))
        for (_, p), g in zip(named_parameters, grads, strict=True)
    ]
    return torch.cat(parts) if parts else loss.new_zeros(1)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    return float(torch.dot(a, b).item() / denom.item()) if float(denom) > 0 else float("nan")


def _loss_groups(
    model: CausalScaleTTC,
    batch: Any,
    loss_config: CausalScaleTTCLossConfig,
    endpoint_weight: float,
    delta_weight: float,
) -> tuple[dict[str, torch.Tensor], Any]:
    targets = _targets(batch, mask_t0_as_proxy=True, foreground_supervision="bbox_geometry")
    output = model(batch.events, targets.delta_t_s, return_dense_features=True)
    if output.endpoint_dense_features is None:
        raise RuntimeError("preflight requires dense student features")
    dense = output.endpoint_dense_features[:, 1:3]
    if batch.dinov3_relation_targets is None or batch.dinov3_relation_valid is None:
        raise RuntimeError("preflight requires train-only DINO relation targets")
    a4_raw = local_relational_distillation_loss(dense, batch.dinov3_relation_targets, batch.dinov3_relation_valid)
    a4d_raw = local_relational_temporal_delta_loss(dense, batch.dinov3_relation_targets, batch.dinov3_relation_valid)
    causal = causal_scale_ttc_loss(
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
    c = causal.components
    cfg = loss_config
    geometry = (
        cfg.foreground_bce_weight * c["foreground_bce"]
        + cfg.foreground_dice_weight * c["foreground_dice"]
        + cfg.foreground_extent_weight * c["foreground_extent"]
        + cfg.foreground_width_weight * c["foreground_width"]
        + cfg.foreground_center_weight * c["foreground_center"]
        + cfg.foreground_pair_ratio_weight * c["foreground_pair_ratio"]
    )
    ttc = causal.total - geometry
    return {
        "A4_endpoint": float(endpoint_weight) * a4_raw,
        "A4D_temporal": float(delta_weight) * a4d_raw,
        "geometry": geometry,
        "TTC": ttc,
        "A4_endpoint_raw": a4_raw,
        "A4D_temporal_raw": a4d_raw,
    }, output


def run(
    config_path: Path,
    checkpoint_path: Path,
    contract_config_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    samples: int,
    batch_size: int,
    gradient_batches: int,
    radii: tuple[int, ...],
) -> dict[str, Any]:
    if samples <= 0 or batch_size <= 0 or gradient_batches <= 0:
        raise ValueError("samples/batch_size/gradient_batches must be positive")
    if radii != (1, 2, 4):
        raise ValueError("A5 preflight is frozen to radii (1,2,4)")
    raw = _read_yaml(config_path)
    data = raw.get("data")
    training = raw.get("training")
    loss_raw = raw.get("loss")
    if not isinstance(data, dict) or not isinstance(training, dict) or not isinstance(loss_raw, dict):
        raise ValueError("A4D config lacks data/training/loss mappings")
    if data.get("official_test_opened") is not False or data.get("codabench_opened") is not False or data.get("evttc_test_opened") is not False:
        raise ValueError("preflight refuses any config that authorizes private/test access")
    if training.get("representation_supervision") != "dinov3_local_relational_temporal_delta":
        raise ValueError("preflight source must be frozen A4D")

    seed_everything(7, deterministic=True)
    device = resolve_device(device_name)
    cache_manifest = (ROOT / str(data["cache_manifest"])).resolve(strict=True)
    teacher_cfg = data.get("dinov3_relational_teacher")
    if not isinstance(teacher_cfg, dict):
        raise ValueError("A4D config lacks DINO teacher")
    teacher_manifest = (ROOT / str(teacher_cfg["manifest"])).resolve(strict=True)
    base = GarlTTCObjectEventV4Dataset(str(cache_manifest), splits=("train",))
    dataset = DINOv3RelationalTeacherDataset(
        base,
        manifest_path=teacher_manifest,
        expected_artifact_sha256=str(teacher_cfg["artifact_sha256"]),
        expected_manifest_sha256=str(teacher_cfg["manifest_sha256"]),
    )
    sample_count = min(int(samples), len(dataset))
    subset = Subset(dataset, list(range(sample_count)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_object_event_v4)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("checkpoint is not causal-scale model_best.pt")
    model_cfg_raw = dict(checkpoint["model_config"])
    if model_cfg_raw.get("transport_enabled", False):
        raise ValueError("preflight must inspect the pre-A5 A4/A4D student")
    model_cfg_raw["risk_thresholds_s"] = tuple(model_cfg_raw["risk_thresholds_s"])
    model = CausalScaleTTC(CausalScaleTTCConfig(**model_cfg_raw)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    loss_config = CausalScaleTTCLossConfig(**loss_raw)
    endpoint_weight = float(training["representation_distillation_weight"])
    delta_weight = float(training["representation_temporal_delta_weight"])

    radius_rows: dict[int, list[dict[str, float]]] = defaultdict(list)
    sample_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, float]] = []
    encoder_params = [(n, p) for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]

    for batch_index, host_batch in enumerate(loader):
        batch = host_batch.to(device)
        delta = batch.delta_t_s[:, None].expand(-1, batch.events.shape[1] - 1)
        with torch.enable_grad():
            groups, output = _loss_groups(model, batch, loss_config, endpoint_weight, delta_weight)
            if batch_index < gradient_batches:
                vectors = {
                    name: _grad_vector(groups[name], encoder_params)
                    for name in ("A4_endpoint", "A4D_temporal", "geometry", "TTC")
                }
                row: dict[str, float] = {
                    "batch": float(batch_index),
                    **{f"norm_{name}": float(vec.norm().item()) for name, vec in vectors.items()},
                    "raw_A4": float(groups["A4_endpoint_raw"].detach().cpu()),
                    "raw_A4D": float(groups["A4D_temporal_raw"].detach().cpu()),
                    "weighted_A4": float(groups["A4_endpoint"].detach().cpu()),
                    "weighted_A4D": float(groups["A4D_temporal"].detach().cpu()),
                    "geometry_loss": float(groups["geometry"].detach().cpu()),
                    "ttc_loss": float(groups["TTC"].detach().cpu()),
                }
                names = list(vectors)
                for i, first in enumerate(names):
                    for second in names[i + 1 :]:
                        row[f"cos_{first}__{second}"] = _cosine(vectors[first], vectors[second])
                gradient_rows.append(row)

        dense = output.endpoint_dense_features
        if dense is None:
            raise RuntimeError("preflight model did not expose dense features")
        student_t1 = dense[:, 1].detach()
        student_t2 = dense[:, 2].detach()
        teacher = batch.dinov3_relation_targets
        teacher_valid = batch.dinov3_relation_valid
        assert teacher is not None and teacher_valid is not None
        relation_t1, relation_t2 = teacher[:, 0], teacher[:, 1]
        valid_t1, valid_t2 = teacher_valid[:, 0], teacher_valid[:, 1]
        feat_h, feat_w = student_t1.shape[-2:]
        fg_mask = _feature_bbox_mask(
            batch.boxes_xyxy[:, 1],
            int(batch.events.shape[-2]),
            int(batch.events.shape[-1]),
            feat_h,
            feat_w,
        )

        for radius in radii:
            tm = _teacher_relation_match(relation_t1, relation_t2, valid_t1, valid_t2, radius=radius)
            tm_reverse = _teacher_relation_match(
                relation_t2, relation_t1, valid_t2, valid_t1, radius=radius
            )
            teacher_cycle, teacher_cycle_valid = _cycle_error_map(
                tm, tm_reverse, radius=radius
            )
            student_forward = local_correlation_match(student_t1, student_t2, radius=radius, temperature=0.07)
            student_reverse = local_correlation_match(student_t2, student_t1, radius=radius, temperature=0.07)
            student_summary = transport_physical_features(
                student_forward,
                student_reverse,
                foreground_weight=fg_mask[:, None].float(),
                radius=radius,
            )
            teacher_valid_pos = tm["fixed_valid"] & tm["local_valid"]
            fg = teacher_valid_pos & fg_mask
            bg = teacher_valid_pos & ~fg_mask
            fixed_global = _mean(tm["fixed_error"], teacher_valid_pos)
            best_global = _mean(tm["best_error"], teacher_valid_pos)
            fixed_fg = _mean(tm["fixed_error"], fg)
            best_fg = _mean(tm["best_error"], fg)
            fixed_bg = _mean(tm["fixed_error"], bg)
            best_bg = _mean(tm["best_error"], bg)
            reduction_global = (fixed_global - best_global) / fixed_global if fixed_global > 0 else float("nan")
            reduction_fg = (fixed_fg - best_fg) / fixed_fg if fixed_fg > 0 else float("nan")
            reduction_bg = (fixed_bg - best_bg) / fixed_bg if fixed_bg > 0 else float("nan")
            teacher_flow_mag = torch.sqrt(tm["dx"].square() + tm["dy"].square())
            student_flow_mag = torch.sqrt(student_forward.dx.square() + student_forward.dy.square())
            row = {
                "radius": float(radius),
                "teacher_fixed_error": fixed_global,
                "teacher_best_local_error": best_global,
                "teacher_error_reduction": reduction_global,
                "teacher_fixed_fg_error": fixed_fg,
                "teacher_best_fg_error": best_fg,
                "teacher_fg_error_reduction": reduction_fg,
                "teacher_fixed_bg_error": fixed_bg,
                "teacher_best_bg_error": best_bg,
                "teacher_bg_error_reduction": reduction_bg,
                "teacher_margin": _mean(tm["margin"], teacher_valid_pos),
                "teacher_entropy": _mean(tm["entropy"], teacher_valid_pos),
                "teacher_displacement": _mean(teacher_flow_mag, teacher_valid_pos),
                "teacher_cycle_error": _mean(teacher_cycle, teacher_valid_pos & teacher_cycle_valid),
                "teacher_fg_cycle_error": _mean(teacher_cycle, fg & teacher_cycle_valid),
                "teacher_bg_cycle_error": _mean(teacher_cycle, bg & teacher_cycle_valid),
                "student_margin": _mean(student_forward.confidence_margin, student_forward.valid),
                "student_entropy": _mean(student_forward.entropy, student_forward.valid),
                "student_displacement": _mean(student_flow_mag, student_forward.valid),
                "student_cycle_error": float(student_summary[:, 8].mean().item()),
                "student_fg_cycle_error": float(student_summary[:, 17].mean().item()),
                "foreground_fraction": float(fg_mask.float().mean().item()),
                "zero_student_delta_error": fixed_global,
                "zero_student_delta_baseline": fixed_global,
            }
            radius_rows[radius].append(row)

            # Per-sample compact diagnostics for offline inspection.
            for local_index, token in enumerate(batch.sample_tokens):
                valid_row = teacher_valid_pos[local_index]
                sample_rows.append(
                    {
                        "sample_token": token,
                        "sequence_id": batch.sequence_ids[local_index],
                        "radius": radius,
                        "teacher_fixed_error": _mean(tm["fixed_error"][local_index], valid_row),
                        "teacher_best_local_error": _mean(tm["best_error"][local_index], valid_row),
                        "teacher_displacement": _mean(teacher_flow_mag[local_index], valid_row),
                        "teacher_entropy": _mean(tm["entropy"][local_index], valid_row),
                        "teacher_cycle_error": _mean(
                            teacher_cycle[local_index],
                            valid_row & teacher_cycle_valid[local_index],
                        ),
                        "student_displacement": _mean(student_flow_mag[local_index], student_forward.valid[local_index]),
                        "student_entropy": _mean(student_forward.entropy[local_index], student_forward.valid[local_index]),
                        "student_confidence_margin": _mean(student_forward.confidence_margin[local_index], student_forward.valid[local_index]),
                        "student_divergence_y": float(student_summary[local_index, 3].item()),
                        "student_divergence_isotropic": float(student_summary[local_index, 4].item()),
                        "student_fg_divergence_y": float(student_summary[local_index, 12].item()),
                        "student_fg_divergence_isotropic": float(student_summary[local_index, 13].item()),
                    }
                )

    aggregated: dict[str, Any] = {}
    for radius in radii:
        rows = radius_rows[radius]
        keys = [key for key in rows[0] if key != "radius"]
        aggregated[str(radius)] = {
            key: float(np.nanmean([float(row[key]) for row in rows])) for key in keys
        }

    grad_summary: dict[str, Any] = {}
    if gradient_rows:
        for key in gradient_rows[0]:
            if key == "batch":
                continue
            values = np.asarray([float(row[key]) for row in gradient_rows], dtype=np.float64)
            grad_summary[key] = float(np.nanmean(values))

    r4 = aggregated["4"]
    contract_raw = _read_yaml(contract_config_path)
    decision_contract = contract_raw.get("decision_contract")
    if not isinstance(decision_contract, dict):
        raise ValueError("A5 contract config lacks decision_contract")
    preflight_contract = decision_contract.get("preflight_contract")
    if not isinstance(preflight_contract, dict):
        raise ValueError("A5 contract config lacks preflight_contract")
    if list(preflight_contract.get("radii", [])) != list(radii):
        raise ValueError("runtime radii differ from preregistered A5 radii")
    thresholds = {
        "teacher_r4_global_error_reduction_min": float(preflight_contract["teacher_r4_global_error_reduction_min"]),
        "teacher_r4_foreground_error_reduction_min": float(preflight_contract["teacher_r4_foreground_error_reduction_min"]),
        "student_r4_entropy_max": float(preflight_contract["student_r4_entropy_max"]),
        "student_r4_confidence_margin_min": float(preflight_contract["student_r4_confidence_margin_min"]),
    }
    checks = {
        "teacher_global_transport": float(r4["teacher_error_reduction"]) >= thresholds["teacher_r4_global_error_reduction_min"],
        "teacher_foreground_transport": float(r4["teacher_fg_error_reduction"]) >= thresholds["teacher_r4_foreground_error_reduction_min"],
        "student_entropy": float(r4["student_entropy"]) <= thresholds["student_r4_entropy_max"],
        "student_confidence": float(r4["student_margin"]) >= thresholds["student_r4_confidence_margin_min"],
    }
    transport_supported = checks["teacher_global_transport"] and checks["teacher_foreground_transport"]
    student_match_supported = checks["student_entropy"] and checks["student_confidence"]
    authorized = transport_supported and student_match_supported

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_csv = output_dir / "a5_transport_preflight_samples.csv"
    gradients_csv = output_dir / "a5_transport_preflight_gradients.csv"
    pd.DataFrame(sample_rows).to_csv(samples_csv, index=False, lineterminator="\n")
    pd.DataFrame(gradient_rows).to_csv(gradients_csv, index=False, lineterminator="\n")
    payload: dict[str, Any] = {
        "artifact_type": "a5_transport_preflight_train_only_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "public_train_only": True,
            "validation_or_test_opened": False,
            "optimizer_steps": 0,
            "samples": sample_count,
            "radii": list(radii),
        },
        "source": {
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256(config_path),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "teacher_manifest": str(teacher_manifest.relative_to(ROOT)),
            "teacher_manifest_sha256": _sha256(teacher_manifest),
            "model_config": asdict(model.config),
            "a5_contract_config": str(contract_config_path.relative_to(ROOT)),
            "a5_contract_config_sha256": _sha256(contract_config_path),
        },
        "teacher_transport": aggregated,
        "gradient_interactions": grad_summary,
        "gradient_batches": len(gradient_rows),
        "decision_thresholds": thresholds,
        "decision_checks": checks,
        "transport_supported": transport_supported,
        "student_match_supported": student_match_supported,
        "a5_corr_authorized": authorized,
        "interpretation_contract": {
            "radii_are_structural_diagnostics_not_validation_selection": True,
            "zero_student_delta_equals_fixed_teacher_delta_baseline": True,
            "bbox_used_only_for_train_only_fg_bg_diagnostic": True,
            "bbox_not_used_by_a5_transport_model": True,
            "A4D_gradient_is_diagnostic_not_a5_training_loss": True,
        },
        "files": {
            "samples_csv": samples_csv.name,
            "samples_csv_sha256": _sha256(samples_csv),
            "gradients_csv": gradients_csv.name,
            "gradients_csv_sha256": _sha256(gradients_csv),
        },
    }
    sign_artifact(payload)
    result_path = output_dir / "a5_transport_preflight.json"
    _atomic_json(result_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--contract-config", type=Path, default=DEFAULT_CONTRACT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-batches", type=int, default=8)
    args = parser.parse_args()
    payload = run(
        args.config.resolve(),
        args.checkpoint.resolve(),
        args.contract_config.resolve(),
        args.output_dir.resolve(),
        device_name=args.device,
        samples=args.samples,
        batch_size=args.batch_size,
        gradient_batches=args.gradient_batches,
        radii=(1, 2, 4),
    )
    print(json.dumps({
        "a5_corr_authorized": payload["a5_corr_authorized"],
        "transport_supported": payload["transport_supported"],
        "student_match_supported": payload["student_match_supported"],
        "r4": payload["teacher_transport"]["4"],
    }, sort_keys=True))
    return 0 if payload["a5_corr_authorized"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
