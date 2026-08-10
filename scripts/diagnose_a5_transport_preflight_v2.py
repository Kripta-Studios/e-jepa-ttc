#!/usr/bin/env python
"""A5-PREFLIGHT-V2: zero-training, train-only transport falsification audit.

V1 established that a local DINO-relation search lowers p->p error but left four
ambiguities: best-of-K bias, temperature dependence, physical correctness, and
student representation quality.  V2 resolves them before any A5 training by:

* comparing real temporal pairs with same-sequence shuffled pairs at identical K;
* auditing A4, A4D and deterministic random-init student features;
* separating hard-argmax evidence from softmax temperature calibration;
* comparing hard/soft transport to bbox-implied affine translation/scale; and
* selecting the *smallest* train-only radius that has >= frozen physical coverage
  and survives the shuffled-pair teacher null.

Validation/test are never opened and no optimizer step is taken.
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
from typing import Any, Iterable

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
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig  # noqa: E402
from e_jepa_ttc.reproducibility import resolve_device, seed_everything  # noqa: E402

DEFAULT_SOURCE_CONFIG = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_a4d_dinov3_temporal_delta_v1.yaml"
DEFAULT_PROTOCOL = ROOT / "configs/experiment/e_jepa_garl_event_causal_scale_a5_preflight_v2.yaml"
DEFAULT_A4 = ROOT / "artifacts/runs/causal_scale_eap_screen_a4_dinov3_relational_rgb_v2_seed7/model_best.pt"
DEFAULT_A4D = ROOT / "artifacts/runs/causal_scale_eap_screen_a4d_dinov3_temporal_delta_seed7/model_best.pt"


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _pearson(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=np.float64)
    b = np.asarray(list(y), dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3 or float(a.std()) <= 1.0e-12 or float(b.std()) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _selected_indices(dataset_size: int, sample_count: int) -> list[int]:
    if dataset_size <= 0 or sample_count <= 0:
        raise ValueError("dataset_size and sample_count must be positive")
    count = min(dataset_size, sample_count)
    # Deterministically cover the complete cache rather than taking the first N.
    return [int((i * dataset_size) // count) for i in range(count)]


def _same_sequence_null_partners(
    sequence_ids: list[str],
    track_ids: list[str],
    original_indices: list[int],
) -> tuple[np.ndarray, float]:
    """Build a deterministic same-sequence derangement, preferring other tracks."""

    if not (len(sequence_ids) == len(track_ids) == len(original_indices)):
        raise ValueError("metadata arrays must have identical length")
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for pos, sequence in enumerate(sequence_ids):
        by_sequence[str(sequence)].append(pos)
    partner = np.full(len(sequence_ids), -1, dtype=np.int64)
    different_track = 0
    valid = 0
    for positions in by_sequence.values():
        if len(positions) < 2:
            continue
        for pos in positions:
            preferred = [p for p in positions if p != pos and track_ids[p] != track_ids[pos]]
            candidates = preferred or [p for p in positions if p != pos]
            # Maximise dataset-index separation to avoid a nearly adjacent pseudo-null.
            chosen = max(
                candidates,
                key=lambda p: (abs(original_indices[p] - original_indices[pos]), -original_indices[p]),
            )
            partner[pos] = chosen
            valid += 1
            different_track += int(track_ids[chosen] != track_ids[pos])
    return partner, float(different_track / valid) if valid else 0.0


def _candidate_offsets(radius: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(values, values, indexing="ij")
    return dx.reshape(-1), dy.reshape(-1)


def _teacher_match(
    previous: torch.Tensor,
    current: torch.Tensor,
    previous_valid: torch.Tensor,
    current_valid: torch.Tensor,
    *,
    radius: int,
    minimum_relation_count: int = 3,
) -> dict[str, torch.Tensor]:
    if previous.shape != current.shape or previous.ndim != 4:
        raise ValueError("teacher tensors must share [B,K,H,W]")
    batch, relation_count, height, width = previous.shape
    kernel = 2 * radius + 1
    candidate_count = kernel * kernel
    patches = functional.unfold(current.float(), kernel_size=kernel, padding=radius).reshape(
        batch, relation_count, candidate_count, height, width
    )
    valid_patches = functional.unfold(current_valid.float(), kernel_size=kernel, padding=radius).reshape(
        batch, relation_count, candidate_count, height, width
    ) > 0.5
    common = previous_valid[:, :, None].bool() & valid_patches
    counts = common.sum(dim=1)
    errors = ((patches - previous.float()[:, :, None]).abs() * common.float()).sum(dim=1)
    errors = errors / counts.clamp_min(1).float()
    candidate_valid = counts >= int(minimum_relation_count)
    large = torch.finfo(errors.dtype).max / 1024.0
    masked = errors.masked_fill(~candidate_valid, large)
    best_error, best_index = masked.min(dim=1)
    top2 = masked.topk(k=2, dim=1, largest=False).values
    margin = (top2[:, 1] - top2[:, 0]).clamp_min(0.0)
    fixed_common = previous_valid.bool() & current_valid.bool()
    fixed_count = fixed_common.sum(dim=1)
    fixed_error = ((current.float() - previous.float()).abs() * fixed_common.float()).sum(dim=1)
    fixed_error = fixed_error / fixed_count.clamp_min(1).float()
    fixed_valid = fixed_count >= int(minimum_relation_count)
    local_valid = candidate_valid.any(dim=1)
    dx_values, dy_values = _candidate_offsets(radius, errors.device, errors.dtype)
    hard_dx = dx_values[best_index]
    hard_dy = dy_values[best_index]
    boundary = (hard_dx.abs() == radius) | (hard_dy.abs() == radius)
    return {
        "fixed_error": fixed_error,
        "best_error": best_error,
        "fixed_valid": fixed_valid,
        "local_valid": local_valid,
        "margin": margin,
        "hard_dx": hard_dx,
        "hard_dy": hard_dy,
        "boundary": boundary,
    }


def _student_correlation(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    radius: int,
    temperatures: tuple[float, ...],
) -> dict[str, Any]:
    if previous.shape != current.shape or previous.ndim != 4:
        raise ValueError("student tensors must share [B,C,H,W]")
    batch, _channels, height, width = previous.shape
    prev = functional.normalize(previous.float(), dim=1, eps=1.0e-6)
    cur = functional.normalize(current.float(), dim=1, eps=1.0e-6)
    padded = functional.pad(cur, (radius, radius, radius, radius))
    ys = torch.arange(height, device=cur.device)
    xs = torch.arange(width, device=cur.device)
    corr: list[torch.Tensor] = []
    valid_maps: list[torch.Tensor] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = padded[:, :, radius + dy : radius + dy + height, radius + dx : radius + dx + width]
            corr.append((prev * shifted).sum(dim=1))
            vy = (ys + dy >= 0) & (ys + dy < height)
            vx = (xs + dx >= 0) & (xs + dx < width)
            valid_maps.append((vy[:, None] & vx[None, :])[None].expand(batch, -1, -1))
    correlation = torch.stack(corr, dim=1)
    candidate_valid = torch.stack(valid_maps, dim=1)
    masked = correlation.masked_fill(~candidate_valid, torch.finfo(correlation.dtype).min)
    top2, top2_index = masked.topk(k=2, dim=1)
    hard_index = top2_index[:, 0]
    dx_values, dy_values = _candidate_offsets(radius, correlation.device, correlation.dtype)
    hard_dx = dx_values[hard_index]
    hard_dy = dy_values[hard_index]
    margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
    count = candidate_valid.sum(dim=1).float()
    masked_zero = correlation.masked_fill(~candidate_valid, 0.0)
    mean = masked_zero.sum(dim=1) / count.clamp_min(1.0)
    variance = (((correlation - mean[:, None]) ** 2) * candidate_valid.float()).sum(dim=1) / count.clamp_min(1.0)
    zscore = (top2[:, 0] - mean) / variance.sqrt().clamp_min(1.0e-6)
    zero_index = radius * (2 * radius + 1) + radius
    zero_corr = correlation[:, zero_index]
    zero_rank = 1 + ((masked > zero_corr[:, None]) & candidate_valid).sum(dim=1)
    boundary = (hard_dx.abs() == radius) | (hard_dy.abs() == radius)

    soft: dict[float, dict[str, torch.Tensor]] = {}
    for temperature in temperatures:
        probability = torch.softmax(masked / float(temperature), dim=1) * candidate_valid.float()
        probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        expected_dx = (probability * dx_values[None, :, None, None]).sum(dim=1)
        expected_dy = (probability * dy_values[None, :, None, None]).sum(dim=1)
        raw_entropy = -(probability.clamp_min(1.0e-12).log() * probability).sum(dim=1)
        norm_entropy = raw_entropy / count.clamp_min(2.0).log()
        effective = raw_entropy.exp()
        top5_mass = probability.topk(k=min(5, probability.shape[1]), dim=1).values.sum(dim=1)
        soft[float(temperature)] = {
            "dx": expected_dx,
            "dy": expected_dy,
            "entropy": norm_entropy.clamp(0.0, 1.0),
            "effective_candidates": effective,
            "effective_fraction": effective / count.clamp_min(1.0),
            "top1_probability": probability.max(dim=1).values,
            "top5_probability_mass": top5_mass,
        }
    return {
        "top1": top2[:, 0],
        "top2": top2[:, 1],
        "margin": margin,
        "zscore": zscore,
        "zero_rank": zero_rank.float(),
        "hard_dx": hard_dx,
        "hard_dy": hard_dy,
        "boundary": boundary,
        "valid": candidate_valid.any(dim=1),
        "soft": soft,
    }


def _bbox_affine_flow(
    boxes_t1: torch.Tensor,
    boxes_t2: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    feature_height: int,
    feature_width: int,
) -> dict[str, torch.Tensor]:
    """BBox-implied affine flow on the dense feature grid, diagnostics only."""

    scale = boxes_t1.new_tensor([
        feature_width / float(source_width),
        feature_height / float(source_height),
        feature_width / float(source_width),
        feature_height / float(source_height),
    ])
    b1 = boxes_t1 * scale
    b2 = boxes_t2 * scale
    w1 = (b1[:, 2] - b1[:, 0]).clamp_min(1.0e-6)
    h1 = (b1[:, 3] - b1[:, 1]).clamp_min(1.0e-6)
    w2 = (b2[:, 2] - b2[:, 0]).clamp_min(1.0e-6)
    h2 = (b2[:, 3] - b2[:, 1]).clamp_min(1.0e-6)
    cx1, cy1 = 0.5 * (b1[:, 0] + b1[:, 2]), 0.5 * (b1[:, 1] + b1[:, 3])
    cx2, cy2 = 0.5 * (b2[:, 0] + b2[:, 2]), 0.5 * (b2[:, 1] + b2[:, 3])
    ys = torch.arange(feature_height, device=b1.device, dtype=b1.dtype) + 0.5
    xs = torch.arange(feature_width, device=b1.device, dtype=b1.dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx[None]
    yy = yy[None]
    sx = (w2 / w1)[:, None, None]
    sy = (h2 / h1)[:, None, None]
    mapped_x = cx2[:, None, None] + sx * (xx - cx1[:, None, None])
    mapped_y = cy2[:, None, None] + sy * (yy - cy1[:, None, None])
    dx = mapped_x - xx
    dy = mapped_y - yy
    fg = (
        (xx >= b1[:, 0, None, None])
        & (xx < b1[:, 2, None, None])
        & (yy >= b1[:, 1, None, None])
        & (yy < b1[:, 3, None, None])
    )
    return {
        "dx": dx,
        "dy": dy,
        "foreground": fg,
        "translation_x": cx2 - cx1,
        "translation_y": cy2 - cy1,
        "log_width_ratio": torch.log(w2 / w1),
        "log_height_ratio": torch.log(h2 / h1),
        "log_isotropic_ratio": 0.5 * (torch.log(w2 / w1) + torch.log(h2 / h1)),
    }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = value[mask & torch.isfinite(value)]
    return float(selected.mean().item()) if selected.numel() else float("nan")


def _hard_cycle_error(forward: dict[str, Any], reverse: dict[str, Any]) -> torch.Tensor:
    dx = forward["hard_dx"].long()
    dy = forward["hard_dy"].long()
    rdx = reverse["hard_dx"]
    rdy = reverse["hard_dy"]
    batch, height, width = dx.shape
    y = torch.arange(height, device=dx.device)[None, :, None].expand(batch, height, width)
    x = torch.arange(width, device=dx.device)[None, None, :].expand(batch, height, width)
    qy = (y + dy).clamp(0, height - 1)
    qx = (x + dx).clamp(0, width - 1)
    batch_index = torch.arange(batch, device=dx.device)[:, None, None]
    back_dx = rdx[batch_index, qy, qx]
    back_dy = rdy[batch_index, qy, qx]
    return torch.sqrt((forward["hard_dx"] + back_dx).square() + (forward["hard_dy"] + back_dy).square() + 1.0e-12)


def _physics_metrics(
    dx: torch.Tensor,
    dy: torch.Tensor,
    physical: dict[str, torch.Tensor],
) -> dict[str, float]:
    fg = physical["foreground"]
    epe = torch.sqrt((dx - physical["dx"]).square() + (dy - physical["dy"]).square() + 1.0e-12)
    zero = torch.sqrt(physical["dx"].square() + physical["dy"].square() + 1.0e-12)
    mean_epe = _masked_mean(epe, fg)
    zero_epe = _masked_mean(zero, fg)
    improvement = (zero_epe - mean_epe) / zero_epe if zero_epe > 1.0e-8 else float("nan")
    return {
        "physical_epe": mean_epe,
        "zero_flow_epe": zero_epe,
        "physical_epe_improvement_over_zero": improvement,
    }


def _flow_scalar_summary(dx: torch.Tensor, dy: torch.Tensor, physical: dict[str, torch.Tensor]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    fg = physical["foreground"].float()
    batch, height, width = dx.shape
    ys = torch.linspace(-0.5, 0.5, height, device=dx.device, dtype=dx.dtype)
    xs = torch.linspace(-0.5, 0.5, width, device=dx.device, dtype=dx.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    for i in range(batch):
        weight = fg[i]
        denom = weight.sum().clamp_min(1.0)
        tx = (dx[i] * weight).sum() / denom
        ty = (dy[i] * weight).sum() / denom
        xmean = (xx * weight).sum() / denom
        ymean = (yy * weight).sum() / denom
        xcenter = xx - xmean
        ycenter = yy - ymean
        dxmean = tx
        dymean = ty
        slope_x = (xcenter * (dx[i] - dxmean) * weight).sum() / ((xcenter.square() * weight).sum().clamp_min(1.0e-6))
        slope_y = (ycenter * (dy[i] - dymean) * weight).sum() / ((ycenter.square() * weight).sum().clamp_min(1.0e-6))
        rows.append({
            "flow_translation_x": float(tx.item()),
            "flow_translation_y": float(ty.item()),
            "flow_divergence_x": float(slope_x.item()),
            "flow_divergence_y": float(slope_y.item()),
            "bbox_translation_x": float(physical["translation_x"][i].item()),
            "bbox_translation_y": float(physical["translation_y"][i].item()),
            "bbox_log_width_ratio": float(physical["log_width_ratio"][i].item()),
            "bbox_log_height_ratio": float(physical["log_height_ratio"][i].item()),
            "bbox_log_isotropic_ratio": float(physical["log_isotropic_ratio"][i].item()),
        })
    return rows


def _load_checkpoint_model(path: Path, device: torch.device) -> tuple[CausalScaleTTC, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError(f"not a causal-scale model_best.pt: {path}")
    raw = dict(checkpoint["model_config"])
    if raw.get("transport_enabled", False):
        raise ValueError("V2 only audits pre-A5 students")
    raw["risk_thresholds_s"] = tuple(raw["risk_thresholds_s"])
    model = CausalScaleTTC(CausalScaleTTCConfig(**raw)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _random_model_like(checkpoint: dict[str, Any], device: torch.device, seed: int) -> CausalScaleTTC:
    seed_everything(seed, deterministic=True)
    raw = dict(checkpoint["model_config"])
    raw["risk_thresholds_s"] = tuple(raw["risk_thresholds_s"])
    model = CausalScaleTTC(CausalScaleTTCConfig(**raw)).to(device)
    model.eval()
    return model


def _collect_metadata_and_teacher(
    loader: DataLoader,
) -> dict[str, Any]:
    targets, valid, boxes = [], [], []
    sequences: list[str] = []
    tracks: list[str] = []
    tokens: list[str] = []
    source_shape: tuple[int, int] | None = None
    for batch in loader:
        if batch.dinov3_relation_targets is None or batch.dinov3_relation_valid is None:
            raise RuntimeError("V2 requires the train-only DINO teacher")
        targets.append(batch.dinov3_relation_targets.cpu())
        valid.append(batch.dinov3_relation_valid.cpu())
        boxes.append(batch.boxes_xyxy.cpu())
        sequences.extend(batch.sequence_ids)
        tracks.extend(batch.track_ids)
        tokens.extend(batch.sample_tokens)
        source_shape = (int(batch.events.shape[-2]), int(batch.events.shape[-1]))
    if source_shape is None:
        raise RuntimeError("empty V2 loader")
    return {
        "teacher": torch.cat(targets),
        "teacher_valid": torch.cat(valid),
        "boxes": torch.cat(boxes),
        "sequence_ids": sequences,
        "track_ids": tracks,
        "sample_tokens": tokens,
        "source_shape": source_shape,
    }


def _collect_student_features(model: CausalScaleTTC, loader: DataLoader, device: torch.device) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for host in loader:
            batch = host.to(device)
            delta = batch.delta_t_s[:, None].expand(-1, batch.events.shape[1] - 1)
            output = model(batch.events, delta, return_dense_features=True)
            if output.endpoint_dense_features is None:
                raise RuntimeError("model did not return dense endpoint features")
            # Keep only t1/t2 in fp16 on CPU to bound V2 memory.
            rows.append(output.endpoint_dense_features[:, 1:3].detach().cpu().to(torch.float16))
    return torch.cat(rows)


def _iter_chunks(size: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, size, batch_size):
        yield slice(start, min(size, start + batch_size))


def _teacher_audit(
    bank: dict[str, Any],
    partner: np.ndarray,
    *,
    device: torch.device,
    radii: tuple[int, ...],
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    teacher = bank["teacher"]
    valid = bank["teacher_valid"]
    boxes = bank["boxes"]
    source_h, source_w = bank["source_shape"]
    summary: dict[str, Any] = {}
    per_sample: list[dict[str, Any]] = []
    for radius in radii:
        aggregate: dict[str, list[float]] = defaultdict(list)
        scalar_rows: list[dict[str, float]] = []
        for sl in _iter_chunks(len(teacher), batch_size):
            idx = np.arange(sl.start, sl.stop)
            partner_idx = partner[idx]
            keep_np = partner_idx >= 0
            if not bool(keep_np.any()):
                continue
            idx = idx[keep_np]
            partner_idx = partner_idx[keep_np]
            t1 = teacher[idx, 0].to(device)
            t2 = teacher[idx, 1].to(device)
            v1 = valid[idx, 0].to(device)
            v2 = valid[idx, 1].to(device)
            shuffled_t2 = teacher[partner_idx, 1].to(device)
            shuffled_v2 = valid[partner_idx, 1].to(device)
            spatial_t2 = torch.roll(t2, shifts=(t2.shape[-2] // 2, t2.shape[-1] // 2), dims=(-2, -1))
            spatial_v2 = torch.roll(v2, shifts=(v2.shape[-2] // 2, v2.shape[-1] // 2), dims=(-2, -1))
            real = _teacher_match(t1, t2, v1, v2, radius=radius)
            shuffled = _teacher_match(t1, shuffled_t2, v1, shuffled_v2, radius=radius)
            spatial_null = _teacher_match(t1, spatial_t2, v1, spatial_v2, radius=radius)
            physical = _bbox_affine_flow(
                boxes[idx, 1].to(device), boxes[idx, 2].to(device),
                source_height=source_h, source_width=source_w,
                feature_height=t1.shape[-2], feature_width=t1.shape[-1],
            )
            fg = physical["foreground"] & real["fixed_valid"] & real["local_valid"]
            global_mask = real["fixed_valid"] & real["local_valid"]
            shuffled_mask = shuffled["fixed_valid"] & shuffled["local_valid"]
            spatial_mask = spatial_null["fixed_valid"] & spatial_null["local_valid"]
            real_fixed = _masked_mean(real["fixed_error"], global_mask)
            real_best = _masked_mean(real["best_error"], global_mask)
            shuffled_fixed = _masked_mean(shuffled["fixed_error"], shuffled_mask)
            shuffled_best = _masked_mean(shuffled["best_error"], shuffled_mask)
            spatial_fixed = _masked_mean(spatial_null["fixed_error"], spatial_mask)
            spatial_best = _masked_mean(spatial_null["best_error"], spatial_mask)
            real_reduction = (real_fixed - real_best) / real_fixed if real_fixed > 0 else float("nan")
            shuffled_reduction = (shuffled_fixed - shuffled_best) / shuffled_fixed if shuffled_fixed > 0 else float("nan")
            spatial_reduction = (spatial_fixed - spatial_best) / spatial_fixed if spatial_fixed > 0 else float("nan")
            real_fg_fixed = _masked_mean(real["fixed_error"], fg)
            real_fg_best = _masked_mean(real["best_error"], fg)
            shuffled_fg = physical["foreground"] & shuffled["fixed_valid"] & shuffled["local_valid"]
            shuffled_fg_fixed = _masked_mean(shuffled["fixed_error"], shuffled_fg)
            shuffled_fg_best = _masked_mean(shuffled["best_error"], shuffled_fg)
            real_fg_reduction = (real_fg_fixed - real_fg_best) / real_fg_fixed if real_fg_fixed > 0 else float("nan")
            shuffled_fg_reduction = (shuffled_fg_fixed - shuffled_fg_best) / shuffled_fg_fixed if shuffled_fg_fixed > 0 else float("nan")
            physics = _physics_metrics(real["hard_dx"], real["hard_dy"], physical)
            magnitude = torch.sqrt(physical["dx"].square() + physical["dy"].square())
            coverage = _masked_mean((magnitude <= radius).float(), physical["foreground"])
            values = {
                "real_fixed_error": real_fixed,
                "real_best_error": real_best,
                "real_error_reduction": real_reduction,
                "shuffled_fixed_error": shuffled_fixed,
                "shuffled_best_error": shuffled_best,
                "shuffled_error_reduction": shuffled_reduction,
                "spatial_null_fixed_error": spatial_fixed,
                "spatial_null_best_error": spatial_best,
                "spatial_null_error_reduction": spatial_reduction,
                "excess_error_reduction": real_reduction - shuffled_reduction,
                "excess_error_reduction_vs_spatial_null": real_reduction - spatial_reduction,
                "real_vs_shuffled_best_error_improvement": (shuffled_best - real_best) / shuffled_best if shuffled_best > 0 else float("nan"),
                "real_vs_spatial_null_best_error_improvement": (spatial_best - real_best) / spatial_best if spatial_best > 0 else float("nan"),
                "real_fg_error_reduction": real_fg_reduction,
                "shuffled_fg_error_reduction": shuffled_fg_reduction,
                "foreground_excess_error_reduction": real_fg_reduction - shuffled_fg_reduction,
                "margin": _masked_mean(real["margin"], global_mask),
                "hard_boundary_fraction": _masked_mean(real["boundary"].float(), global_mask),
                "bbox_physical_coverage": coverage,
                **physics,
            }
            for key, value in values.items():
                aggregate[key].append(value)
            scalar_rows.extend(_flow_scalar_summary(real["hard_dx"], real["hard_dy"], physical))
            for local, global_idx in enumerate(idx.tolist()):
                per_sample.append({
                    "kind": "teacher",
                    "model": "dinov3_relations",
                    "radius": radius,
                    "sample_token": bank["sample_tokens"][global_idx],
                    "sequence_id": bank["sequence_ids"][global_idx],
                    "partner_sample_token": bank["sample_tokens"][int(partner[global_idx])],
                    "real_best_error": float(real["best_error"][local][real["local_valid"][local]].mean().item()),
                    "shuffled_best_error": float(shuffled["best_error"][local][shuffled["local_valid"][local]].mean().item()),
                    "spatial_null_best_error": float(spatial_null["best_error"][local][spatial_null["local_valid"][local]].mean().item()),
                })
        row = {key: _finite_mean(values) for key, values in aggregate.items()}
        for flow_name, bbox_name, output_name in (
            ("flow_translation_x", "bbox_translation_x", "translation_x_pearson"),
            ("flow_translation_y", "bbox_translation_y", "translation_y_pearson"),
            ("flow_divergence_x", "bbox_log_width_ratio", "divergence_x_vs_log_width_pearson"),
            ("flow_divergence_y", "bbox_log_height_ratio", "divergence_y_vs_log_height_pearson"),
        ):
            row[output_name] = _pearson([r[flow_name] for r in scalar_rows], [r[bbox_name] for r in scalar_rows])
        summary[str(radius)] = row
    return summary, per_sample


def _student_audit(
    model_name: str,
    features: torch.Tensor,
    bank: dict[str, Any],
    partner: np.ndarray,
    *,
    device: torch.device,
    radii: tuple[int, ...],
    temperatures: tuple[float, ...],
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    boxes = bank["boxes"]
    source_h, source_w = bank["source_shape"]
    radius_summary: dict[str, Any] = {}
    temperature_rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    for radius in radii:
        hard_aggregate: dict[str, list[float]] = defaultdict(list)
        temp_aggregate: dict[float, dict[str, list[float]]] = {
            tau: defaultdict(list) for tau in temperatures
        }
        scalar_rows: list[dict[str, float]] = []
        for sl in _iter_chunks(len(features), batch_size):
            idx = np.arange(sl.start, sl.stop)
            partner_idx = partner[idx]
            keep_np = partner_idx >= 0
            if not bool(keep_np.any()):
                continue
            idx = idx[keep_np]
            partner_idx = partner_idx[keep_np]
            t1 = features[idx, 0].to(device=device, dtype=torch.float32)
            t2 = features[idx, 1].to(device=device, dtype=torch.float32)
            shuffled_t2 = features[partner_idx, 1].to(device=device, dtype=torch.float32)
            spatial_t2 = torch.roll(t2, shifts=(t2.shape[-2] // 2, t2.shape[-1] // 2), dims=(-2, -1))
            real = _student_correlation(t1, t2, radius=radius, temperatures=temperatures)
            shuffled = _student_correlation(t1, shuffled_t2, radius=radius, temperatures=temperatures)
            spatial_null = _student_correlation(t1, spatial_t2, radius=radius, temperatures=temperatures)
            reverse = _student_correlation(t2, t1, radius=radius, temperatures=temperatures)
            physical = _bbox_affine_flow(
                boxes[idx, 1].to(device), boxes[idx, 2].to(device),
                source_height=source_h, source_width=source_w,
                feature_height=t1.shape[-2], feature_width=t1.shape[-1],
            )
            fg = physical["foreground"] & real["valid"]
            hard_physics = _physics_metrics(real["hard_dx"], real["hard_dy"], physical)
            cycle = _hard_cycle_error(real, reverse)
            values = {
                "real_top1_cosine": _masked_mean(real["top1"], real["valid"]),
                "shuffled_top1_cosine": _masked_mean(shuffled["top1"], shuffled["valid"]),
                "real_minus_shuffled_top1_cosine": _masked_mean(real["top1"], real["valid"]) - _masked_mean(shuffled["top1"], shuffled["valid"]),
                "spatial_null_top1_cosine": _masked_mean(spatial_null["top1"], spatial_null["valid"]),
                "real_minus_spatial_null_top1_cosine": _masked_mean(real["top1"], real["valid"]) - _masked_mean(spatial_null["top1"], spatial_null["valid"]),
                "real_margin": _masked_mean(real["margin"], real["valid"]),
                "shuffled_margin": _masked_mean(shuffled["margin"], shuffled["valid"]),
                "real_zscore": _masked_mean(real["zscore"], real["valid"]),
                "shuffled_zscore": _masked_mean(shuffled["zscore"], shuffled["valid"]),
                "zero_offset_rank": _masked_mean(real["zero_rank"], real["valid"]),
                "hard_boundary_fraction": _masked_mean(real["boundary"].float(), real["valid"]),
                "hard_cycle_error_cells": _masked_mean(cycle, real["valid"]),
                **hard_physics,
            }
            for key, value in values.items():
                hard_aggregate[key].append(value)
            scalar_rows.extend(_flow_scalar_summary(real["hard_dx"], real["hard_dy"], physical))
            for tau in temperatures:
                soft = real["soft"][tau]
                shuffled_soft = shuffled["soft"][tau]
                soft_physics = _physics_metrics(soft["dx"], soft["dy"], physical)
                tvals = {
                    "entropy": _masked_mean(soft["entropy"], real["valid"]),
                    "shuffled_entropy": _masked_mean(shuffled_soft["entropy"], shuffled["valid"]),
                    "effective_candidates": _masked_mean(soft["effective_candidates"], real["valid"]),
                    "effective_fraction": _masked_mean(soft["effective_fraction"], real["valid"]),
                    "top1_probability": _masked_mean(soft["top1_probability"], real["valid"]),
                    "top5_probability_mass": _masked_mean(soft["top5_probability_mass"], real["valid"]),
                    **{f"soft_{k}": v for k, v in soft_physics.items()},
                }
                for key, value in tvals.items():
                    temp_aggregate[tau][key].append(value)
            for local, global_idx in enumerate(idx.tolist()):
                per_sample.append({
                    "kind": "student",
                    "model": model_name,
                    "radius": radius,
                    "sample_token": bank["sample_tokens"][global_idx],
                    "sequence_id": bank["sequence_ids"][global_idx],
                    "partner_sample_token": bank["sample_tokens"][int(partner[global_idx])],
                    "real_top1_cosine": float(real["top1"][local][real["valid"][local]].mean().item()),
                    "shuffled_top1_cosine": float(shuffled["top1"][local][shuffled["valid"][local]].mean().item()),
                    "spatial_null_top1_cosine": float(spatial_null["top1"][local][spatial_null["valid"][local]].mean().item()),
                })
        row = {key: _finite_mean(values) for key, values in hard_aggregate.items()}
        for flow_name, bbox_name, output_name in (
            ("flow_translation_x", "bbox_translation_x", "translation_x_pearson"),
            ("flow_translation_y", "bbox_translation_y", "translation_y_pearson"),
            ("flow_divergence_x", "bbox_log_width_ratio", "divergence_x_vs_log_width_pearson"),
            ("flow_divergence_y", "bbox_log_height_ratio", "divergence_y_vs_log_height_pearson"),
        ):
            row[output_name] = _pearson([r[flow_name] for r in scalar_rows], [r[bbox_name] for r in scalar_rows])
        radius_summary[str(radius)] = row
        for tau in temperatures:
            temperature_rows.append({
                "model": model_name,
                "radius": radius,
                "temperature": tau,
                **{key: _finite_mean(values) for key, values in temp_aggregate[tau].items()},
            })
    return radius_summary, temperature_rows, per_sample


def _decision(
    protocol: dict[str, Any],
    teacher: dict[str, Any],
    students: dict[str, dict[str, Any]],
    temperature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selection = protocol["selection"]
    radii = [int(v) for v in protocol["radii"]]
    eligible: list[int] = []
    radius_checks: dict[str, Any] = {}
    for radius in radii:
        row = teacher[str(radius)]
        checks = {
            "bbox_physical_coverage": float(row["bbox_physical_coverage"]) >= float(selection["bbox_physical_coverage_min"]),
            "teacher_excess_reduction": float(row["excess_error_reduction"]) >= float(selection["teacher_excess_error_reduction_min"]),
            "teacher_real_vs_shuffled": float(row["real_vs_shuffled_best_error_improvement"]) >= float(selection["teacher_real_vs_shuffled_best_error_improvement_min"]),
            "teacher_real_vs_spatial_null": float(row["real_vs_spatial_null_best_error_improvement"]) >= float(selection["teacher_real_vs_spatial_null_best_error_improvement_min"]),
            "teacher_foreground_excess": float(row["foreground_excess_error_reduction"]) >= float(selection["teacher_foreground_excess_error_reduction_min"]),
        }
        radius_checks[str(radius)] = checks
        if all(checks.values()):
            eligible.append(radius)
    selected_radius = min(eligible) if eligible else None
    result: dict[str, Any] = {
        "radius_checks": radius_checks,
        "eligible_radii": eligible,
        "selected_radius": selected_radius,
        "selected_temperature": None,
        "a5_corr_authorized": False,
        "next_action": "REJECT_RELATIONAL_TRANSPORT" if selected_radius is None else "DIAGNOSE_STUDENT",
    }
    if selected_radius is None:
        return result

    a4 = students["A4"][str(selected_radius)]
    random = students["RANDOM"][str(selected_radius)]
    hard_epe_improvement = float(a4["physical_epe_improvement_over_zero"])
    random_improvement = float(random["physical_epe_improvement_over_zero"])
    student_checks = {
        "hard_physics": hard_epe_improvement >= float(selection["student_hard_epe_improvement_over_zero_min"]),
        "learned_over_random": (hard_epe_improvement - random_improvement) >= float(selection["student_hard_epe_advantage_over_random_min"]),
        "temporal_specificity": float(a4["real_minus_shuffled_top1_cosine"]) >= float(selection["student_real_vs_shuffled_top1_cosine_min"]),
        "spatial_null_specificity": float(a4["real_minus_spatial_null_top1_cosine"]) >= float(selection["student_real_vs_spatial_null_top1_cosine_min"]),
        "confidence_margin": float(a4["real_margin"]) >= float(selection["student_confidence_margin_min"]),
    }
    result["student_checks"] = student_checks
    result["a4_minus_a4d_hard_epe_improvement"] = (
        hard_epe_improvement - float(students["A4D"][str(selected_radius)]["physical_epe_improvement_over_zero"])
    )
    result["a4_minus_random_hard_epe_improvement"] = hard_epe_improvement - random_improvement
    if not all(student_checks.values()):
        result["next_action"] = "STUDENT_REPRESENTATION_BOTTLENECK"
        return result

    rows = [
        row for row in temperature_rows
        if row["model"] == "A4" and int(row["radius"]) == selected_radius
    ]
    rows.sort(key=lambda row: float(row["temperature"]), reverse=True)
    acceptable = [
        row for row in rows
        if float(row["entropy"]) <= float(selection["student_entropy_max"])
    ]
    if not acceptable:
        result["next_action"] = "SOFTMAX_CALIBRATION_REQUIRED"
        return result
    chosen = acceptable[0]
    result["selected_temperature"] = float(chosen["temperature"])
    result["temperature_selection_rule"] = "largest preregistered tau satisfying entropy_max after hard matching passes"
    result["a5_corr_authorized"] = True
    result["next_action"] = "AUTHORIZE_A5_CORR"
    return result


def run(
    *,
    source_config_path: Path,
    protocol_path: Path,
    a4_checkpoint_path: Path,
    a4d_checkpoint_path: Path,
    output_dir: Path,
    device_name: str,
    sample_count_override: int | None,
    batch_size: int,
) -> dict[str, Any]:
    source = _read_yaml(source_config_path)
    protocol_root = _read_yaml(protocol_path)
    protocol = protocol_root.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("V2 protocol YAML lacks protocol mapping")
    data = source.get("data")
    if not isinstance(data, dict):
        raise ValueError("source config lacks data mapping")
    if any(data.get(key) is not False for key in ("official_test_opened", "codabench_opened", "evttc_test_opened")):
        raise ValueError("V2 refuses any config authorizing private/test access")
    if protocol.get("public_train_only") is not True or protocol.get("validation_or_test_opened") is not False or int(protocol.get("optimizer_steps", -1)) != 0:
        raise ValueError("V2 protocol violated train-only/zero-training contract")
    radii = tuple(int(v) for v in protocol["radii"])
    temperatures = tuple(float(v) for v in protocol["temperatures"])
    if radii != (1, 2, 4):
        raise ValueError("V2 radii are frozen to 1/2/4")
    if temperatures != (0.02, 0.04, 0.07, 0.10):
        raise ValueError("V2 temperature grid is frozen to 0.02/0.04/0.07/0.10")
    samples = int(sample_count_override or protocol["sample_count"])
    if samples <= 0 or batch_size <= 0:
        raise ValueError("samples and batch_size must be positive")

    seed_everything(int(protocol["seed"]), deterministic=True)
    device = resolve_device(device_name)
    cache_manifest = (ROOT / str(data["cache_manifest"])).resolve(strict=True)
    teacher_cfg = data.get("dinov3_relational_teacher")
    if not isinstance(teacher_cfg, dict):
        raise ValueError("source config lacks DINO teacher")
    teacher_manifest = (ROOT / str(teacher_cfg["manifest"])).resolve(strict=True)
    base = GarlTTCObjectEventV4Dataset(str(cache_manifest), splits=("train",))
    dataset = DINOv3RelationalTeacherDataset(
        base,
        manifest_path=teacher_manifest,
        expected_artifact_sha256=str(teacher_cfg["artifact_sha256"]),
        expected_manifest_sha256=str(teacher_cfg["manifest_sha256"]),
    )
    indices = _selected_indices(len(dataset), samples)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_object_event_v4)
    bank = _collect_metadata_and_teacher(loader)
    partner, different_track_fraction = _same_sequence_null_partners(
        bank["sequence_ids"], bank["track_ids"], indices
    )
    valid_null_fraction = float((partner >= 0).mean())
    if valid_null_fraction < 0.95:
        raise RuntimeError(f"same-sequence null coverage too low: {valid_null_fraction:.3f}")

    teacher_summary, per_sample = _teacher_audit(
        bank, partner, device=device, radii=radii, batch_size=batch_size
    )

    a4_model, a4_checkpoint = _load_checkpoint_model(a4_checkpoint_path, device)
    a4d_model, _a4d_checkpoint = _load_checkpoint_model(a4d_checkpoint_path, device)
    random_model = _random_model_like(a4_checkpoint, device, seed=7001)
    models = {"A4": a4_model, "A4D": a4d_model, "RANDOM": random_model}
    student_summary: dict[str, Any] = {}
    temperature_rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        features = _collect_student_features(model, loader, device)
        summary, temps, rows = _student_audit(
            model_name, features, bank, partner,
            device=device, radii=radii, temperatures=temperatures, batch_size=batch_size,
        )
        student_summary[model_name] = summary
        temperature_rows.extend(temps)
        per_sample.extend(rows)
        del features
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decision = _decision(protocol, teacher_summary, student_summary, temperature_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_csv = output_dir / "a5_preflight_v2_teacher_radius.csv"
    student_csv = output_dir / "a5_preflight_v2_student_radius.csv"
    temperature_csv = output_dir / "a5_preflight_v2_temperature.csv"
    sample_csv = output_dir / "a5_preflight_v2_samples.csv"
    pd.DataFrame([{"radius": int(r), **teacher_summary[str(r)]} for r in radii]).to_csv(teacher_csv, index=False, lineterminator="\n")
    pd.DataFrame([
        {"model": model, "radius": int(r), **student_summary[model][str(r)]}
        for model in ("A4", "A4D", "RANDOM") for r in radii
    ]).to_csv(student_csv, index=False, lineterminator="\n")
    pd.DataFrame(temperature_rows).to_csv(temperature_csv, index=False, lineterminator="\n")
    pd.DataFrame(per_sample).to_csv(sample_csv, index=False, lineterminator="\n")

    payload: dict[str, Any] = {
        "artifact_type": "a5_transport_preflight_train_only_v2",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "public_train_only": True,
            "validation_or_test_opened": False,
            "optimizer_steps": 0,
            "samples": len(indices),
            "selection_spans_complete_train_cache": True,
            "radii": list(radii),
            "temperatures": list(temperatures),
            "sequence_count": len(set(bank["sequence_ids"])),
            "same_sequence_null_fraction": valid_null_fraction,
            "null_different_track_fraction": different_track_fraction,
        },
        "source": {
            "source_config": str(source_config_path.relative_to(ROOT)),
            "source_config_sha256": _sha256(source_config_path),
            "protocol": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": _sha256(protocol_path),
            "a4_checkpoint": str(a4_checkpoint_path.relative_to(ROOT)),
            "a4_checkpoint_sha256": _sha256(a4_checkpoint_path),
            "a4d_checkpoint": str(a4d_checkpoint_path.relative_to(ROOT)),
            "a4d_checkpoint_sha256": _sha256(a4d_checkpoint_path),
            "teacher_manifest": str(teacher_manifest.relative_to(ROOT)),
            "teacher_manifest_sha256": _sha256(teacher_manifest),
            "a4_model_config": asdict(a4_model.config),
        },
        "teacher": teacher_summary,
        "students": student_summary,
        "decision": decision,
        "interpretation_contract": protocol.get("interpretation", {}),
        "files": {
            "teacher_radius_csv": teacher_csv.name,
            "teacher_radius_csv_sha256": _sha256(teacher_csv),
            "student_radius_csv": student_csv.name,
            "student_radius_csv_sha256": _sha256(student_csv),
            "temperature_csv": temperature_csv.name,
            "temperature_csv_sha256": _sha256(temperature_csv),
            "samples_csv": sample_csv.name,
            "samples_csv_sha256": _sha256(sample_csv),
        },
    }
    sign_artifact(payload)
    _atomic_json(output_dir / "a5_transport_preflight_v2.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--a4-checkpoint", type=Path, default=DEFAULT_A4)
    parser.add_argument("--a4d-checkpoint", type=Path, default=DEFAULT_A4D)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    payload = run(
        source_config_path=args.source_config.resolve(),
        protocol_path=args.protocol.resolve(),
        a4_checkpoint_path=args.a4_checkpoint.resolve(),
        a4d_checkpoint_path=args.a4d_checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        device_name=args.device,
        sample_count_override=args.samples,
        batch_size=args.batch_size,
    )
    print(json.dumps(payload["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
