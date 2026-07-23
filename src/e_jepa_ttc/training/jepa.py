"""Self-supervised JEPA-style pretraining for voxel caches."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from e_jepa_ttc.data.evttc import NAVIGATION_FEATURE_NAMES
from e_jepa_ttc.data.ml_cache import validate_voxel_cache
from e_jepa_ttc.models import build_encoder
from e_jepa_ttc.utils.io import ensure_parent, write_structured

EVENT_MOTION_FEATURE_NAMES = (
    "event_log_total_mass",
    "event_log_late_minus_early_mass",
    "event_temporal_mass_slope",
    "event_centroid_dx",
    "event_centroid_dy",
    "event_polarity_balance",
)


class VoxelOnlyDataset(Dataset[torch.Tensor]):
    """Dataset backed by voxel tensors only."""

    def __init__(self, x: np.ndarray, indices: np.ndarray) -> None:
        self.x = x
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        real_idx = int(self.indices[idx])
        return torch.from_numpy(self.x[real_idx].astype(np.float32, copy=False))


class TemporalVoxelPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset of context windows paired with future target windows."""

    def __init__(
        self,
        x: np.ndarray,
        context_indices: np.ndarray,
        target_indices: np.ndarray,
    ) -> None:
        self.x = x
        self.context_indices = context_indices.astype(np.int64)
        self.target_indices = target_indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.context_indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_idx = int(self.context_indices[idx])
        target_idx = self.target_indices[idx]
        mask = target_idx >= 0
        safe_target_idx = np.where(mask, target_idx, context_idx)
        context = torch.from_numpy(self.x[context_idx].astype(np.float32, copy=False))
        target = torch.from_numpy(self.x[safe_target_idx].astype(np.float32, copy=False))
        return context, target, torch.from_numpy(mask)


class JEPAPredictor(nn.Module):
    """Small latent predictor used between context and target encoders."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict target latent vectors from context latents."""

        return self.net(x)


class TemporalJEPAPredictor(nn.Module):
    """Predict horizon-conditioned future latent vectors from context latents."""

    def __init__(self, dim: int, horizon_count: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if horizon_count <= 0:
            msg = "horizon_count must be positive."
            raise ValueError(msg)
        hidden = hidden_dim or dim * 2
        self.horizon_embedding = nn.Embedding(horizon_count, dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, context_z: torch.Tensor, horizon_ids: torch.Tensor) -> torch.Tensor:
        """Predict one latent vector per requested future horizon."""

        horizon_z = self.horizon_embedding(horizon_ids.to(device=context_z.device))
        batch, dim = context_z.shape
        context = context_z[:, None, :].expand(batch, horizon_z.shape[0], dim)
        horizon = horizon_z[None, :, :].expand(batch, horizon_z.shape[0], dim)
        return self.net(torch.cat([context, horizon], dim=-1))


class DenseTemporalJEPAPredictor(nn.Module):
    """Predict horizon-conditioned dense future tokens from context tokens."""

    def __init__(
        self,
        dim: int,
        horizon_count: int,
        *,
        motion_dim: int = 0,
        layer_count: int = 0,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if horizon_count <= 0:
            msg = "horizon_count must be positive."
            raise ValueError(msg)
        if motion_dim < 0:
            msg = "motion_dim must be non-negative."
            raise ValueError(msg)
        if layer_count < 0:
            msg = "layer_count must be non-negative."
            raise ValueError(msg)
        hidden = hidden_dim or dim * 2
        self.motion_dim = motion_dim
        self.horizon_embedding = nn.Embedding(horizon_count, dim)
        self.motion_projection = nn.Linear(motion_dim, dim) if motion_dim else None
        self.layer_embedding = nn.Embedding(layer_count, dim) if layer_count else None
        input_dim = dim * (2 + bool(motion_dim) + bool(layer_count))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        horizon_ids: torch.Tensor,
        motion_features: torch.Tensor | None,
        layer_id: int | None = None,
    ) -> torch.Tensor:
        """Predict one dense token grid per requested future horizon."""

        batch, token_count, dim = context_tokens.shape
        horizon_z = self.horizon_embedding(horizon_ids.to(device=context_tokens.device))
        context = context_tokens[:, None, :, :].expand(
            batch,
            horizon_z.shape[0],
            token_count,
            dim,
        )
        horizon = horizon_z[None, :, None, :].expand(
            batch,
            horizon_z.shape[0],
            token_count,
            dim,
        )
        pieces = [context, horizon]
        if self.motion_projection is not None:
            if motion_features is None:
                msg = "motion_features are required for motion-conditioned dense JEPA."
                raise ValueError(msg)
            motion_z = self.motion_projection(motion_features.to(device=context_tokens.device))
            motion = motion_z[:, None, None, :].expand(
                batch,
                horizon_z.shape[0],
                token_count,
                dim,
            )
            pieces.append(motion)
        if self.layer_embedding is not None:
            if layer_id is None:
                msg = "layer_id is required for layer-conditioned dense JEPA."
                raise ValueError(msg)
            layer_idx = torch.tensor(layer_id, device=context_tokens.device, dtype=torch.long)
            layer_z = self.layer_embedding(layer_idx).view(1, 1, 1, dim)
            layer = layer_z.expand(batch, horizon_z.shape[0], token_count, dim)
            pieces.append(layer)
        return self.net(torch.cat(pieces, dim=-1))


class DenseTemporalTransformerJEPAPredictor(nn.Module):
    """Transformer predictor for horizon/action-conditioned dense future tokens."""

    def __init__(
        self,
        dim: int,
        horizon_count: int,
        *,
        motion_dim: int = 0,
        layer_count: int = 0,
        depth: int = 2,
        num_heads: int = 6,
        mlp_ratio: float = 3.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if horizon_count <= 0:
            msg = "horizon_count must be positive."
            raise ValueError(msg)
        if motion_dim < 0:
            msg = "motion_dim must be non-negative."
            raise ValueError(msg)
        if layer_count < 0:
            msg = "layer_count must be non-negative."
            raise ValueError(msg)
        if depth <= 0:
            msg = "depth must be positive."
            raise ValueError(msg)
        if dim % num_heads != 0:
            msg = "dim must be divisible by num_heads."
            raise ValueError(msg)
        self.motion_dim = motion_dim
        self.horizon_embedding = nn.Embedding(horizon_count, dim)
        self.motion_projection = nn.Linear(motion_dim, dim) if motion_dim else None
        self.layer_embedding = nn.Embedding(layer_count, dim) if layer_count else None
        self.input_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=int(dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        horizon_ids: torch.Tensor,
        motion_features: torch.Tensor | None,
        layer_id: int | None = None,
    ) -> torch.Tensor:
        """Predict dense future tokens with token-token attention per horizon."""

        batch, token_count, dim = context_tokens.shape
        horizon_z = self.horizon_embedding(horizon_ids.to(device=context_tokens.device))
        tokens = context_tokens[:, None, :, :] + horizon_z[None, :, None, :]
        if self.motion_projection is not None:
            if motion_features is None:
                msg = "motion_features are required for motion-conditioned dense JEPA."
                raise ValueError(msg)
            motion_z = self.motion_projection(motion_features.to(device=context_tokens.device))
            tokens = tokens + motion_z[:, None, None, :]
        if self.layer_embedding is not None:
            if layer_id is None:
                msg = "layer_id is required for layer-conditioned dense JEPA."
                raise ValueError(msg)
            layer_idx = torch.tensor(layer_id, device=context_tokens.device, dtype=torch.long)
            tokens = tokens + self.layer_embedding(layer_idx).view(1, 1, 1, dim)

        flat_tokens = self.input_norm(tokens.reshape(batch * horizon_z.shape[0], token_count, dim))
        for layer in self.layers:
            flat_tokens = layer(flat_tokens)
        flat_tokens = self.output_projection(self.output_norm(flat_tokens))
        return flat_tokens.view(batch, horizon_z.shape[0], token_count, dim)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_indices(split: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    split_text = split.astype(str)
    mask = np.isin(split_text, np.array(names, dtype=str))
    return np.flatnonzero(mask).astype(np.int64)


def _build_temporal_pairs(
    *,
    split: np.ndarray,
    sequence_id: np.ndarray,
    timestamp_us: np.ndarray,
    context_start_us: np.ndarray,
    context_end_us: np.ndarray,
    split_names: tuple[str, ...],
    horizons_ms: tuple[int, ...],
    max_target_slop_ms: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not horizons_ms:
        msg = "horizons_ms must contain at least one horizon."
        raise ValueError(msg)
    if any(horizon <= 0 for horizon in horizons_ms):
        msg = "All temporal horizons must be positive."
        raise ValueError(msg)
    if max_target_slop_ms < 0:
        msg = "max_target_slop_ms must be non-negative."
        raise ValueError(msg)

    selected = _split_indices(split, split_names)
    sequence_text = sequence_id.astype(str)
    timestamps = timestamp_us.astype(np.int64)
    context_starts = context_start_us.astype(np.int64)
    context_ends = context_end_us.astype(np.int64)
    expected_shape = timestamps.shape
    if context_starts.shape != expected_shape or context_ends.shape != expected_shape:
        msg = "timestamp_us, context_start_us and context_end_us must have identical shapes."
        raise ValueError(msg)
    if np.any(context_starts >= context_ends):
        msg = "Every cached context window must have positive duration."
        raise ValueError(msg)
    if np.any(timestamps != context_ends):
        msg = "Cached timestamp_us must equal context_end_us for causal pairing."
        raise ValueError(msg)
    max_slop_us = int(max_target_slop_ms * 1000)
    contexts: list[int] = []
    targets: list[np.ndarray] = []
    slop_by_horizon: list[list[float]] = [[] for _ in horizons_ms]

    for sequence in sorted(set(sequence_text[selected].tolist())):
        sequence_indices = selected[sequence_text[selected] == sequence]
        order = np.argsort(timestamps[sequence_indices], kind="stable")
        sequence_indices = sequence_indices[order]
        sequence_times = timestamps[sequence_indices]
        sequence_starts = context_starts[sequence_indices]
        for context_idx in sequence_indices:
            row = np.full((len(horizons_ms),), -1, dtype=np.int64)
            context_end = int(context_ends[context_idx])
            for horizon_idx, horizon_ms in enumerate(horizons_ms):
                # Horizon denotes a gap after the causal context. The target
                # is a complete cached window whose *start* is at or after
                # context_end + horizon, never a window ending there.
                target_start = context_end + int(horizon_ms * 1000)
                target_pos = int(np.searchsorted(sequence_starts, target_start, side="left"))
                if target_pos >= sequence_times.shape[0]:
                    continue
                target_idx = int(sequence_indices[target_pos])
                slop_us = int(context_starts[target_idx] - target_start)
                if 0 <= slop_us <= max_slop_us:
                    if context_starts[target_idx] < context_end:
                        msg = "Temporal target overlaps its causal context."
                        raise RuntimeError(msg)
                    row[horizon_idx] = target_idx
                    slop_by_horizon[horizon_idx].append(slop_us / 1000.0)
            if np.any(row >= 0):
                contexts.append(int(context_idx))
                targets.append(row)

    context_indices = np.array(contexts, dtype=np.int64)
    target_indices = (
        np.stack(targets).astype(np.int64)
        if targets
        else np.empty((0, len(horizons_ms)), dtype=np.int64)
    )
    per_horizon: dict[str, dict[str, float | int | None]] = {}
    for horizon_idx, horizon_ms in enumerate(horizons_ms):
        valid = target_indices[:, horizon_idx] >= 0 if target_indices.size else np.array([])
        slops = slop_by_horizon[horizon_idx]
        count = int(valid.sum()) if target_indices.size else 0
        per_horizon[str(horizon_ms)] = {
            "count": count,
            "coverage": float(count / max(context_indices.shape[0], 1)),
            "mean_slop_ms": float(np.mean(slops)) if slops else None,
            "max_slop_ms": float(np.max(slops)) if slops else None,
        }

    stats: dict[str, Any] = {
        "split_names": list(split_names),
        "context_count": int(context_indices.shape[0]),
        "target_pair_count": int((target_indices >= 0).sum()) if target_indices.size else 0,
        "horizons_ms": list(horizons_ms),
        "max_target_slop_ms": max_target_slop_ms,
        "future_window_semantics": "disjoint_window_start_after_context_plus_horizon",
        "target_windows_are_disjoint": True,
        "per_horizon": per_horizon,
    }
    return context_indices, target_indices, stats


def _masked_context(x: torch.Tensor, *, mask_ratio: float, block_count: int) -> torch.Tensor:
    if not 0.0 <= mask_ratio < 1.0:
        msg = "mask_ratio must be in [0, 1)."
        raise ValueError(msg)
    if block_count <= 0:
        msg = "block_count must be positive."
        raise ValueError(msg)
    if mask_ratio == 0.0:
        return x

    out = x.clone()
    batch, _channels, height, width = out.shape
    block_area = max(1, int(height * width * mask_ratio / block_count))
    block_side = max(1, int(block_area**0.5))
    max_h = max(1, min(height, block_side))
    max_w = max(1, min(width, block_side))

    for batch_idx in range(batch):
        for _ in range(block_count):
            block_h = int(torch.randint(max(1, max_h // 2), max_h + 1, ()).item())
            block_w = int(torch.randint(max(1, max_w // 2), max_w + 1, ()).item())
            y0 = int(torch.randint(0, height - block_h + 1, ()).item())
            x0 = int(torch.randint(0, width - block_w + 1, ()).item())
            out[batch_idx, :, y0 : y0 + block_h, x0 : x0 + block_w] = 0.0
    return out


def _tubelet_masked_context(
    x: torch.Tensor,
    *,
    mask_ratio: float,
    block_count: int,
    event_bins: int,
) -> torch.Tensor:
    """Mask causal event channels with spatio-temporal event-tubelet blocks."""

    if not 0.0 <= mask_ratio < 1.0:
        msg = "mask_ratio must be in [0, 1)."
        raise ValueError(msg)
    if block_count <= 0:
        msg = "block_count must be positive."
        raise ValueError(msg)
    if event_bins <= 0:
        msg = "event_bins must be positive."
        raise ValueError(msg)
    if mask_ratio == 0.0:
        return x

    event_channel_count = event_bins * 2
    if x.shape[1] < event_channel_count:
        msg = (
            f"Tubelet masking expects at least {event_channel_count} event channels "
            f"for {event_bins} bins, got {x.shape[1]}."
        )
        raise ValueError(msg)

    out = x.clone()
    batch, _channels, height, width = out.shape
    event = out[:, :event_channel_count].view(batch, 2, event_bins, height, width)
    block_volume = max(1, int(event_bins * height * width * mask_ratio / block_count))
    max_t = max(1, min(event_bins, block_volume))

    for batch_idx in range(batch):
        for _ in range(block_count):
            block_t = int(torch.randint(1, max_t + 1, ()).item())
            spatial_area = max(1, block_volume // block_t)
            block_side = max(1, int(spatial_area**0.5))
            max_h = max(1, min(height, block_side))
            max_w = max(1, min(width, block_side))
            block_h = int(torch.randint(max(1, max_h // 2), max_h + 1, ()).item())
            block_w = int(torch.randint(max(1, max_w // 2), max_w + 1, ()).item())
            t0 = int(torch.randint(0, event_bins - block_t + 1, ()).item())
            y0 = int(torch.randint(0, height - block_h + 1, ()).item())
            x0 = int(torch.randint(0, width - block_w + 1, ()).item())
            event[
                batch_idx,
                :,
                t0 : t0 + block_t,
                y0 : y0 + block_h,
                x0 : x0 + block_w,
            ] = 0.0
    return out


def _mask_context(
    x: torch.Tensor,
    *,
    mask_ratio: float,
    block_count: int,
    mask_mode: str,
    event_bins: int,
) -> torch.Tensor:
    if mask_mode == "spatial":
        return _masked_context(x, mask_ratio=mask_ratio, block_count=block_count)
    if mask_mode == "tubelet":
        return _tubelet_masked_context(
            x,
            mask_ratio=mask_ratio,
            block_count=block_count,
            event_bins=event_bins,
        )
    msg = "mask_mode must be one of {'spatial', 'tubelet'}."
    raise ValueError(msg)


@torch.no_grad()
def _update_ema(target: nn.Module, online: nn.Module, *, momentum: float) -> None:
    for target_param, online_param in zip(target.parameters(), online.parameters(), strict=True):
        target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)
    for target_buffer, online_buffer in zip(target.buffers(), online.buffers(), strict=True):
        target_buffer.copy_(online_buffer)


def _variance_loss(z: torch.Tensor, *, min_std: float) -> torch.Tensor:
    z = z.float()
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(min_std - std).mean()


def _effective_rank(z: torch.Tensor) -> torch.Tensor:
    """Return entropy effective rank across independent sample embeddings."""

    rows = z.float().reshape(-1, z.shape[-1])
    if rows.shape[0] <= 1:
        return rows.sum() * 0.0
    centered = rows - rows.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    mass = singular_values.sum()
    if bool(mass <= 1e-12):
        return mass * 0.0
    probability = singular_values / mass
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return entropy.exp()


def _visreg_sketch_loss(z: torch.Tensor, *, projection_count: int) -> torch.Tensor:
    """Approximate VISReg shape regularization with Gaussian SWD sketches."""

    if projection_count <= 0:
        msg = "projection_count must be positive."
        raise ValueError(msg)
    z = z.float().reshape(-1, z.shape[-1])
    if z.shape[0] <= 1:
        return z.sum() * 0.0
    projections = torch.randn(
        z.shape[-1],
        projection_count,
        device=z.device,
        dtype=z.dtype,
    )
    projections = functional.normalize(projections, dim=0)
    projected = torch.sort(z @ projections, dim=0).values
    quantiles = (torch.arange(z.shape[0], device=z.device, dtype=torch.float32) + 0.5) / z.shape[0]
    gaussian = math.sqrt(2.0) * torch.special.erfinv(2.0 * quantiles - 1.0)
    gaussian = gaussian[:, None].to(dtype=projected.dtype)
    return torch.square(projected - gaussian.expand_as(projected)).mean()


def _visreg_components(
    z: torch.Tensor,
    *,
    projection_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return VISReg center, scale, and normalized-shape losses."""

    z = z.float().reshape(-1, z.shape[-1])
    if z.shape[0] <= 1:
        zero = z.sum() * 0.0
        return zero, zero, zero
    mean = z.mean(dim=0, keepdim=True)
    centered = z - mean
    std = torch.sqrt(centered.var(dim=0, unbiased=False, keepdim=True) + 1e-4)
    normalized = centered / std.detach().clamp_min(1e-4)
    center_loss = torch.square(mean).mean()
    scale_loss = torch.square(1.0 - std).mean()
    shape_loss = _visreg_sketch_loss(normalized, projection_count=projection_count)
    return center_loss, scale_loss, shape_loss


def _embedding_regularization_loss(
    context_z: torch.Tensor,
    pred_z: torch.Tensor,
    *,
    regularizer: str,
    variance_weight: float,
    min_std: float,
    visreg_center_weight: float,
    visreg_sketch_weight: float,
    visreg_projection_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    center_loss = context_z.sum() * 0.0
    scale_loss = _variance_loss(context_z, min_std=min_std) + _variance_loss(
        pred_z,
        min_std=min_std,
    )
    sketch_loss = context_z.sum() * 0.0
    if regularizer == "visreg":
        context_center, context_scale, context_shape = _visreg_components(
            context_z,
            projection_count=visreg_projection_count,
        )
        pred_center, pred_scale, pred_shape = _visreg_components(
            pred_z,
            projection_count=visreg_projection_count,
        )
        center_loss = context_center + pred_center
        scale_loss = context_scale + pred_scale
        sketch_loss = context_shape + pred_shape
    elif regularizer != "variance":
        msg = "regularizer must be one of {'variance', 'visreg'}."
        raise ValueError(msg)
    return (
        visreg_center_weight * center_loss
        + variance_weight * scale_loss
        + visreg_sketch_weight * sketch_loss,
        center_loss,
        scale_loss,
        sketch_loss,
    )


def _context_motion_features(x: torch.Tensor, *, bins: int) -> torch.Tensor:
    """Extract causal motion proxy features from the context voxel window."""

    if bins <= 0:
        msg = "bins must be positive."
        raise ValueError(msg)
    if x.shape[1] < bins * 2:
        msg = f"Expected at least {bins * 2} event channels, got {x.shape[1]}."
        raise ValueError(msg)

    events = x[:, : bins * 2].float().view(x.shape[0], 2, bins, x.shape[-2], x.shape[-1])
    mass_by_time = events.abs().sum(dim=(1, 3, 4))
    pos_mass = events[:, 0].abs().sum(dim=(1, 2, 3))
    neg_mass = events[:, 1].abs().sum(dim=(1, 2, 3))
    total_mass = mass_by_time.sum(dim=1).clamp_min(1e-6)
    midpoint = max(1, bins // 2)
    early_mass = mass_by_time[:, :midpoint].sum(dim=1)
    late_mass = mass_by_time[:, midpoint:].sum(dim=1)
    temporal_slope = (late_mass - early_mass) / total_mass
    polarity_balance = (pos_mass - neg_mass) / (pos_mass + neg_mass).clamp_min(1e-6)

    spatial_mass = events.abs().sum(dim=1)
    early_map = spatial_mass[:, :midpoint].sum(dim=1)
    late_map = spatial_mass[:, midpoint:].sum(dim=1)
    grid_x = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device, dtype=torch.float32)
    grid_y = torch.linspace(-1.0, 1.0, x.shape[-2], device=x.device, dtype=torch.float32)
    early_total = early_map.sum(dim=(1, 2)).clamp_min(1e-6)
    late_total = late_map.sum(dim=(1, 2)).clamp_min(1e-6)
    early_cx = (early_map * grid_x[None, None, :]).sum(dim=(1, 2)) / early_total
    late_cx = (late_map * grid_x[None, None, :]).sum(dim=(1, 2)) / late_total
    early_cy = (early_map * grid_y[None, :, None]).sum(dim=(1, 2)) / early_total
    late_cy = (late_map * grid_y[None, :, None]).sum(dim=(1, 2)) / late_total

    return torch.stack(
        [
            torch.log1p(total_mass),
            torch.log1p(late_mass.clamp_min(0.0)) - torch.log1p(early_mass.clamp_min(0.0)),
            temporal_slope,
            late_cx - early_cx,
            late_cy - early_cy,
            polarity_balance,
        ],
        dim=1,
    )


def _context_action_features(
    x: torch.Tensor,
    *,
    bins: int,
    metadata_channels: bool,
    navigation_feature_count: int,
) -> torch.Tensor:
    """Extract causal event-motion and ego-action features from context only."""

    pieces = [_context_motion_features(x, bins=bins)]
    if navigation_feature_count:
        event_channel_count = bins * 2
        metadata_channel_count = 2 if metadata_channels else 0
        navigation_start = event_channel_count + metadata_channel_count
        navigation_end = navigation_start + navigation_feature_count
        if x.shape[1] < navigation_end:
            msg = (
                "Expected navigation channels at "
                f"[{navigation_start}:{navigation_end}], got {x.shape[1]} channels."
            )
            raise ValueError(msg)
        navigation = x[:, navigation_start:navigation_end].float().mean(dim=(2, 3))
        pieces.append(navigation)
    return torch.nan_to_num(torch.cat(pieces, dim=1), nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_action_features(
    features: torch.Tensor,
    *,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return features
    return (features - mean.to(device=features.device)) / std.to(device=features.device)


def _estimate_action_feature_stats(
    x: np.ndarray,
    indices: np.ndarray,
    *,
    bins: int,
    metadata_channels: bool,
    navigation_feature_count: int,
    batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate action-feature normalization from train context windows only."""

    if indices.size == 0:
        msg = "Cannot estimate action feature stats without train context indices."
        raise ValueError(msg)
    rows: list[torch.Tensor] = []
    for start in range(0, int(indices.size), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = torch.from_numpy(x[batch_indices].astype(np.float32, copy=False))
        rows.append(
            _context_action_features(
                batch,
                bins=bins,
                metadata_channels=metadata_channels,
                navigation_feature_count=navigation_feature_count,
            )
        )
    features = torch.cat(rows, dim=0)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def _cache_bool(cache: NpzFile, key: str) -> bool:
    if key not in cache.files:
        return False
    return bool(np.asarray(cache[key]).item())


def _cache_string_tuple(cache: NpzFile, key: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if key not in cache.files:
        return fallback
    return tuple(str(value) for value in np.asarray(cache[key]).astype(str).tolist())


def _without_future_navigation(
    future_x: torch.Tensor,
    *,
    bins: int,
    metadata_channels: bool,
    navigation_feature_count: int,
) -> torch.Tensor:
    """Remove future ego-navigation from target inputs while retaining future events."""

    if navigation_feature_count <= 0:
        return future_x
    navigation_start = bins * 2 + (2 if metadata_channels else 0)
    navigation_end = navigation_start + navigation_feature_count
    if future_x.ndim != 5 or future_x.shape[2] < navigation_end:
        msg = (
            "Future target tensor does not contain the declared navigation channels at "
            f"[{navigation_start}:{navigation_end}]."
        )
        raise ValueError(msg)
    target = future_x.clone()
    target[:, :, navigation_start:navigation_end] = 0.0
    return target


def _objective_name(
    *,
    use_temporal: bool,
    dense_tokens: bool,
    motion_conditioning: bool,
    action_conditioning: bool,
    deep_supervision: bool,
    dense_predictor: str,
    context_token_weight: float,
    regularizer: str,
    mask_mode: str,
) -> str:
    if not use_temporal:
        return "masked_same_window"
    regularizer_prefix = "visreg_" if regularizer == "visreg" else ""
    mask_prefix = "tubeletmask_" if mask_mode == "tubelet" else ""
    deep_prefix = "deep_" if deep_supervision and dense_tokens else ""
    context_prefix = "alltoken_" if dense_tokens and context_token_weight > 0.0 else ""
    predictor_prefix = "transformer_" if dense_tokens and dense_predictor == "transformer" else ""
    prefix = f"{regularizer_prefix}{mask_prefix}{deep_prefix}{context_prefix}{predictor_prefix}"
    if dense_tokens and action_conditioning:
        return f"{prefix}dense_temporal_token_action_multihorizon"
    if dense_tokens and motion_conditioning:
        return f"{prefix}dense_temporal_token_motion_multihorizon"
    if dense_tokens:
        return f"{prefix}dense_temporal_token_multihorizon"
    return "temporal_multihorizon"


def _forward_token_layers(
    encoder: nn.Module,
    x: torch.Tensor,
    layer_indices: tuple[int, ...],
) -> list[torch.Tensor]:
    if not layer_indices:
        return [encoder.forward_tokens(x)]
    if not hasattr(encoder, "forward_intermediate_tokens"):
        msg = f"{encoder.__class__.__name__} does not support deep token supervision."
        raise ValueError(msg)
    return encoder.forward_intermediate_tokens(x, layer_indices)


def _temporal_straightening_loss(
    context_z: torch.Tensor,
    pred_z: torch.Tensor,
    future_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize curvature in predicted latent multi-horizon trajectories."""

    if pred_z.shape[1] < 2:
        return pred_z.sum() * 0.0
    context_state = context_z.mean(dim=1) if context_z.ndim == 3 else context_z
    pred_state = pred_z.mean(dim=2) if pred_z.ndim == 4 else pred_z
    states = torch.cat([context_state[:, None, :], pred_state], dim=1)
    velocities = states[:, 1:] - states[:, :-1]
    curvature = 1.0 - functional.cosine_similarity(
        velocities[:, :-1],
        velocities[:, 1:],
        dim=-1,
        eps=1e-6,
    )
    valid = future_mask.to(device=curvature.device, dtype=torch.bool)
    valid_curvature = valid[:, :-1] & valid[:, 1:]
    if not bool(valid_curvature.any()):
        return curvature.sum() * 0.0
    return curvature[valid_curvature].mean()


def _dense_temporal_alignment(
    *,
    predictor: nn.Module,
    context_layers: list[torch.Tensor],
    target_layers: list[torch.Tensor],
    horizon_ids: torch.Tensor,
    motion_features: torch.Tensor | None,
    future_mask: torch.Tensor,
    layer_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(context_layers) != len(target_layers):
        msg = "context_layers and target_layers must have the same length."
        raise ValueError(msg)
    if layer_ids and len(layer_ids) != len(context_layers):
        msg = "layer_ids must match context_layers when provided."
        raise ValueError(msg)
    batch, horizon_count = future_mask.shape
    valid = future_mask.to(dtype=torch.float32, device=future_mask.device)
    valid_count = valid.sum().clamp_min(1.0)
    valid_bool = future_mask.to(dtype=torch.bool, device=future_mask.device)
    losses: list[torch.Tensor] = []
    straightening_losses: list[torch.Tensor] = []
    pred_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for layer_position, (context_z, flat_target) in enumerate(
        zip(context_layers, target_layers, strict=True)
    ):
        layer_id = layer_ids[layer_position] if layer_ids else None
        pred = predictor(context_z, horizon_ids, motion_features, layer_id)
        target_z = flat_target.view(batch, horizon_count, *flat_target.shape[1:])
        pred_norm = functional.normalize(pred, dim=-1)
        target_norm = functional.normalize(target_z, dim=-1)
        per_pair_loss = functional.smooth_l1_loss(
            pred_norm,
            target_norm,
            beta=0.1,
            reduction="none",
        ).mean(dim=tuple(range(2, pred_norm.ndim)))
        losses.append((per_pair_loss * valid).sum() / valid_count)
        straightening_losses.append(_temporal_straightening_loss(context_z, pred, future_mask))
        # Collapse diagnostics and regularization operate across independent
        # samples/pairs. Flattening spatial positions here allowed fixed
        # positional embeddings to masquerade as content variance.
        pred_rows.append(pred[valid_bool].mean(dim=1))
        target_rows.append(target_z[valid_bool].mean(dim=1))
    return (
        torch.stack(losses).mean(),
        torch.stack(straightening_losses).mean(),
        torch.cat(pred_rows, dim=0),
        torch.cat(target_rows, dim=0),
        context_layers[-1].mean(dim=1),
    )


def _dense_context_alignment(
    *,
    predictor: nn.Module,
    context_layers: list[torch.Tensor],
    target_layers: list[torch.Tensor],
    context_horizon_id: torch.Tensor,
    motion_features: torch.Tensor | None,
    layer_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(context_layers) != len(target_layers):
        msg = "context_layers and target_layers must have the same length."
        raise ValueError(msg)
    if layer_ids and len(layer_ids) != len(context_layers):
        msg = "layer_ids must match context_layers when provided."
        raise ValueError(msg)
    losses: list[torch.Tensor] = []
    pred_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for layer_position, (context_z, target_z) in enumerate(
        zip(context_layers, target_layers, strict=True)
    ):
        layer_id = layer_ids[layer_position] if layer_ids else None
        pred = predictor(context_z, context_horizon_id, motion_features, layer_id).squeeze(1)
        pred_norm = functional.normalize(pred, dim=-1)
        target_norm = functional.normalize(target_z, dim=-1)
        losses.append(
            functional.smooth_l1_loss(
                pred_norm,
                target_norm,
                beta=0.1,
                reduction="none",
            ).mean()
        )
        pred_rows.append(pred.mean(dim=1))
        target_rows.append(target_z.mean(dim=1))
    return torch.stack(losses).mean(), torch.cat(pred_rows, dim=0), torch.cat(target_rows, dim=0)


def _jepa_loss(
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    x: torch.Tensor,
    *,
    future_x: torch.Tensor | None,
    future_mask: torch.Tensor | None,
    horizon_ids: torch.Tensor | None,
    mask_ratio: float,
    block_count: int,
    mask_mode: str,
    regularizer: str,
    variance_weight: float,
    min_std: float,
    visreg_center_weight: float,
    visreg_sketch_weight: float,
    visreg_projection_count: int,
    temporal_straightening_weight: float,
    dense_tokens: bool,
    motion_conditioning: bool,
    deep_supervision_layers: tuple[int, ...],
    bins: int,
    metadata_channels: bool,
    navigation_feature_count: int,
    action_feature_mean: torch.Tensor | None,
    action_feature_std: torch.Tensor | None,
    context_token_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    context = _mask_context(
        x,
        mask_ratio=mask_ratio,
        block_count=block_count,
        mask_mode=mask_mode,
        event_bins=bins,
    )
    context_token_loss = torch.tensor(0.0, device=x.device)
    temporal_straightening_loss = torch.tensor(0.0, device=x.device)
    context_token_target_count = 0
    if dense_tokens and future_x is not None:
        context_layers = _forward_token_layers(encoder, context, deep_supervision_layers)
        context_z = context_layers[-1]
    else:
        context_layers = []
        context_z = encoder(context)
    if future_x is None:
        pred = predictor(context_z)
        with torch.no_grad():
            target_z = target_encoder(x)
        pred_norm = functional.normalize(pred, dim=-1)
        target_norm = functional.normalize(target_z, dim=-1)
        alignment_loss = functional.smooth_l1_loss(pred_norm, target_norm, beta=0.1)
        future_alignment_loss = alignment_loss
        pred_for_variance = pred
        target_for_metrics = target_z
        valid_fraction = 1.0
        target_pair_count = int(x.shape[0])
    else:
        if future_mask is None or horizon_ids is None:
            msg = "future_mask and horizon_ids are required for temporal JEPA."
            raise ValueError(msg)
        batch, horizon_count = future_mask.shape
        target_future_x = _without_future_navigation(
            future_x,
            bins=bins,
            metadata_channels=metadata_channels,
            navigation_feature_count=navigation_feature_count,
        )
        if dense_tokens:
            motion_features = (
                _normalize_action_features(
                    _context_action_features(
                        x,
                        bins=bins,
                        metadata_channels=metadata_channels,
                        navigation_feature_count=navigation_feature_count,
                    ),
                    mean=action_feature_mean,
                    std=action_feature_std,
                )
                if motion_conditioning
                else None
            )
            with torch.no_grad():
                flat_target_layers = _forward_token_layers(
                    target_encoder,
                    target_future_x.flatten(0, 1),
                    deep_supervision_layers,
                )
            (
                alignment_loss,
                temporal_straightening_loss,
                pred_for_variance,
                target_for_metrics,
                context_for_variance,
            ) = _dense_temporal_alignment(
                predictor=predictor,
                context_layers=context_layers,
                target_layers=flat_target_layers,
                horizon_ids=horizon_ids,
                motion_features=motion_features,
                future_mask=future_mask.to(device=context_z.device),
                layer_ids=deep_supervision_layers,
            )
            future_alignment_loss = alignment_loss
            if context_token_weight > 0.0:
                context_horizon_id = torch.tensor(
                    [horizon_count],
                    device=context_z.device,
                    dtype=torch.long,
                )
                with torch.no_grad():
                    context_target_layers = _forward_token_layers(
                        target_encoder,
                        x,
                        deep_supervision_layers,
                    )
                context_token_loss, context_pred_rows, context_target_rows = (
                    _dense_context_alignment(
                        predictor=predictor,
                        context_layers=context_layers,
                        target_layers=context_target_layers,
                        context_horizon_id=context_horizon_id,
                        motion_features=motion_features,
                        layer_ids=deep_supervision_layers,
                    )
                )
                alignment_loss = future_alignment_loss + context_token_weight * context_token_loss
                pred_for_variance = torch.cat([pred_for_variance, context_pred_rows], dim=0)
                target_for_metrics = torch.cat([target_for_metrics, context_target_rows], dim=0)
                context_token_target_count = int(context_pred_rows.shape[0])
            pred = pred_for_variance
            target_z = target_for_metrics
        else:
            pred = predictor(context_z, horizon_ids)
            with torch.no_grad():
                target_z = target_encoder(target_future_x.flatten(0, 1)).view(
                    batch, horizon_count, -1
                )
            pred_norm = functional.normalize(pred, dim=-1)
            target_norm = functional.normalize(target_z, dim=-1)
            per_pair_loss = functional.smooth_l1_loss(
                pred_norm,
                target_norm,
                beta=0.1,
                reduction="none",
            ).mean(dim=tuple(range(2, pred_norm.ndim)))
            valid = future_mask.to(device=per_pair_loss.device, dtype=per_pair_loss.dtype)
            valid_count = valid.sum().clamp_min(1.0)
            alignment_loss = (per_pair_loss * valid).sum() / valid_count
            temporal_straightening_loss = _temporal_straightening_loss(
                context_z,
                pred,
                future_mask,
            )
            future_alignment_loss = alignment_loss
            valid_bool = future_mask.to(device=pred.device, dtype=torch.bool)
            pred_for_variance = pred[valid_bool]
            target_for_metrics = target_z[valid_bool]
        valid = future_mask.to(device=context_z.device, dtype=torch.float32)
        valid_fraction = float(valid.mean().detach().cpu())
        target_pair_count = int(future_mask.sum().detach().cpu())
    if not (dense_tokens and future_x is not None):
        context_for_variance = context_z.mean(dim=1) if context_z.ndim == 3 else context_z
    if pred_for_variance.shape[0] <= 1:
        pred_for_variance = pred.reshape(-1, pred.shape[-1])
    if target_for_metrics.shape[0] <= 1:
        target_for_metrics = target_z.reshape(-1, target_z.shape[-1])
    if temporal_straightening_weight > 0.0:
        alignment_loss = (
            alignment_loss + temporal_straightening_weight * temporal_straightening_loss
        )
    regularization, visreg_center, variance, visreg_sketch = _embedding_regularization_loss(
        context_for_variance,
        pred_for_variance,
        regularizer=regularizer,
        variance_weight=variance_weight,
        min_std=min_std,
        visreg_center_weight=visreg_center_weight if regularizer == "visreg" else 0.0,
        visreg_sketch_weight=visreg_sketch_weight if regularizer == "visreg" else 0.0,
        visreg_projection_count=visreg_projection_count,
    )
    loss = alignment_loss + regularization
    metrics = {
        "alignment_loss": float(alignment_loss.detach().cpu()),
        "future_alignment_loss": float(future_alignment_loss.detach().cpu()),
        "context_token_loss": float(context_token_loss.detach().cpu()),
        "context_token_weight": float(context_token_weight),
        "context_token_target_count": float(context_token_target_count),
        "temporal_straightening_loss": float(temporal_straightening_loss.detach().cpu()),
        "temporal_straightening_weight": float(temporal_straightening_weight),
        "regularization_loss": float(regularization.detach().cpu()),
        "regularizer": 1.0 if regularizer == "visreg" else 0.0,
        "visreg_center_loss": float(visreg_center.detach().cpu()),
        "visreg_center_weight": float(visreg_center_weight if regularizer == "visreg" else 0.0),
        "variance_loss": float(variance.detach().cpu()),
        "visreg_sketch_loss": float(visreg_sketch.detach().cpu()),
        "visreg_sketch_weight": float(visreg_sketch_weight if regularizer == "visreg" else 0.0),
        "visreg_projection_count": float(visreg_projection_count if regularizer == "visreg" else 0),
        "context_embedding_std": float(
            context_for_variance.detach().float().std(dim=0, unbiased=False).mean().cpu()
        ),
        "pred_embedding_std": float(
            pred_for_variance.detach().float().std(dim=0, unbiased=False).mean().cpu()
        ),
        "target_embedding_std": float(
            target_for_metrics.detach().float().std(dim=0, unbiased=False).mean().cpu()
        ),
        "context_effective_rank": float(_effective_rank(context_for_variance.detach()).cpu()),
        "pred_effective_rank": float(_effective_rank(pred_for_variance.detach()).cpu()),
        "target_effective_rank": float(_effective_rank(target_for_metrics.detach()).cpu()),
        "valid_target_fraction": valid_fraction,
        "target_pair_count": float(target_pair_count),
        "deep_supervision_layer_count": float(max(1, len(deep_supervision_layers))),
    }
    return loss, metrics


def _run_epoch(
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    horizon_ids: torch.Tensor | None,
    mask_ratio: float,
    block_count: int,
    mask_mode: str,
    ema_momentum: float,
    regularizer: str,
    variance_weight: float,
    min_std: float,
    visreg_center_weight: float,
    visreg_sketch_weight: float,
    visreg_projection_count: int,
    temporal_straightening_weight: float,
    dense_tokens: bool,
    motion_conditioning: bool,
    deep_supervision_layers: tuple[int, ...],
    bins: int,
    metadata_channels: bool,
    navigation_feature_count: int,
    action_feature_mean: torch.Tensor | None,
    action_feature_std: torch.Tensor | None,
    context_token_weight: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    encoder.train(train_mode)
    predictor.train(train_mode)
    target_encoder.eval()
    use_amp = scaler is not None and device.type == "cuda"
    rows: list[dict[str, float]] = []

    for batch in loader:
        future_x = None
        future_mask = None
        if isinstance(batch, list | tuple):
            x, future_x, future_mask = batch
            future_x = future_x.to(device=device, non_blocking=True)
            future_mask = future_mask.to(device=device, non_blocking=True)
        else:
            x = batch
        x = x.to(device=device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss, metrics = _jepa_loss(
                encoder,
                target_encoder,
                predictor,
                x,
                future_x=future_x,
                future_mask=future_mask,
                horizon_ids=horizon_ids,
                mask_ratio=mask_ratio,
                block_count=block_count,
                mask_mode=mask_mode,
                regularizer=regularizer,
                variance_weight=variance_weight,
                min_std=min_std,
                visreg_center_weight=visreg_center_weight,
                visreg_sketch_weight=visreg_sketch_weight,
                visreg_projection_count=visreg_projection_count,
                temporal_straightening_weight=temporal_straightening_weight,
                dense_tokens=dense_tokens,
                motion_conditioning=motion_conditioning,
                deep_supervision_layers=deep_supervision_layers,
                bins=bins,
                metadata_channels=metadata_channels,
                navigation_feature_count=navigation_feature_count,
                action_feature_mean=action_feature_mean,
                action_feature_std=action_feature_std,
                context_token_weight=context_token_weight,
            )
        if optimizer is not None:
            if scaler is None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*encoder.parameters(), *predictor.parameters()],
                    1.0,
                )
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [*encoder.parameters(), *predictor.parameters()],
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            _update_ema(target_encoder, encoder, momentum=ema_momentum)
        rows.append({"loss": float(loss.detach().cpu()), **metrics})

    if not rows:
        return {"loss": float("nan")}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def pretrain_jepa(
    *,
    cache_path: str | Path,
    output_dir: str | Path,
    epochs: int = 120,
    batch_size: int = 128,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device_name: str = "auto",
    pretrain_splits: tuple[str, ...] = ("train",),
    validation_splits: tuple[str, ...] = ("validation",),
    temporal_horizons_ms: tuple[int, ...] = (20, 60, 100, 240, 500),
    max_target_slop_ms: int = 10,
    mask_ratio: float = 0.45,
    block_count: int = 4,
    mask_mode: str = "spatial",
    ema_momentum: float = 0.99,
    regularizer: str = "variance",
    variance_weight: float = 1.0,
    min_std: float = 0.05,
    visreg_center_weight: float = 1.0,
    visreg_sketch_weight: float = 1.0,
    visreg_projection_count: int = 32,
    temporal_straightening_weight: float = 0.0,
    dense_tokens: bool = True,
    motion_conditioning: bool = True,
    deep_supervision_layers: tuple[int, ...] = (),
    dense_predictor: str = "mlp",
    context_token_weight: float = 0.0,
    model_name: str = "tiny-cnn",
    navigation_mode: str = "enabled",
) -> dict[str, Any]:
    """Pretrain an encoder with a JEPA-style latent prediction objective."""

    if navigation_mode not in ("enabled", "disabled"):
        raise ValueError("navigation_mode must be 'enabled' or 'disabled'")

    if epochs <= 0:
        msg = "epochs must be positive."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)
    if dense_predictor not in {"mlp", "transformer"}:
        msg = "dense_predictor must be one of {'mlp', 'transformer'}."
        raise ValueError(msg)
    if regularizer not in {"variance", "visreg"}:
        msg = "regularizer must be one of {'variance', 'visreg'}."
        raise ValueError(msg)
    if mask_mode not in {"spatial", "tubelet"}:
        msg = "mask_mode must be one of {'spatial', 'tubelet'}."
        raise ValueError(msg)
    if context_token_weight < 0.0:
        msg = "context_token_weight must be non-negative."
        raise ValueError(msg)
    if visreg_sketch_weight < 0.0:
        msg = "visreg_sketch_weight must be non-negative."
        raise ValueError(msg)
    if visreg_center_weight < 0.0:
        msg = "visreg_center_weight must be non-negative."
        raise ValueError(msg)
    if visreg_projection_count <= 0:
        msg = "visreg_projection_count must be positive."
        raise ValueError(msg)
    if temporal_straightening_weight < 0.0:
        msg = "temporal_straightening_weight must be non-negative."
        raise ValueError(msg)
    _set_seed(seed)

    cache = np.load(cache_path, allow_pickle=False)
    validate_voxel_cache(cache)
    x = cache["x"]
    if navigation_mode == "disabled":
        if bool(cache.get("navigation_channels", False)):
            nav_count = len(cache["navigation_feature_names"])
            x[:, -nav_count:, :, :] = 0.0

    split = cache["split"].astype(str)
    bins = int(cache["bins"]) if "bins" in cache.files else int(x.shape[1] // 2)
    metadata_channels = _cache_bool(cache, "metadata_channels")
    navigation_channels = _cache_bool(cache, "navigation_channels")
    navigation_feature_names = (
        _cache_string_tuple(cache, "navigation_feature_names", NAVIGATION_FEATURE_NAMES)
        if navigation_channels
        else ()
    )
    use_temporal = bool(temporal_horizons_ms)
    use_dense_tokens = bool(dense_tokens and use_temporal)
    use_motion_conditioning = bool(motion_conditioning and use_dense_tokens)
    use_deep_supervision = bool(deep_supervision_layers and use_dense_tokens)
    use_context_token_loss = bool(use_dense_tokens and context_token_weight > 0.0)
    navigation_feature_count = len(navigation_feature_names) if use_motion_conditioning else 0
    use_action_conditioning = bool(navigation_feature_count)
    action_feature_names = (
        (*EVENT_MOTION_FEATURE_NAMES, *navigation_feature_names) if use_motion_conditioning else ()
    )
    action_feature_dim = len(action_feature_names)
    action_feature_mean: torch.Tensor | None = None
    action_feature_std: torch.Tensor | None = None
    objective = _objective_name(
        use_temporal=use_temporal,
        dense_tokens=use_dense_tokens,
        motion_conditioning=use_motion_conditioning,
        action_conditioning=use_action_conditioning,
        deep_supervision=use_deep_supervision,
        dense_predictor=dense_predictor,
        context_token_weight=context_token_weight if use_context_token_loss else 0.0,
        regularizer=regularizer,
        mask_mode=mask_mode,
    )
    model_tag = model_name.replace("-", "_")
    train_pair_stats = None
    validation_pair_stats = None
    if use_temporal:
        missing = {
            "timestamp_us",
            "context_start_us",
            "context_end_us",
            "sequence_id",
        } - set(cache.files)
        if missing:
            msg = f"Temporal JEPA requires cache fields: {sorted(missing)}."
            raise ValueError(msg)
        timestamp_us = cache["timestamp_us"].astype(np.int64)
        context_start_us = cache["context_start_us"].astype(np.int64)
        context_end_us = cache["context_end_us"].astype(np.int64)
        sequence_id = cache["sequence_id"].astype(str)
        train_idx, train_target_idx, train_pair_stats = _build_temporal_pairs(
            split=split,
            sequence_id=sequence_id,
            timestamp_us=timestamp_us,
            context_start_us=context_start_us,
            context_end_us=context_end_us,
            split_names=pretrain_splits,
            horizons_ms=temporal_horizons_ms,
            max_target_slop_ms=max_target_slop_ms,
        )
        val_idx, val_target_idx, validation_pair_stats = _build_temporal_pairs(
            split=split,
            sequence_id=sequence_id,
            timestamp_us=timestamp_us,
            context_start_us=context_start_us,
            context_end_us=context_end_us,
            split_names=validation_splits,
            horizons_ms=temporal_horizons_ms,
            max_target_slop_ms=max_target_slop_ms,
        )
        if train_idx.size == 0:
            msg = (
                "No temporal context/future pairs found for pretrain splits "
                f"{pretrain_splits}; check horizons or target slop."
            )
            raise ValueError(msg)
        if use_motion_conditioning:
            action_feature_mean, action_feature_std = _estimate_action_feature_stats(
                x,
                train_idx,
                bins=bins,
                metadata_channels=metadata_channels,
                navigation_feature_count=navigation_feature_count,
            )
    else:
        train_idx = _split_indices(split, pretrain_splits)
        val_idx = _split_indices(split, validation_splits)
        train_target_idx = np.empty((0, 0), dtype=np.int64)
        val_target_idx = np.empty((0, 0), dtype=np.int64)
        if train_idx.size == 0:
            msg = f"No samples found for pretrain splits {pretrain_splits}."
            raise ValueError(msg)
        if use_motion_conditioning:
            action_feature_mean, action_feature_std = _estimate_action_feature_stats(
                x,
                train_idx,
                bins=bins,
                metadata_channels=metadata_channels,
                navigation_feature_count=navigation_feature_count,
            )

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if use_temporal:
        train_dataset: Dataset[Any] = TemporalVoxelPairDataset(x, train_idx, train_target_idx)
        val_dataset: Dataset[Any] | None = (
            TemporalVoxelPairDataset(x, val_idx, val_target_idx) if val_idx.size else None
        )
        horizon_ids = torch.arange(len(temporal_horizons_ms), dtype=torch.long, device=device)
    else:
        train_dataset = VoxelOnlyDataset(x, train_idx)
        val_dataset = VoxelOnlyDataset(x, val_idx) if val_idx.size else None
        horizon_ids = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        if val_dataset is not None
        else None
    )

    encoder = build_encoder(model_name, in_channels=int(x.shape[1])).to(device)
    target_encoder = build_encoder(model_name, in_channels=int(x.shape[1])).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    for param in target_encoder.parameters():
        param.requires_grad_(False)
    predictor: nn.Module
    if use_dense_tokens:
        dense_predictor_kwargs = {
            "dim": encoder.output_dim,
            "horizon_count": len(temporal_horizons_ms) + int(use_context_token_loss),
            "motion_dim": action_feature_dim,
            "layer_count": max(deep_supervision_layers) + 1 if use_deep_supervision else 0,
        }
        if dense_predictor == "transformer":
            predictor = DenseTemporalTransformerJEPAPredictor(**dense_predictor_kwargs).to(device)
        else:
            predictor = DenseTemporalJEPAPredictor(**dense_predictor_kwargs).to(device)
    elif use_temporal:
        predictor = TemporalJEPAPredictor(
            dim=encoder.output_dim,
            horizon_count=len(temporal_horizons_ms),
        ).to(device)
    else:
        predictor = JEPAPredictor(dim=encoder.output_dim).to(device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *predictor.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "jepa_encoder_best.pt"
    last_path = output / "jepa_encoder_last.pt"
    history_path = output / "history.jsonl"
    metrics_path = output / "metrics.json"
    best_score = float("inf")
    best_epoch = -1
    best_selected_by = "validation_loss" if val_loader is not None else "training_loss"
    start_time = time.perf_counter()
    history: list[dict[str, Any]] = []
    conditioning_metadata = {
        "motion_conditioning": use_motion_conditioning,
        "action_conditioning": use_action_conditioning,
        "uses_navigation_action_conditioning": use_action_conditioning,
        "metadata_channels": metadata_channels,
        "navigation_channels": navigation_channels,
        "event_motion_feature_names": list(EVENT_MOTION_FEATURE_NAMES)
        if use_motion_conditioning
        else [],
        "navigation_feature_names": list(navigation_feature_names),
        "action_feature_names": list(action_feature_names),
        "action_feature_dim": action_feature_dim,
        "motion_feature_dim": action_feature_dim,
        "action_feature_normalization": use_motion_conditioning,
        "action_feature_normalization_source": (
            "pretrain_context_indices_train_only" if use_motion_conditioning else None
        ),
        "action_feature_mean": action_feature_mean.tolist()
        if action_feature_mean is not None
        else [],
        "action_feature_std": action_feature_std.tolist() if action_feature_std is not None else [],
    }

    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, epochs + 1):
            train_metrics = _run_epoch(
                encoder,
                target_encoder,
                predictor,
                train_loader,
                optimizer,
                device=device,
                scaler=scaler,
                horizon_ids=horizon_ids,
                mask_ratio=mask_ratio,
                block_count=block_count,
                mask_mode=mask_mode,
                ema_momentum=ema_momentum,
                regularizer=regularizer,
                variance_weight=variance_weight,
                min_std=min_std,
                visreg_center_weight=visreg_center_weight,
                visreg_sketch_weight=visreg_sketch_weight,
                visreg_projection_count=visreg_projection_count,
                temporal_straightening_weight=temporal_straightening_weight,
                dense_tokens=use_dense_tokens,
                motion_conditioning=use_motion_conditioning,
                deep_supervision_layers=deep_supervision_layers if use_deep_supervision else (),
                bins=bins,
                metadata_channels=metadata_channels,
                navigation_feature_count=navigation_feature_count,
                action_feature_mean=action_feature_mean,
                action_feature_std=action_feature_std,
                context_token_weight=context_token_weight if use_context_token_loss else 0.0,
            )
            validation_metrics = None
            if val_loader is not None:
                with torch.no_grad():
                    validation_metrics = _run_epoch(
                        encoder,
                        target_encoder,
                        predictor,
                        val_loader,
                        None,
                        device=device,
                        scaler=None,
                        horizon_ids=horizon_ids,
                        mask_ratio=mask_ratio,
                        block_count=block_count,
                        mask_mode=mask_mode,
                        ema_momentum=ema_momentum,
                        regularizer=regularizer,
                        variance_weight=variance_weight,
                        min_std=min_std,
                        visreg_center_weight=visreg_center_weight,
                        visreg_sketch_weight=visreg_sketch_weight,
                        visreg_projection_count=visreg_projection_count,
                        temporal_straightening_weight=temporal_straightening_weight,
                        dense_tokens=use_dense_tokens,
                        motion_conditioning=use_motion_conditioning,
                        deep_supervision_layers=(
                            deep_supervision_layers if use_deep_supervision else ()
                        ),
                        bins=bins,
                        metadata_channels=metadata_channels,
                        navigation_feature_count=navigation_feature_count,
                        action_feature_mean=action_feature_mean,
                        action_feature_std=action_feature_std,
                        context_token_weight=(
                            context_token_weight if use_context_token_loss else 0.0
                        ),
                    )
            score = (
                validation_metrics["loss"]
                if validation_metrics is not None
                else train_metrics["loss"]
            )
            row = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(row)
            history_file.write(json.dumps(row, sort_keys=True) + "\n")
            history_file.flush()
            if score < best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "model": f"{model_tag}_jepa",
                        "model_name": model_name,
                        "objective": objective,
                        "encoder_state_dict": encoder.state_dict(),
                        "target_encoder_state_dict": target_encoder.state_dict(),
                        "predictor_state_dict": predictor.state_dict(),
                        "epoch": epoch,
                        "checkpoint_role": "best",
                        "checkpoint_selected_by": best_selected_by,
                        "cache_path": str(cache_path),
                        "seed": seed,
                        "in_channels": int(x.shape[1]),
                        "pretrain_splits": list(pretrain_splits),
                        "validation_splits": list(validation_splits),
                        "temporal_horizons_ms": list(temporal_horizons_ms),
                        "max_target_slop_ms": max_target_slop_ms,
                        "mask_mode": mask_mode,
                        "dense_tokens": use_dense_tokens,
                        "dense_predictor": dense_predictor if use_dense_tokens else None,
                        "context_token_weight": context_token_weight
                        if use_context_token_loss
                        else 0.0,
                        "context_token_loss": use_context_token_loss,
                        "regularizer": regularizer,
                        "visreg_center_weight": visreg_center_weight
                        if regularizer == "visreg"
                        else 0.0,
                        "visreg_sketch_weight": visreg_sketch_weight
                        if regularizer == "visreg"
                        else 0.0,
                        "visreg_projection_count": visreg_projection_count
                        if regularizer == "visreg"
                        else 0,
                        "temporal_straightening_weight": temporal_straightening_weight,
                        **conditioning_metadata,
                        "deep_supervision_layers": list(deep_supervision_layers)
                        if use_deep_supervision
                        else [],
                        "deep_supervision_layer_conditioning": use_deep_supervision,
                        "bins": bins,
                        "encoder_name": encoder.__class__.__name__,
                    },
                    best_path,
                )

    torch.save(
        {
            "model": f"{model_tag}_jepa",
            "model_name": model_name,
            "objective": objective,
            "encoder_state_dict": encoder.state_dict(),
            "target_encoder_state_dict": target_encoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "epoch": epochs,
            "checkpoint_role": "last",
            "checkpoint_selected_by": "final_epoch",
            "cache_path": str(cache_path),
            "seed": seed,
            "in_channels": int(x.shape[1]),
            "pretrain_splits": list(pretrain_splits),
            "validation_splits": list(validation_splits),
            "temporal_horizons_ms": list(temporal_horizons_ms),
            "max_target_slop_ms": max_target_slop_ms,
            "mask_mode": mask_mode,
            "dense_tokens": use_dense_tokens,
            "dense_predictor": dense_predictor if use_dense_tokens else None,
            "context_token_weight": context_token_weight if use_context_token_loss else 0.0,
            "context_token_loss": use_context_token_loss,
            "regularizer": regularizer,
            "visreg_center_weight": visreg_center_weight if regularizer == "visreg" else 0.0,
            "visreg_sketch_weight": visreg_sketch_weight if regularizer == "visreg" else 0.0,
            "visreg_projection_count": visreg_projection_count if regularizer == "visreg" else 0,
            "temporal_straightening_weight": temporal_straightening_weight,
            **conditioning_metadata,
            "deep_supervision_layers": list(deep_supervision_layers)
            if use_deep_supervision
            else [],
            "deep_supervision_layer_conditioning": use_deep_supervision,
            "bins": bins,
            "encoder_name": encoder.__class__.__name__,
        },
        last_path,
    )
    summary: dict[str, Any] = {
        "model": f"{model_tag}_jepa",
        "model_name": model_name,
        "objective": objective,
        "cache": str(cache_path),
        "output_dir": output.as_posix(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": seed,
        "pretrain_seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "pretrain_splits": list(pretrain_splits),
        "validation_splits": list(validation_splits),
        "train_count": int(train_idx.size),
        "validation_count": int(val_idx.size),
        "temporal_horizons_ms": list(temporal_horizons_ms),
        "max_target_slop_ms": max_target_slop_ms,
        "dense_tokens": use_dense_tokens,
        "dense_predictor": dense_predictor if use_dense_tokens else None,
        "context_token_weight": context_token_weight if use_context_token_loss else 0.0,
        "context_token_loss": use_context_token_loss,
        **conditioning_metadata,
        "deep_supervision": use_deep_supervision,
        "deep_supervision_layers": list(deep_supervision_layers) if use_deep_supervision else [],
        "deep_supervision_layer_conditioning": use_deep_supervision,
        "bins": bins,
        "train_pair_stats": train_pair_stats,
        "validation_pair_stats": validation_pair_stats,
        "mask_ratio": mask_ratio,
        "block_count": block_count,
        "mask_mode": mask_mode,
        "ema_momentum": ema_momentum,
        "regularizer": regularizer,
        "variance_weight": variance_weight,
        "min_std": min_std,
        "visreg_center_weight": visreg_center_weight if regularizer == "visreg" else 0.0,
        "visreg_sketch_weight": visreg_sketch_weight if regularizer == "visreg" else 0.0,
        "visreg_projection_count": visreg_projection_count if regularizer == "visreg" else 0,
        "temporal_straightening_weight": temporal_straightening_weight,
        "best_epoch": best_epoch,
        "best_loss": best_score,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
        "checkpoint_selection": {
            "recommended_for_downstream": best_path.as_posix(),
            "recommended_role": "best",
            "selected_by": best_selected_by,
            "last_checkpoint": last_path.as_posix(),
            "last_role": "last",
        },
        "elapsed_seconds": time.perf_counter() - start_time,
        "last": history[-1] if history else None,
        "leakage_audit": {
            "uses_ttc_labels": False,
            "uses_future_events_as_ssl_targets": use_temporal,
            "motion_conditioning_uses_context_only": use_motion_conditioning,
            "action_conditioning_uses_context_only": use_action_conditioning,
            "action_feature_normalization_uses_train_only": use_motion_conditioning,
            "uses_future_navigation": False,
            "future_navigation_channels_zeroed_before_target_encoder": bool(
                navigation_feature_count
            ),
            "deep_supervision_uses_intermediate_target_layers": use_deep_supervision,
            "deep_supervision_layer_conditioning": use_deep_supervision,
            "context_token_loss_uses_current_context_only": use_context_token_loss,
            "tubelet_masking_uses_context_event_channels_only": mask_mode == "tubelet",
            "tubelet_masking_preserves_auxiliary_channels": mask_mode == "tubelet",
            "visreg_uses_batch_embeddings_only": regularizer == "visreg",
            "visreg_uses_ttc_labels": False,
            "temporal_straightening_uses_predictions_only": temporal_straightening_weight > 0.0,
            "targets_cross_sequence_boundary": False,
            "targets_cross_split_boundary": False,
            "target_timestamps_are_after_context": use_temporal,
            "target_windows_are_disjoint": use_temporal,
            "target_window_semantics": (
                "disjoint_window_start_after_context_plus_horizon" if use_temporal else None
            ),
            "collapse_statistics_mix_token_positions": False,
            "supervised_labels_reserved_for_finetune_only": True,
        },
    }
    write_structured(metrics_path, summary)
    ensure_parent(output / "encoder_config.json")
    write_structured(
        output / "encoder_config.json",
        {
            "in_channels": int(x.shape[1]),
            "encoder": encoder.__class__.__name__,
            "model_name": model_name,
            "output_dim": int(encoder.output_dim),
        },
    )
    return summary
