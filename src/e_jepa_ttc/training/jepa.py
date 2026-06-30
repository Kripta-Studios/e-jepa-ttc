"""Self-supervised JEPA-style pretraining for voxel caches."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from e_jepa_ttc.models import TinyCNNEncoder
from e_jepa_ttc.utils.io import ensure_parent, write_structured


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
    max_slop_us = int(max_target_slop_ms * 1000)
    contexts: list[int] = []
    targets: list[np.ndarray] = []
    slop_by_horizon: list[list[float]] = [[] for _ in horizons_ms]

    for sequence in sorted(set(sequence_text[selected].tolist())):
        sequence_indices = selected[sequence_text[selected] == sequence]
        order = np.argsort(timestamps[sequence_indices], kind="stable")
        sequence_indices = sequence_indices[order]
        sequence_times = timestamps[sequence_indices]
        for local_idx, context_idx in enumerate(sequence_indices):
            row = np.full((len(horizons_ms),), -1, dtype=np.int64)
            context_time = int(sequence_times[local_idx])
            for horizon_idx, horizon_ms in enumerate(horizons_ms):
                target_time = context_time + int(horizon_ms * 1000)
                target_pos = int(np.searchsorted(sequence_times, target_time, side="left"))
                if target_pos >= sequence_times.shape[0]:
                    continue
                slop_us = int(sequence_times[target_pos] - target_time)
                if 0 <= slop_us <= max_slop_us:
                    row[horizon_idx] = int(sequence_indices[target_pos])
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


def _jepa_loss(
    encoder: TinyCNNEncoder,
    target_encoder: TinyCNNEncoder,
    predictor: nn.Module,
    x: torch.Tensor,
    *,
    future_x: torch.Tensor | None,
    future_mask: torch.Tensor | None,
    horizon_ids: torch.Tensor | None,
    mask_ratio: float,
    block_count: int,
    variance_weight: float,
    min_std: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    context = _masked_context(x, mask_ratio=mask_ratio, block_count=block_count)
    context_z = encoder(context)
    if future_x is None:
        pred = predictor(context_z)
        with torch.no_grad():
            target_z = target_encoder(x)
        pred_norm = functional.normalize(pred, dim=-1)
        target_norm = functional.normalize(target_z, dim=-1)
        alignment_loss = functional.smooth_l1_loss(pred_norm, target_norm, beta=0.1)
        pred_for_variance = pred
        target_for_metrics = target_z
        valid_fraction = 1.0
        target_pair_count = int(x.shape[0])
    else:
        if future_mask is None or horizon_ids is None:
            msg = "future_mask and horizon_ids are required for temporal JEPA."
            raise ValueError(msg)
        batch, horizon_count = future_mask.shape
        pred = predictor(context_z, horizon_ids)
        with torch.no_grad():
            target_z = target_encoder(future_x.flatten(0, 1)).view(batch, horizon_count, -1)
        pred_norm = functional.normalize(pred, dim=-1)
        target_norm = functional.normalize(target_z, dim=-1)
        per_pair_loss = functional.smooth_l1_loss(
            pred_norm,
            target_norm,
            beta=0.1,
            reduction="none",
        ).mean(dim=-1)
        valid = future_mask.to(device=per_pair_loss.device, dtype=per_pair_loss.dtype)
        valid_count = valid.sum().clamp_min(1.0)
        alignment_loss = (per_pair_loss * valid).sum() / valid_count
        valid_bool = future_mask.to(device=pred.device, dtype=torch.bool)
        pred_for_variance = pred[valid_bool]
        target_for_metrics = target_z[valid_bool]
        valid_fraction = float(valid.mean().detach().cpu())
        target_pair_count = int(future_mask.sum().detach().cpu())
    if pred_for_variance.shape[0] <= 1:
        pred_for_variance = pred.reshape(-1, pred.shape[-1])
    if target_for_metrics.shape[0] <= 1:
        target_for_metrics = target_z.reshape(-1, target_z.shape[-1])
    variance = _variance_loss(context_z, min_std=min_std) + _variance_loss(
        pred_for_variance,
        min_std=min_std,
    )
    loss = alignment_loss + variance_weight * variance
    metrics = {
        "alignment_loss": float(alignment_loss.detach().cpu()),
        "variance_loss": float(variance.detach().cpu()),
        "context_embedding_std": float(context_z.detach().float().std(dim=0).mean().cpu()),
        "pred_embedding_std": float(
            pred_for_variance.detach().float().std(dim=0).mean().cpu()
        ),
        "target_embedding_std": float(
            target_for_metrics.detach().float().std(dim=0).mean().cpu()
        ),
        "valid_target_fraction": valid_fraction,
        "target_pair_count": float(target_pair_count),
    }
    return loss, metrics


def _run_epoch(
    encoder: TinyCNNEncoder,
    target_encoder: TinyCNNEncoder,
    predictor: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    horizon_ids: torch.Tensor | None,
    mask_ratio: float,
    block_count: int,
    ema_momentum: float,
    variance_weight: float,
    min_std: float,
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
                variance_weight=variance_weight,
                min_std=min_std,
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
    ema_momentum: float = 0.99,
    variance_weight: float = 1.0,
    min_std: float = 0.05,
) -> dict[str, Any]:
    """Pretrain a TinyCNN encoder with a JEPA-style latent prediction objective."""

    if epochs <= 0:
        msg = "epochs must be positive."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)
    _set_seed(seed)

    cache = np.load(cache_path, allow_pickle=False)
    x = cache["x"]
    split = cache["split"].astype(str)
    use_temporal = bool(temporal_horizons_ms)
    train_pair_stats = None
    validation_pair_stats = None
    if use_temporal:
        missing = {"timestamp_us", "sequence_id"} - set(cache.files)
        if missing:
            msg = f"Temporal JEPA requires cache fields: {sorted(missing)}."
            raise ValueError(msg)
        timestamp_us = cache["timestamp_us"].astype(np.int64)
        sequence_id = cache["sequence_id"].astype(str)
        train_idx, train_target_idx, train_pair_stats = _build_temporal_pairs(
            split=split,
            sequence_id=sequence_id,
            timestamp_us=timestamp_us,
            split_names=pretrain_splits,
            horizons_ms=temporal_horizons_ms,
            max_target_slop_ms=max_target_slop_ms,
        )
        val_idx, val_target_idx, validation_pair_stats = _build_temporal_pairs(
            split=split,
            sequence_id=sequence_id,
            timestamp_us=timestamp_us,
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
    else:
        train_idx = _split_indices(split, pretrain_splits)
        val_idx = _split_indices(split, validation_splits)
        train_target_idx = np.empty((0, 0), dtype=np.int64)
        val_target_idx = np.empty((0, 0), dtype=np.int64)
        if train_idx.size == 0:
            msg = f"No samples found for pretrain splits {pretrain_splits}."
            raise ValueError(msg)

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

    encoder = TinyCNNEncoder(in_channels=int(x.shape[1])).to(device)
    target_encoder = TinyCNNEncoder(in_channels=int(x.shape[1])).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    for param in target_encoder.parameters():
        param.requires_grad_(False)
    predictor: nn.Module
    if use_temporal:
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
    start_time = time.perf_counter()
    history: list[dict[str, Any]] = []

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
                ema_momentum=ema_momentum,
                variance_weight=variance_weight,
                min_std=min_std,
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
                        ema_momentum=ema_momentum,
                        variance_weight=variance_weight,
                        min_std=min_std,
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
                        "model": "tiny_cnn_jepa",
                        "objective": (
                            "temporal_multihorizon" if use_temporal else "masked_same_window"
                        ),
                        "encoder_state_dict": encoder.state_dict(),
                        "target_encoder_state_dict": target_encoder.state_dict(),
                        "predictor_state_dict": predictor.state_dict(),
                        "epoch": epoch,
                        "cache_path": str(cache_path),
                        "seed": seed,
                        "in_channels": int(x.shape[1]),
                        "pretrain_splits": list(pretrain_splits),
                        "validation_splits": list(validation_splits),
                        "temporal_horizons_ms": list(temporal_horizons_ms),
                        "max_target_slop_ms": max_target_slop_ms,
                    },
                    best_path,
                )

    torch.save(
        {
            "model": "tiny_cnn_jepa",
            "objective": "temporal_multihorizon" if use_temporal else "masked_same_window",
            "encoder_state_dict": encoder.state_dict(),
            "target_encoder_state_dict": target_encoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "epoch": epochs,
            "cache_path": str(cache_path),
            "seed": seed,
            "in_channels": int(x.shape[1]),
            "pretrain_splits": list(pretrain_splits),
            "validation_splits": list(validation_splits),
            "temporal_horizons_ms": list(temporal_horizons_ms),
            "max_target_slop_ms": max_target_slop_ms,
        },
        last_path,
    )
    summary: dict[str, Any] = {
        "model": "tiny_cnn_jepa",
        "objective": "temporal_multihorizon" if use_temporal else "masked_same_window",
        "cache": str(cache_path),
        "output_dir": output.as_posix(),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": seed,
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
        "train_pair_stats": train_pair_stats,
        "validation_pair_stats": validation_pair_stats,
        "mask_ratio": mask_ratio,
        "block_count": block_count,
        "ema_momentum": ema_momentum,
        "variance_weight": variance_weight,
        "min_std": min_std,
        "best_epoch": best_epoch,
        "best_loss": best_score,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "last": history[-1] if history else None,
        "leakage_audit": {
            "uses_ttc_labels": False,
            "uses_future_events_as_ssl_targets": use_temporal,
            "targets_cross_sequence_boundary": False,
            "targets_cross_split_boundary": False,
            "target_timestamps_are_after_context": use_temporal,
            "supervised_labels_reserved_for_finetune_only": True,
        },
    }
    write_structured(metrics_path, summary)
    ensure_parent(output / "encoder_config.json")
    write_structured(
        output / "encoder_config.json",
        {
            "in_channels": int(x.shape[1]),
            "encoder": "TinyCNNEncoder",
            "output_dim": int(encoder.output_dim),
        },
    )
    return summary
