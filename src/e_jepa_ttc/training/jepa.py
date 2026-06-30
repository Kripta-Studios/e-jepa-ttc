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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_indices(split: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    split_text = split.astype(str)
    mask = np.isin(split_text, np.array(names, dtype=str))
    return np.flatnonzero(mask).astype(np.int64)


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
    predictor: JEPAPredictor,
    x: torch.Tensor,
    *,
    mask_ratio: float,
    block_count: int,
    variance_weight: float,
    min_std: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    context = _masked_context(x, mask_ratio=mask_ratio, block_count=block_count)
    context_z = encoder(context)
    pred = predictor(context_z)
    with torch.no_grad():
        target_z = target_encoder(x)
    pred_norm = functional.normalize(pred, dim=-1)
    target_norm = functional.normalize(target_z, dim=-1)
    alignment_loss = functional.smooth_l1_loss(pred_norm, target_norm, beta=0.1)
    variance = _variance_loss(context_z, min_std=min_std) + _variance_loss(pred, min_std=min_std)
    loss = alignment_loss + variance_weight * variance
    metrics = {
        "alignment_loss": float(alignment_loss.detach().cpu()),
        "variance_loss": float(variance.detach().cpu()),
        "context_embedding_std": float(context_z.detach().float().std(dim=0).mean().cpu()),
        "pred_embedding_std": float(pred.detach().float().std(dim=0).mean().cpu()),
        "target_embedding_std": float(target_z.detach().float().std(dim=0).mean().cpu()),
    }
    return loss, metrics


def _run_epoch(
    encoder: TinyCNNEncoder,
    target_encoder: TinyCNNEncoder,
    predictor: JEPAPredictor,
    loader: DataLoader[torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
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

    for x in loader:
        x = x.to(device=device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss, metrics = _jepa_loss(
                encoder,
                target_encoder,
                predictor,
                x,
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
    train_idx = _split_indices(split, pretrain_splits)
    val_idx = _split_indices(split, validation_splits)
    if train_idx.size == 0:
        msg = f"No samples found for pretrain splits {pretrain_splits}."
        raise ValueError(msg)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    train_dataset = VoxelOnlyDataset(x, train_idx)
    val_dataset = VoxelOnlyDataset(x, val_idx) if val_idx.size else None
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
                        "encoder_state_dict": encoder.state_dict(),
                        "target_encoder_state_dict": target_encoder.state_dict(),
                        "predictor_state_dict": predictor.state_dict(),
                        "epoch": epoch,
                        "cache_path": str(cache_path),
                        "seed": seed,
                        "in_channels": int(x.shape[1]),
                        "pretrain_splits": list(pretrain_splits),
                        "validation_splits": list(validation_splits),
                    },
                    best_path,
                )

    torch.save(
        {
            "model": "tiny_cnn_jepa",
            "encoder_state_dict": encoder.state_dict(),
            "target_encoder_state_dict": target_encoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "epoch": epochs,
            "cache_path": str(cache_path),
            "seed": seed,
            "in_channels": int(x.shape[1]),
            "pretrain_splits": list(pretrain_splits),
            "validation_splits": list(validation_splits),
        },
        last_path,
    )
    summary: dict[str, Any] = {
        "model": "tiny_cnn_jepa",
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
