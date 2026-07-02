"""Supervised TinyCNN training from materialized voxel caches."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from e_jepa_ttc.evaluation.metrics import regression_metrics
from e_jepa_ttc.models import build_regressor
from e_jepa_ttc.utils.io import ensure_parent, write_structured


class VoxelCacheDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset backed by arrays from an `.npz` voxel cache."""

    def __init__(self, x: np.ndarray, y_log_ttc: np.ndarray, indices: np.ndarray) -> None:
        self.x = x
        self.y_log_ttc = y_log_ttc
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])
        x = torch.from_numpy(self.x[real_idx].astype(np.float32, copy=False))
        y = torch.tensor(self.y_log_ttc[real_idx], dtype=torch.float32)
        return x, y


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_indices(split: np.ndarray, name: str) -> np.ndarray:
    return np.flatnonzero(split.astype(str) == name).astype(np.int64)


def _evaluate_model(
    model: nn.Module,
    dataset: VoxelCacheDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, non_blocking=True)
            pred_log = model(x).detach().cpu().numpy()
            predictions.append(np.exp(pred_log))
            targets.append(np.exp(y.numpy()))
    elapsed = time.perf_counter() - start
    if not predictions:
        return np.empty(0), np.empty(0), elapsed
    return np.concatenate(targets), np.concatenate(predictions), elapsed


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    *,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    losses: list[float] = []
    use_amp = scaler is not None and device.type == "cuda"
    for x, y in loader:
        x = x.to(device=device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(x)
            loss = loss_fn(pred, y)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _load_pretrained_encoder(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    expected_model_name: str,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("encoder_state_dict")
    if state is None:
        msg = f"Checkpoint {checkpoint_path} does not contain encoder_state_dict."
        raise ValueError(msg)
    source_model_name = checkpoint.get("model_name", "tiny-cnn")
    if source_model_name != expected_model_name:
        msg = (
            f"Checkpoint encoder is {source_model_name!r}, but training model is "
            f"{expected_model_name!r}."
        )
        raise ValueError(msg)
    model.encoder.load_state_dict(state)
    return {
        "path": Path(checkpoint_path).as_posix(),
        "source_epoch": checkpoint.get("epoch"),
        "source_model": checkpoint.get("model"),
        "source_model_name": source_model_name,
    }


def train_tiny_cnn(
    *,
    cache_path: str | Path,
    output_dir: str | Path,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device_name: str = "auto",
    pretrained_encoder_path: str | Path | None = None,
    freeze_encoder: bool = False,
    train_fraction: float = 1.0,
    model_name: str = "tiny-cnn",
    evaluation_splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, Any]:
    """Train a supervised TTC model on a materialized voxel cache."""

    _set_seed(seed)
    if not 0.0 < train_fraction <= 1.0:
        msg = "train_fraction must be in (0, 1]."
        raise ValueError(msg)
    if not evaluation_splits:
        msg = "evaluation_splits must contain at least one split."
        raise ValueError(msg)
    unknown_eval_splits = set(evaluation_splits) - {"train", "validation", "test"}
    if unknown_eval_splits:
        msg = f"Unknown evaluation splits: {sorted(unknown_eval_splits)}."
        raise ValueError(msg)
    cache = np.load(cache_path, allow_pickle=False)
    x = cache["x"]
    y_ttc = cache["y_ttc"].astype(np.float32)
    split = cache["split"].astype(str)
    y_log = np.log(np.clip(y_ttc, 1e-4, None)).astype(np.float32)

    train_idx = _split_indices(split, "train")
    val_idx = _split_indices(split, "validation")
    test_idx = _split_indices(split, "test")
    split_indices = {"train": train_idx, "validation": val_idx, "test": test_idx}
    if train_idx.size == 0 or val_idx.size == 0:
        msg = "Cache must contain train and validation splits."
        raise ValueError(msg)
    missing_eval_splits = [
        split_name
        for split_name in evaluation_splits
        if split_indices[split_name].size == 0
    ]
    if missing_eval_splits:
        msg = f"Requested evaluation splits are empty: {missing_eval_splits}."
        raise ValueError(msg)
    full_train_count = int(train_idx.size)
    if train_fraction < 1.0:
        rng = np.random.default_rng(seed)
        subset_count = max(1, int(round(train_idx.size * train_fraction)))
        train_idx = np.sort(
            rng.choice(train_idx, size=subset_count, replace=False)
        ).astype(np.int64)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model_tag = model_name.replace("-", "_")

    train_dataset = VoxelCacheDataset(x, y_log, train_idx)
    val_dataset = VoxelCacheDataset(x, y_log, val_idx)
    datasets = {
        "train": train_dataset,
        "validation": val_dataset,
        "test": VoxelCacheDataset(x, y_log, test_idx),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_regressor(model_name, in_channels=int(x.shape[1])).to(device)
    pretrained_encoder: dict[str, Any] | None = None
    if pretrained_encoder_path is not None:
        pretrained_encoder = _load_pretrained_encoder(
            model,
            pretrained_encoder_path,
            device=device,
            expected_model_name=model_name,
        )
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad_(False)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        msg = "No trainable parameters remain after applying freeze settings."
        raise ValueError(msg)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "tiny_cnn_best.pt"
    last_path = output / "tiny_cnn_last.pt"
    metrics_path = output / "metrics.json"
    history_path = output / "history.jsonl"
    best_val_mae = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, epochs + 1):
            train_loss = _train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device=device,
                scaler=scaler,
            )
            val_true, val_pred, val_seconds = _evaluate_model(
                model,
                val_dataset,
                device=device,
                batch_size=batch_size,
            )
            val_metrics = regression_metrics(val_true, val_pred)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation": val_metrics,
                "validation_seconds": val_seconds,
            }
            history.append(row)
            history_file.write(json.dumps(row, sort_keys=True) + "\n")
            history_file.flush()
            if val_metrics["mae_s"] < best_val_mae:
                best_val_mae = val_metrics["mae_s"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "epoch": epoch,
                        "cache_path": str(cache_path),
                        "seed": seed,
                        "in_channels": int(x.shape[1]),
                    },
                    best_path,
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "epoch": epochs,
            "cache_path": str(cache_path),
            "seed": seed,
            "in_channels": int(x.shape[1]),
        },
        last_path,
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    split_results: dict[str, Any] = {}
    predictions: dict[str, list[float]] = {}
    targets: dict[str, list[float]] = {}
    for split_name in evaluation_splits:
        dataset = datasets[split_name]
        y_true, y_pred, seconds = _evaluate_model(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
        )
        split_results[split_name] = {
            "count": int(y_true.shape[0]),
            "metrics": regression_metrics(y_true, y_pred),
            "seconds": seconds,
        }
        predictions[split_name] = y_pred.astype(float).tolist()
        targets[split_name] = y_true.astype(float).tolist()

    summary: dict[str, Any] = {
        "model": f"{model_tag}_voxel_supervised",
        "model_name": model_name,
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
        "pretrained_encoder": pretrained_encoder,
        "freeze_encoder": freeze_encoder,
        "train_fraction": train_fraction,
        "evaluation_splits": list(evaluation_splits),
        "full_train_count": full_train_count,
        "effective_train_count": int(train_idx.size),
        "best_epoch": best_epoch,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "splits": split_results,
    }
    write_structured(metrics_path, summary)
    prediction_arrays: dict[str, np.ndarray] = {}
    for split_name in evaluation_splits:
        prediction_arrays[f"{split_name}_pred"] = np.array(
            predictions[split_name],
            dtype=np.float32,
        )
        prediction_arrays[f"{split_name}_true"] = np.array(
            targets[split_name],
            dtype=np.float32,
        )
    np.savez(ensure_parent(output / "predictions.npz"), **prediction_arrays)
    return summary


def evaluate_supervised_checkpoint(
    *,
    cache_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    batch_size: int = 64,
    device_name: str = "auto",
    evaluation_splits: tuple[str, ...] = ("test",),
    model_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate a saved supervised TTC checkpoint without retraining."""

    if not evaluation_splits:
        msg = "evaluation_splits must contain at least one split."
        raise ValueError(msg)
    unknown_eval_splits = set(evaluation_splits) - {"train", "validation", "test"}
    if unknown_eval_splits:
        msg = f"Unknown evaluation splits: {sorted(unknown_eval_splits)}."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)

    cache = np.load(cache_path, allow_pickle=False)
    x = cache["x"]
    y_ttc = cache["y_ttc"].astype(np.float32)
    split = cache["split"].astype(str)
    y_log = np.log(np.clip(y_ttc, 1e-4, None)).astype(np.float32)
    split_indices = {
        "train": _split_indices(split, "train"),
        "validation": _split_indices(split, "validation"),
        "test": _split_indices(split, "test"),
    }
    missing_eval_splits = [
        split_name
        for split_name in evaluation_splits
        if split_indices[split_name].size == 0
    ]
    if missing_eval_splits:
        msg = f"Requested evaluation splits are empty: {missing_eval_splits}."
        raise ValueError(msg)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_model_name = str(checkpoint.get("model_name", "tiny-cnn"))
    selected_model_name = model_name or checkpoint_model_name
    if selected_model_name != checkpoint_model_name:
        msg = (
            f"Checkpoint model is {checkpoint_model_name!r}, but evaluation model is "
            f"{selected_model_name!r}."
        )
        raise ValueError(msg)
    model = build_regressor(selected_model_name, in_channels=int(x.shape[1])).to(device)
    state = checkpoint.get("model_state_dict")
    if state is None:
        msg = f"Checkpoint {checkpoint_path} does not contain model_state_dict."
        raise ValueError(msg)
    model.load_state_dict(state)

    split_results: dict[str, Any] = {}
    predictions: dict[str, list[float]] = {}
    targets: dict[str, list[float]] = {}
    for split_name in evaluation_splits:
        dataset = VoxelCacheDataset(x, y_log, split_indices[split_name])
        y_true, y_pred, seconds = _evaluate_model(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
        )
        split_results[split_name] = {
            "count": int(y_true.shape[0]),
            "metrics": regression_metrics(y_true, y_pred),
            "seconds": seconds,
        }
        predictions[split_name] = y_pred.astype(float).tolist()
        targets[split_name] = y_true.astype(float).tolist()

    summary: dict[str, Any] = {
        "checkpoint": Path(checkpoint_path).as_posix(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "cache": str(cache_path),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_name": selected_model_name,
        "batch_size": batch_size,
        "evaluation_splits": list(evaluation_splits),
        "splits": split_results,
    }
    if output_path is not None:
        output = ensure_parent(output_path)
        write_structured(output, summary)
        prediction_arrays: dict[str, np.ndarray] = {}
        for split_name in evaluation_splits:
            prediction_arrays[f"{split_name}_pred"] = np.array(
                predictions[split_name],
                dtype=np.float32,
            )
            prediction_arrays[f"{split_name}_true"] = np.array(
                targets[split_name],
                dtype=np.float32,
            )
        np.savez(output.with_suffix(".predictions.npz"), **prediction_arrays)
    return summary
