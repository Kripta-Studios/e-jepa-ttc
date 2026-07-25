"""Supervised TinyCNN training from materialized voxel caches."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from e_jepa_ttc.artifacts.protocol import get_current_protocol_identity
from e_jepa_ttc.data.ml_cache import validate_voxel_cache
from e_jepa_ttc.evaluation.metrics import regression_metrics
from e_jepa_ttc.models import build_regressor
from e_jepa_ttc.training.checkpoints import checkpoint_provenance
from e_jepa_ttc.utils.io import ensure_parent, write_structured


def _hash_file(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


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


def _split_indices_any(split: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    split_text = split.astype(str)
    mask = np.isin(split_text, np.array(names, dtype=str))
    return np.flatnonzero(mask).astype(np.int64)


def _available_split_names(split: np.ndarray) -> set[str]:
    return set(split.astype(str).tolist())


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
        **checkpoint_provenance(checkpoint_path, checkpoint),
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
    subset_manifest_path: str | Path | None = None,
    model_name: str = "tiny-cnn",
    navigation_mode: str = "enabled",
    evaluation_splits: tuple[str, ...] = ("train", "validation", "test"),
    train_splits: tuple[str, ...] = ("train",),
    validation_splits: tuple[str, ...] = ("validation",),
    dry_run_fingerprint: bool = False,
) -> dict[str, Any] | str:
    """Train a supervised TTC model on a materialized voxel cache."""

    _set_seed(seed)
    if not 0.0 < train_fraction <= 1.0:
        msg = "train_fraction must be in (0, 1]."
        raise ValueError(msg)
    if navigation_mode not in ("enabled", "disabled"):
        raise ValueError("navigation_mode must be 'enabled' or 'disabled'")
    if not evaluation_splits:
        msg = "evaluation_splits must contain at least one split."
        raise ValueError(msg)
    if not train_splits:
        msg = "train_splits must contain at least one split."
        raise ValueError(msg)
    if not validation_splits:
        msg = "validation_splits must contain at least one split."
        raise ValueError(msg)
    cache = np.load(cache_path, allow_pickle=False)
    validate_voxel_cache(cache)
    x = cache["x"]
    if navigation_mode == "disabled":
        if bool(cache.get("navigation_channels", False)):
            nav_count = len(cache["navigation_feature_names"])
            x[:, -nav_count:, :, :] = 0.0

    y_ttc = cache["y_ttc"].astype(np.float32)
    split = cache["split"].astype(str)
    y_log = np.log(np.clip(y_ttc, 1e-4, None)).astype(np.float32)
    available_splits = _available_split_names(split)
    missing_names = sorted(
        (set(train_splits) | set(validation_splits) | set(evaluation_splits)) - available_splits
    )
    if missing_names:
        msg = f"Requested split names are missing from cache: {missing_names}."
        raise ValueError(msg)

    train_idx = _split_indices_any(split, train_splits)
    val_idx = _split_indices_any(split, validation_splits)
    split_indices = {
        split_name: _split_indices(split, split_name) for split_name in available_splits
    }
    if train_idx.size == 0 or val_idx.size == 0:
        msg = "Requested train and validation splits must be non-empty."
        raise ValueError(msg)
    missing_eval_splits = [
        split_name for split_name in evaluation_splits if split_indices[split_name].size == 0
    ]
    if missing_eval_splits:
        msg = f"Requested evaluation splits are empty: {missing_eval_splits}."
        raise ValueError(msg)
    full_train_count = int(train_idx.size)
    subset_sha256 = ""
    if subset_manifest_path is not None and Path(subset_manifest_path).exists():
        manifest_path = Path(subset_manifest_path)
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        unsigned_manifest = dict(manifest_data)
        declared_subset_sha256 = unsigned_manifest.pop("sha256", None)
        canonical_manifest = json.dumps(
            unsigned_manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        computed_subset_sha256 = hashlib.sha256(canonical_manifest).hexdigest()
        if declared_subset_sha256 != computed_subset_sha256:
            raise ValueError(
                f"Subset manifest signature mismatch: {manifest_path}. "
                "Regenerate it rather than editing indices by hand."
            )
        requested_train_idx = np.array(manifest_data["global_indices"], dtype=np.int64)
        if requested_train_idx.ndim != 1 or requested_train_idx.size == 0:
            raise ValueError("Subset manifest global_indices must be a non-empty vector.")
        if np.unique(requested_train_idx).size != requested_train_idx.size:
            raise ValueError("Subset manifest contains duplicate global indices.")
        if not np.isin(requested_train_idx, train_idx).all():
            raise ValueError("Subset manifest contains indices outside the requested train split.")
        train_idx = np.sort(requested_train_idx)
        if "sequence_id" in cache:
            declared_sequences = np.asarray(manifest_data.get("sequence_ids", [])).astype(str)
            actual_sequences = cache["sequence_id"][train_idx].astype(str)
            if not np.array_equal(declared_sequences, actual_sequences):
                raise ValueError("Subset manifest sequence_ids do not match cache indices.")
        subset_sha256 = computed_subset_sha256
        print(f"Loaded subset manifest from {manifest_path} with {train_idx.size} samples.")
    elif train_fraction < 1.0:
        rng = np.random.default_rng(seed)
        if "sequence_id" in cache:
            seqs = cache["sequence_id"][train_idx]
            unique_seqs = np.sort(np.unique(seqs))
            subset_idx = []
            for seq in unique_seqs:
                seq_idx = train_idx[seqs == seq]
                shuffled = rng.permutation(seq_idx)
                subset_count = max(1, int(round(seq_idx.size * train_fraction)))
                subset_idx.append(shuffled[:subset_count])
            train_idx = np.sort(np.concatenate(subset_idx)).astype(np.int64)
        else:
            shuffled = rng.permutation(train_idx)
            subset_count = max(1, int(round(train_idx.size * train_fraction)))
            train_idx = np.sort(shuffled[:subset_count]).astype(np.int64)

        if subset_manifest_path is not None:
            manifest_path = Path(subset_manifest_path)
            ensure_parent(manifest_path)

            payload = {
                "global_indices": train_idx.tolist(),
                "sequence_ids": (
                    cache["sequence_id"][train_idx].astype(str).tolist()
                    if "sequence_id" in cache
                    else []
                ),
                "ttc_bins": [],  # Could compute bins here if needed
                "source_split_hash": "",
                "seed": seed,
                "requested_fraction": train_fraction,
                "effective_fraction": float(train_idx.size / full_train_count),
            }
            payload_bytes = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            subset_sha256 = hashlib.sha256(payload_bytes).hexdigest()
            payload["sha256"] = subset_sha256

            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved subset manifest to {manifest_path}")

    commit = _get_git_commit()
    cache_sha256 = _hash_file(cache_path)
    split_manifest_sha256 = (
        str(cache["split_manifest_sha256"]) if "split_manifest_sha256" in cache else ""
    )

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model_tag = model_name.replace("-", "_")

    train_dataset = VoxelCacheDataset(x, y_log, train_idx)
    val_dataset = VoxelCacheDataset(x, y_log, val_idx)
    datasets = {
        split_name: VoxelCacheDataset(x, y_log, indices)
        for split_name, indices in split_indices.items()
    }
    datasets["__train_selection__"] = train_dataset
    datasets["__validation_selection__"] = val_dataset
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
    run_fingerprint_payload = {
        "git_commit": commit,
        "protocol_version": get_current_protocol_identity()[0],
        "protocol_sha256": get_current_protocol_identity()[1],
        "cache_sha256": cache_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "subset_manifest_sha256": subset_sha256,
        "model_name": model_name,
        "resolved_model_config": {
            "in_channels": int(x.shape[1]),
            "width": getattr(model, "width", 48) if hasattr(model, "width") else 48,
        },
        "navigation_mode": navigation_mode,
        "label_fraction": train_fraction,
        "seed": seed,
        "pretraining_checkpoint_sha256": pretrained_encoder.get("checkpoint_sha256", "")
        if pretrained_encoder
        else "",
        "optimizer_config": {"learning_rate": learning_rate, "weight_decay": weight_decay},
        "training_steps": epochs,
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(run_fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if dry_run_fingerprint:
        return run_fingerprint

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
                        "checkpoint_role": "best",
                        "checkpoint_selected_by": "validation_mae",
                        "cache_path": str(cache_path),
                        "seed": seed,
                        "pretrained_encoder": pretrained_encoder,
                        "in_channels": int(x.shape[1]),
                        "cache_sha256": cache_sha256,
                        "split_manifest_sha256": split_manifest_sha256,
                        "subset_manifest_sha256": subset_sha256,
                        "navigation_mode": navigation_mode,
                        "label_fraction": train_fraction,
                        "protocol_version": get_current_protocol_identity()[0],
                        "protocol_sha256": get_current_protocol_identity()[1],
                        "git_commit": commit,
                        "resolved_model_config": {
                            "in_channels": int(x.shape[1]),
                            "width": getattr(model, "width", 48) if hasattr(model, "width") else 48,
                        },
                        "run_fingerprint": run_fingerprint,
                        "run_fingerprint_payload": run_fingerprint_payload,
                    },
                    best_path,
                )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "epoch": epochs,
            "checkpoint_role": "last",
            "checkpoint_selected_by": "final_epoch",
            "cache_path": str(cache_path),
            "seed": seed,
            "pretrained_encoder": pretrained_encoder,
            "in_channels": int(x.shape[1]),
            "cache_sha256": cache_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "subset_manifest_sha256": subset_sha256,
            "navigation_mode": navigation_mode,
            "label_fraction": train_fraction,
            "protocol_version": get_current_protocol_identity()[0],
            "protocol_sha256": get_current_protocol_identity()[1],
            "git_commit": commit,
            "resolved_model_config": {
                "in_channels": int(x.shape[1]),
                "width": getattr(model, "width", 48) if hasattr(model, "width") else 48,
            },
            "run_fingerprint": run_fingerprint,
            "run_fingerprint_payload": run_fingerprint_payload,
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
        "downstream_seed": seed,
        "pretrain_seed": (
            pretrained_encoder.get("source_seed") if pretrained_encoder is not None else None
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "pretrained_encoder": pretrained_encoder,
        "freeze_encoder": freeze_encoder,
        "train_fraction": train_fraction,
        "train_splits": list(train_splits),
        "validation_splits": list(validation_splits),
        "evaluation_splits": list(evaluation_splits),
        "full_train_count": full_train_count,
        "effective_train_count": int(train_idx.size),
        "best_epoch": best_epoch,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "splits": split_results,
        "cache_sha256": cache_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "subset_manifest_sha256": subset_sha256,
        "navigation_mode": navigation_mode,
        "protocol_version": get_current_protocol_identity()[0],
        "protocol_sha256": get_current_protocol_identity()[1],
        "git_commit": commit,
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_payload": run_fingerprint_payload,
        "final_test_opened": False,
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
    evaluation_splits: tuple[str, ...] = ("validation",),
    model_name: str | None = None,
    allow_final_test_evaluation: bool = False,
) -> dict[str, Any]:
    """Evaluate a saved supervised TTC checkpoint without retraining."""

    if not evaluation_splits:
        msg = "evaluation_splits must contain at least one split."
        raise ValueError(msg)
    if not allow_final_test_evaluation and any(
        "test" in s or "CPLA-high" in s for s in evaluation_splits
    ):
        raise ValueError(
            "Evaluation on test or CPLA-high splits requires allow_final_test_evaluation=True."
        )
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)

    cache = np.load(cache_path, allow_pickle=False)
    validate_voxel_cache(cache)
    x = cache["x"]
    y_ttc = cache["y_ttc"].astype(np.float32)
    split = cache["split"].astype(str)
    y_log = np.log(np.clip(y_ttc, 1e-4, None)).astype(np.float32)
    available_splits = _available_split_names(split)
    missing_names = sorted(set(evaluation_splits) - available_splits)
    if missing_names:
        msg = f"Requested split names are missing from cache: {missing_names}."
        raise ValueError(msg)
    split_indices = {
        split_name: _split_indices(split, split_name) for split_name in available_splits
    }
    missing_eval_splits = [
        split_name for split_name in evaluation_splits if split_indices[split_name].size == 0
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
        "checkpoint_seed": checkpoint.get("seed"),
        "downstream_seed": checkpoint.get("seed"),
        "pretrained_encoder": checkpoint.get("pretrained_encoder"),
        "pretrain_seed": (
            checkpoint.get("pretrained_encoder", {}).get("source_seed")
            if isinstance(checkpoint.get("pretrained_encoder"), dict)
            else None
        ),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "checkpoint_selected_by": checkpoint.get("checkpoint_selected_by"),
        "checkpoint_cache": checkpoint.get("cache_path"),
        "cache": str(cache_path),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_name": selected_model_name,
        "batch_size": batch_size,
        "evaluation_splits": list(evaluation_splits),
        "splits": split_results,
        "final_test_opened": allow_final_test_evaluation,
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
