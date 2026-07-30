"""Efficient staged trainer for EvTTC Garl/OGE architecture gates."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Subset

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset, ShardLocalSampler
from e_jepa_ttc.evaluation.bootstrap import sequence_bootstrap_interval
from e_jepa_ttc.evaluation.object_ttc import (
    grouped_ttc_selection_components,
    object_ttc_metrics,
)
from e_jepa_ttc.models.mask_decoder import boxes_to_soft_masks, foreground_mask_loss
from e_jepa_ttc.models.object_geo_jepa_ttc import ObjectGeometryJEPATTC, OGEConfig, OGEOutput
from e_jepa_ttc.training.health_monitor import embedding_health
from e_jepa_ttc.utils.io import write_structured


@dataclass(frozen=True)
class OGETrainerConfig:
    """Training controls chosen for the 12 GB VRAM / 32 GB RAM host."""

    epochs: int = 24
    batch_size: int = 8
    gradient_accumulation: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    precision: str = "bf16"
    num_workers: int = 4
    early_stopping_patience: int = 4
    early_stopping_min_epochs: int = 6
    early_stopping_min_delta_relative: float = 0.003
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    log_ttc_loss_weight: float = 1.0
    inverse_nll_loss_weight: float = 0.0
    mask_loss_weight: float = 0.25
    risk_loss_weight: float = 0.0
    router_balance_weight: float = 0.01
    residual_penalty_weight: float = 0.02
    collapse_patience: int = 3
    collapse_dimension_fraction: float = 0.80
    seed: int = 7

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.gradient_accumulation) <= 0:
            raise ValueError("epochs, batch_size and gradient_accumulation must be positive.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16.")
        if self.num_workers < 0 or self.early_stopping_patience < 0 or self.collapse_patience <= 0:
            raise ValueError("Worker and patience counts must be non-negative.")
        loss_weights = (
            self.log_ttc_loss_weight,
            self.inverse_nll_loss_weight,
            self.mask_loss_weight,
            self.risk_loss_weight,
            self.router_balance_weight,
            self.residual_penalty_weight,
        )
        if any(weight < 0.0 for weight in loss_weights):
            raise ValueError("Loss weights must be non-negative.")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        repository_root = Path(__file__).resolve().parents[3]
        return (
            subprocess.check_output(
                ["git", "-C", str(repository_root), "rev-parse", "HEAD"]
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_hash() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[1]
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _state_dict_hash(module: nn.Module) -> str:
    """Hash parameters and buffers in a stable key order."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _oge_config_from_checkpoint(payload: dict[str, Any]) -> OGEConfig:
    """Rebuild a config while preserving known ``init=False`` audit fields."""

    definitions = {definition.name: definition for definition in fields(OGEConfig)}
    unknown = sorted(set(payload) - set(definitions))
    if unknown:
        raise ValueError(f"Checkpoint model_config contains unknown fields: {unknown}.")
    arguments = {
        name: value
        for name, value in payload.items()
        if definitions[name].init
    }
    return OGEConfig(**arguments)


def _selection_hash(
    train_indices: list[int],
    validation_indices: list[int],
) -> str:
    payload = {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _indices(length: int, maximum: int | None) -> list[int]:
    if maximum is None or maximum >= length:
        return list(range(length))
    if maximum <= 0:
        raise ValueError("Sample limits must be positive.")
    return np.linspace(0, length - 1, maximum, dtype=np.int64).tolist()


def _loader(
    dataset: EAPObjectCacheDataset,
    indices: list[int],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    train: bool,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor | list[str]]]:
    subset = Subset(dataset, indices)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        # Every selected sample contributes to every arm.  Dropping a
        # remainder changed the effective dataset and was unnecessary because
        # these models do not use BatchNorm.
        "drop_last": False,
    }
    if train:
        kwargs["sampler"] = ShardLocalSampler(dataset, source_indices=indices, seed=seed)
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(subset, **kwargs)


def _tensor(
    batch: dict[str, torch.Tensor | list[str]],
    key: str,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor field {key!r}.")
    return value.to(device=device, dtype=dtype, non_blocking=device.type == "cuda")


def _shutdown_loader(loader: DataLoader[Any]) -> None:
    """Release persistent Windows workers deterministically after a run."""

    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if hasattr(loader, "_iterator"):
        loader._iterator = None


def _forward(
    model: ObjectGeometryJEPATTC,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> OGEOutput:
    end_us = _tensor(batch, "context_window_end_us", device, dtype=torch.float32)
    times_s = (end_us - end_us[:, :1]) * 1e-6
    context_events = _tensor(batch, "context_events", device, dtype=torch.float32)
    if model.config.backbone == "base_event_tubelet":
        if "context_event_metadata" not in batch:
            raise ValueError("BASE backbone requires exact causal event metadata.")
        metadata = _tensor(
            batch,
            "context_event_metadata",
            device,
            dtype=torch.float32,
        )
        actions = _tensor(
            batch,
            "context_ego_actions",
            device,
            dtype=torch.float32,
        )
        action_valid = _tensor(batch, "context_ego_action_mask", device).to(
            dtype=torch.float32
        )
        auxiliary = torch.cat((metadata, actions, action_valid[..., None]), dim=-1)
        auxiliary = auxiliary[..., None, None].expand(
            -1,
            -1,
            -1,
            context_events.shape[-2],
            context_events.shape[-1],
        )
        context_events = torch.cat((context_events, auxiliary), dim=2)
    intrinsics = (
        _tensor(
            batch,
            "context_intrinsics_normalized",
            device,
            dtype=torch.float32,
        )
        if "context_intrinsics_normalized" in batch
        else None
    )
    return model(
        context_events,
        context_times_s=times_s,
        context_boxes=_tensor(batch, "context_boxes", device, dtype=torch.float32),
        context_object_mask=_tensor(batch, "context_object_mask", device).bool(),
        context_ego_actions=_tensor(
            batch,
            "context_ego_actions",
            device,
            dtype=torch.float32,
        ),
        context_ego_action_mask=_tensor(batch, "context_ego_action_mask", device).bool(),
        context_intrinsics_normalized=intrinsics,
    )


def object_geo_loss(
    prediction: OGEOutput,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    *,
    config: OGEConfig,
    trainer: OGETrainerConfig,
) -> dict[str, torch.Tensor]:
    """Supervise TTC, risk, optional masks, router and residual magnitude."""

    target_ttc = _tensor(batch, "ttc_s", device, dtype=torch.float32).reshape(-1)
    target_inverse = target_ttc.clamp_min(1e-3).reciprocal()
    variance = prediction.inverse_ttc_log_variance.exp()
    inverse_nll = (
        0.5 * (prediction.inverse_ttc_mean - target_inverse).square() / variance
        + 0.5 * prediction.inverse_ttc_log_variance
    ).mean()
    log_ttc = functional.smooth_l1_loss(
        prediction.ttc_seconds.clamp_min(1e-3).log(),
        target_ttc.clamp_min(1e-3).log(),
        # Exact historical BASE downstream loss.
        beta=0.25,
    )
    thresholds = torch.tensor(
        config.risk_thresholds_s,
        device=device,
        dtype=target_ttc.dtype,
    )
    risk_target = (target_ttc[:, None] <= thresholds[None]).to(target_ttc.dtype)
    risk = functional.binary_cross_entropy_with_logits(
        prediction.risk_logits[:, 0],
        risk_target,
    )
    zero = target_ttc.new_zeros(())
    mask_total = zero
    if prediction.mask_logits is not None:
        boxes = _tensor(batch, "context_boxes", device, dtype=torch.float32)[:, -1, 0]
        target_mask = boxes_to_soft_masks(boxes, prediction.mask_logits.shape[-2:])
        mask_total = foreground_mask_loss(
            prediction.mask_logits[:, -1],
            target_mask,
        )["mask_total"]
    if prediction.refined_mask_logits is not None:
        boxes = _tensor(batch, "context_boxes", device, dtype=torch.float32)[:, -1, 0]
        target_mask = boxes_to_soft_masks(boxes, prediction.refined_mask_logits.shape[-2:])
        mask_total = (
            mask_total
            + foreground_mask_loss(
                prediction.refined_mask_logits,
                target_mask,
            )["mask_total"]
        )
    router = prediction.diagnostics["router_balance_loss"]
    residual = prediction.diagnostics["residual_fraction"]
    total = (
        trainer.log_ttc_loss_weight * log_ttc
        + trainer.inverse_nll_loss_weight * inverse_nll
        + trainer.risk_loss_weight * risk
        + trainer.mask_loss_weight * mask_total
        + trainer.router_balance_weight * router
        + trainer.residual_penalty_weight * residual
    )
    return {
        "total": total,
        "log_ttc": log_ttc,
        "inverse_nll": inverse_nll,
        "risk": risk,
        "mask": mask_total,
        "router_balance": router,
        "residual_fraction": residual,
    }


def _autocast(
    device: torch.device,
    precision: str,
) -> torch.amp.autocast_mode.autocast:
    enabled = device.type == "cuda" and precision != "fp32"
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, enabled=enabled, dtype=dtype)


def _evaluate(
    model: ObjectGeometryJEPATTC,
    loader: DataLoader[dict[str, torch.Tensor | list[str]]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    true: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    risk: list[np.ndarray] = []
    mask_iou: list[np.ndarray] = []
    center_error: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    sequences: list[str] = []
    sample_tokens: list[str] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            output = _forward(model, batch, device)
            true.append(_tensor(batch, "ttc_s", device).reshape(-1).cpu().numpy())
            predicted.append(output.ttc_seconds.cpu().numpy())
            risk.append(torch.sigmoid(output.risk_logits[:, 0]).cpu().numpy())
            embeddings.append(output.object_token.float().cpu().numpy())
            if output.mask_logits is not None:
                boxes = _tensor(batch, "context_boxes", device, dtype=torch.float32)[:, -1, 0]
                target_mask = boxes_to_soft_masks(boxes, output.mask_logits.shape[-2:]) >= 0.5
                predicted_mask = torch.sigmoid(output.mask_logits[:, -1]) >= 0.5
                intersection = (target_mask & predicted_mask).sum(dim=(-1, -2)).float()
                union = (target_mask | predicted_mask).sum(dim=(-1, -2)).float()
                mask_iou.append((intersection / union.clamp_min(1.0)).cpu().numpy())
                if output.predicted_boxes is not None:
                    predicted_box = output.predicted_boxes[:, -1]
                    target_center = 0.5 * (boxes[:, :2] + boxes[:, 2:])
                    predicted_center = 0.5 * (predicted_box[:, :2] + predicted_box[:, 2:])
                    center_error.append(
                        ((predicted_center - target_center).square().sum(dim=-1).sqrt())
                        .cpu()
                        .numpy()
                    )
            batch_sequences = batch["sequence_id"]
            if not isinstance(batch_sequences, list):
                raise TypeError("sequence_id must collate to a list.")
            sequences.extend(str(value) for value in batch_sequences)
            batch_tokens = batch["sample_token"]
            if not isinstance(batch_tokens, list):
                raise TypeError("sample_token must collate to a list.")
            sample_tokens.extend(str(value) for value in batch_tokens)
    arrays = {
        "ttc_true": np.concatenate(true),
        "ttc_pred": np.concatenate(predicted),
        "risk_probability": np.concatenate(risk),
        "sequence_id": np.asarray(sequences),
        "sample_token": np.asarray(sample_tokens),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - started
    metrics = object_ttc_metrics(
        arrays["ttc_true"],
        arrays["ttc_pred"],
        arrays["risk_probability"],
    )
    metrics.update(
        grouped_ttc_selection_components(
            arrays["ttc_true"],
            arrays["ttc_pred"],
            arrays["sequence_id"],
        )
    )
    per_sequence = [
        float(
            np.mean(
                np.abs(
                    arrays["ttc_true"][arrays["sequence_id"] == sequence]
                    - arrays["ttc_pred"][arrays["sequence_id"] == sequence]
                )
            )
        )
        for sequence in np.unique(arrays["sequence_id"])
    ]
    metrics["sequence_macro_mae_s"] = float(np.mean(per_sequence))
    metrics["worst_sequence_mae_s"] = float(np.max(per_sequence))
    metrics["evaluation_seconds"] = evaluation_seconds
    metrics["milliseconds_per_window"] = (
        1000.0 * evaluation_seconds / max(arrays["ttc_true"].shape[0], 1)
    )
    embedding_array = np.concatenate(embeddings)
    arrays["object_embedding"] = embedding_array
    metrics["embedding_health"] = embedding_health(torch.from_numpy(embedding_array))
    if mask_iou:
        iou = np.concatenate(mask_iou)
        arrays["mask_iou"] = iou
        metrics["mask_iou_mean"] = float(iou.mean())
        metrics["target_recall_iou_0_1"] = float(np.mean(iou >= 0.1))
    if center_error:
        error = np.concatenate(center_error)
        arrays["center_error_fraction_diagonal"] = error
        metrics["center_error_fraction_diagonal"] = float(error.mean() / np.sqrt(2.0))
    return metrics, arrays


def evaluate_object_geo_ttc_checkpoint(
    *,
    checkpoint_path: str | Path,
    cache_manifest_path: str | Path,
    output_dir: str | Path,
    splits: tuple[str, ...] = ("validation",),
    device_name: str = "auto",
    batch_size: int = 32,
    num_workers: int = 4,
    allow_diagnostic_test: bool = False,
) -> dict[str, Any]:
    """Evaluate a frozen OGE checkpoint on explicit sequence-level splits.

    ``test`` refers only to the labelled EvTTC family-holdout diagnostic. The
    official Benchmark-10 uses a separate guarded inference path and is never
    accepted by this function.
    """

    checkpoint_source = Path(checkpoint_path)
    cache_source = Path(cache_manifest_path)
    output = Path(output_dir)
    assert_no_sealed_benchmark_paths((checkpoint_source, cache_source, output))
    requested_splits = tuple(dict.fromkeys(splits))
    allowed_splits = {"train", "validation", "calibration", "test"}
    invalid = sorted(set(requested_splits) - allowed_splits)
    if not requested_splits or invalid:
        raise ValueError(f"Invalid evaluation splits: {invalid or requested_splits}.")
    if "test" in requested_splits and not allow_diagnostic_test:
        raise PermissionError(
            "The labelled EvTTC family holdout requires --allow-diagnostic-test."
        )
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    checkpoint = torch.load(checkpoint_source, map_location=device, weights_only=False)
    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model_config or model_state_dict.")
    model_config = checkpoint["model_config"]
    if not isinstance(model_config, dict):
        raise TypeError("Checkpoint model_config must be a mapping.")
    model = ObjectGeometryJEPATTC(_oge_config_from_checkpoint(model_config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = EAPObjectCacheDataset(cache_source, splits=requested_splits)
    if len(dataset) == 0:
        dataset.close()
        raise ValueError(f"No samples found for splits {requested_splits}.")
    indices = list(range(len(dataset)))
    loader = _loader(
        dataset,
        indices,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        train=False,
        seed=int(checkpoint.get("trainer_config", {}).get("seed", 0)),
    )
    try:
        metrics, predictions = _evaluate(model, loader, device)
        unique, counts = np.unique(predictions["sequence_id"], return_counts=True)
        sequence_counts = {
            str(sequence): int(count) for sequence, count in zip(unique, counts, strict=True)
        }
        bootstrap = {
            "mae_s": sequence_bootstrap_interval(
                predictions["ttc_true"],
                predictions["ttc_pred"],
                predictions["sequence_id"],
                iterations=2000,
                confidence=0.95,
                seed=0,
            ),
            "mean_abs_relative_error": sequence_bootstrap_interval(
                predictions["ttc_true"],
                predictions["ttc_pred"],
                predictions["sequence_id"],
                metric=lambda truth, estimate: float(
                    np.mean(np.abs(estimate - truth) / np.maximum(np.abs(truth), 1e-6))
                ),
                iterations=2000,
                confidence=0.95,
                seed=0,
            ),
        }
        output.mkdir(parents=True, exist_ok=True)
        predictions_path = output / "predictions.npz"
        np.savez_compressed(predictions_path, **predictions)
        payload: dict[str, Any] = {
            "artifact_type": "evttc_oge_split_evaluation_v1",
            "checkpoint": checkpoint_source.as_posix(),
            "checkpoint_sha256": _hash_file(checkpoint_source),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_role": checkpoint.get("role"),
            "run_fingerprint": checkpoint.get("run_fingerprint"),
            "cache_manifest": cache_source.as_posix(),
            "cache_manifest_sha256": _hash_file(cache_source),
            "git_commit": _git_commit(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "splits": list(requested_splits),
            "sample_count": len(dataset),
            "sequence_counts": sequence_counts,
            "metrics": metrics,
            "sequence_bootstrap_95": bootstrap,
            "predictions": predictions_path.as_posix(),
            "diagnostic_test_opened": "test" in requested_splits,
            "scientific_scope": (
                "Sequence-disjoint EvTTC family-holdout diagnostic; not the sealed "
                "official Benchmark-10 and not an official leaderboard result."
                if "test" in requested_splits
                else "Development evaluation; no test split was consumed."
            ),
            "benchmark10_opened": False,
        }
        write_structured(output / "summary.json", payload)
        return payload
    finally:
        _shutdown_loader(loader)
        dataset.close()


def train_object_geo_ttc(
    *,
    cache_manifest_path: str | Path,
    output_dir: str | Path,
    model_config: OGEConfig,
    trainer_config: OGETrainerConfig | None = None,
    device_name: str = "auto",
    resume: bool = False,
    dry_run_fingerprint: bool = False,
) -> dict[str, Any] | str:
    """Train one gate arm and select by the frozen sequence-macro TTC score."""

    trainer = trainer_config or OGETrainerConfig()
    assert_no_sealed_benchmark_paths((cache_manifest_path, output_dir))
    manifest = json.loads(Path(cache_manifest_path).read_text(encoding="utf-8"))
    assert_no_sealed_benchmark_paths(
        Path(cache_manifest_path).parent / shard["path"] for shard in manifest["shards"]
    )
    _set_seed(trainer.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    train_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("train",))
    validation_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("validation",))
    train_indices = _indices(len(train_dataset), trainer.max_train_samples)
    validation_indices = _indices(len(validation_dataset), trainer.max_validation_samples)
    train_loader = _loader(
        train_dataset,
        train_indices,
        batch_size=trainer.batch_size,
        num_workers=trainer.num_workers,
        device=device,
        train=True,
        seed=trainer.seed,
    )
    validation_loader = _loader(
        validation_dataset,
        validation_indices,
        batch_size=trainer.batch_size * 2,
        num_workers=max(0, trainer.num_workers // 2),
        device=device,
        train=False,
        seed=trainer.seed,
    )
    model = ObjectGeometryJEPATTC(model_config).to(device)
    initial_backbone_sha256 = _state_dict_hash(model.backbone)
    initial_common_head_sha256 = _state_dict_hash(model.direct_log_ttc_head)
    sample_selection_sha256 = _selection_hash(train_indices, validation_indices)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer.learning_rate,
        weight_decay=trainer.weight_decay,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and trainer.precision == "fp16"
        else None
    )
    fingerprint_payload = {
        "git_commit": _git_commit(),
        "source_tree_sha256": _source_tree_hash(),
        "cache_manifest_sha256": _hash_file(cache_manifest_path),
        "base_encoder_checkpoint_sha256": (
            _hash_file(model_config.base_encoder_checkpoint)
            if model_config.base_encoder_checkpoint is not None
            else None
        ),
        "model_config": asdict(model_config),
        "trainer_config": asdict(trainer),
        "sample_selection_sha256": sample_selection_sha256,
        "initial_backbone_sha256": initial_backbone_sha256,
        "initial_common_head_sha256": initial_common_head_sha256,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if dry_run_fingerprint:
        train_dataset.close()
        validation_dataset.close()
        return fingerprint
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    last_path = output / "last.pt"
    weights_only_path = output / "weights_only.pt"
    resume_path = output / "resume.pt"
    history_path = output / "history.jsonl"
    best_value = float("inf")
    best_epoch = -1
    no_improvement = 0
    collapse_evaluations = 0
    start_epoch = 1
    stopped_early = False
    if resume:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["run_fingerprint"] != fingerprint:
            raise ValueError("Resume fingerprint differs from this architecture/data/config.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_value = float(state["best_value"])
        best_epoch = int(state["best_epoch"])
        no_improvement = int(state["no_improvement"])
        collapse_evaluations = int(state.get("collapse_evaluations", 0))
    elif resume_path.exists():
        raise FileExistsError(f"{resume_path} exists; pass resume=True or use a new directory.")

    def checkpoint(epoch: int, role: str) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "trainer_config": asdict(trainer),
            "epoch": epoch,
            "role": role,
            "selected_by": "validation_sequence_macro_selection_score",
            "run_fingerprint": fingerprint,
            "run_fingerprint_payload": fingerprint_payload,
        }

    history_mode = "a" if resume else "w"
    started = time.perf_counter()
    completed_epoch = start_epoch - 1
    with history_path.open(history_mode, encoding="utf-8") as history_file:
        for epoch in range(start_epoch, trainer.epochs + 1):
            model.train()
            if isinstance(train_loader.sampler, ShardLocalSampler):
                train_loader.sampler.set_epoch(epoch - 1)
            optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            batches = 0
            for step, batch in enumerate(train_loader, start=1):
                with _autocast(device, trainer.precision):
                    prediction = _forward(model, batch, device)
                    losses = object_geo_loss(
                        prediction,
                        batch,
                        device,
                        config=model_config,
                        trainer=trainer,
                    )
                    scaled_loss = losses["total"] / trainer.gradient_accumulation
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                if step % trainer.gradient_accumulation == 0 or step == len(train_loader):
                    if scaler is None:
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    else:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                batches += 1
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
            validation_metrics, _ = _evaluate(model, validation_loader, device)
            collapsed_fraction = float(
                validation_metrics["embedding_health"]["collapsed_dimension_fraction"]
            )
            if collapsed_fraction > trainer.collapse_dimension_fraction:
                collapse_evaluations += 1
            else:
                collapse_evaluations = 0
            if collapse_evaluations >= trainer.collapse_patience:
                raise RuntimeError(
                    "Embedding collapse persisted for "
                    f"{collapse_evaluations} validations ({collapsed_fraction:.1%} dimensions)."
                )
            value = float(validation_metrics["sequence_macro_selection_score"])
            row = {
                "epoch": epoch,
                "train": {name: total / max(batches, 1) for name, total in totals.items()},
                "validation": validation_metrics,
            }
            history_file.write(json.dumps(row, sort_keys=True) + "\n")
            history_file.flush()
            threshold = (
                best_value * (1.0 - trainer.early_stopping_min_delta_relative)
                if np.isfinite(best_value)
                else float("inf")
            )
            if value < threshold:
                best_value = value
                best_epoch = epoch
                no_improvement = 0
                _atomic_save(checkpoint(epoch, "best"), best_path)
            else:
                no_improvement += 1
            completed_epoch = epoch
            resume_state = checkpoint(epoch, "resume")
            resume_state.update(
                {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "best_value": best_value,
                    "best_epoch": best_epoch,
                    "no_improvement": no_improvement,
                    "collapse_evaluations": collapse_evaluations,
                }
            )
            _atomic_save(resume_state, resume_path)
            if (
                trainer.early_stopping_patience > 0
                and epoch >= trainer.early_stopping_min_epochs
                and no_improvement >= trainer.early_stopping_patience
            ):
                stopped_early = True
                break
    _atomic_save(checkpoint(completed_epoch, "last"), last_path)
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected["model_state_dict"])
    _atomic_save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "epoch": best_epoch,
            "role": "weights_only",
            "selected_by": "validation_sequence_macro_selection_score",
            "run_fingerprint": fingerprint,
        },
        weights_only_path,
    )
    resume_path.unlink(missing_ok=True)
    validation_metrics, predictions = _evaluate(model, validation_loader, device)
    np.savez_compressed(output / "validation_predictions.npz", **predictions)
    summary: dict[str, Any] = {
        "architecture": asdict(model_config),
        "trainer": asdict(trainer),
        "cache_manifest": Path(cache_manifest_path).as_posix(),
        "cache_manifest_sha256": _hash_file(cache_manifest_path),
        "run_fingerprint": fingerprint,
        "source_tree_sha256": fingerprint_payload["source_tree_sha256"],
        "git_commit": fingerprint_payload["git_commit"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "sample_selection_sha256": sample_selection_sha256,
        "initial_backbone_sha256": initial_backbone_sha256,
        "initial_common_head_sha256": initial_common_head_sha256,
        "train_batches_per_epoch": len(train_loader),
        "maximum_optimizer_steps": (
            (len(train_loader) + trainer.gradient_accumulation - 1)
            // trainer.gradient_accumulation
        )
        * trainer.epochs,
        "completed_optimizer_steps": (
            (len(train_loader) + trainer.gradient_accumulation - 1)
            // trainer.gradient_accumulation
        )
        * completed_epoch,
        "epochs_completed": completed_epoch,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
        "weights_only_checkpoint": weights_only_path.as_posix(),
        "resume_checkpoint": None,
        "checkpoint_policy": ["best", "last", "weights_only"],
        "validation": validation_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "benchmark10_opened": False,
    }
    write_structured(output / "summary.json", summary)
    _shutdown_loader(train_loader)
    _shutdown_loader(validation_loader)
    train_dataset.close()
    validation_dataset.close()
    return summary


__all__ = [
    "OGETrainerConfig",
    "evaluate_object_geo_ttc_checkpoint",
    "object_geo_loss",
    "train_object_geo_ttc",
]
