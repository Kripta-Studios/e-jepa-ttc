"""On-demand CARLA DVS Looming pretraining for the EvTTC BASE encoder."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.carla_looming import (
    CARLA_LOOMING_DATASET_ID,
    CarlaLoomingSequence,
    build_carla_window_sample,
    carla_window_references_ms,
    read_carla_looming_manifest,
)
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.models import build_encoder
from e_jepa_ttc.representations.voxel_grid import encode_voxel_grid
from e_jepa_ttc.training.jepa import DenseTemporalJEPAPredictor, _jepa_loss, _update_ema
from e_jepa_ttc.utils.io import read_structured, write_structured

EVTTC_BASE_INPUT_CHANNELS = 21
EVTTC_BASE_EVENT_CHANNELS = 10


@dataclass(frozen=True)
class CarlaJEPATrainerConfig:
    """Training and input controls for a 12 GB VRAM / 32 GB RAM host."""

    epochs: int = 30
    batch_size: int = 24
    gradient_accumulation: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    precision: str = "bf16"
    num_workers: int = 8
    prefetch_factor: int = 2
    context_ms: int = 100
    stride_ms: int = 50
    horizons_ms: tuple[int, ...] = (50, 100, 250)
    future_window_ms: int = 100
    max_windows_per_sequence: int = 16
    width: int = 160
    height: int = 90
    bins: int = 5
    mask_ratio: float = 0.45
    mask_blocks: int = 4
    context_token_weight: float = 0.25
    variance_weight: float = 1.0
    minimum_std: float = 0.05
    ema_start: float = 0.99
    ema_end: float = 0.9999
    early_stopping_patience: int = 6
    early_stopping_min_epochs: int = 8
    collapse_patience: int = 3
    collapse_dimension_fraction: float = 0.80
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        positive = (
            self.epochs,
            self.batch_size,
            self.gradient_accumulation,
            self.num_workers + 1,
            self.prefetch_factor,
            self.context_ms,
            self.stride_ms,
            self.future_window_ms,
            self.max_windows_per_sequence,
            self.width,
            self.height,
            self.bins,
            self.mask_blocks,
            self.collapse_patience,
        )
        if min(positive) <= 0:
            raise ValueError("CARLA JEPA integer controls must be positive.")
        if not self.horizons_ms or any(horizon <= 0 for horizon in self.horizons_ms):
            raise ValueError("CARLA JEPA horizons must contain positive values.")
        if tuple(sorted(set(self.horizons_ms))) != self.horizons_ms:
            raise ValueError("CARLA JEPA horizons must be sorted and unique.")
        if self.bins * 2 != EVTTC_BASE_EVENT_CHANNELS:
            raise ValueError("EvTTC BASE compatibility requires exactly five event bins.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16.")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must lie in [0, 1).")
        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie in [0, 1).")
        if not 0.0 < self.ema_start <= self.ema_end < 1.0:
            raise ValueError("EMA momentum must satisfy 0 < start <= end < 1.")
        if not 0.0 <= self.collapse_dimension_fraction <= 1.0:
            raise ValueError("collapse_dimension_fraction must lie in [0, 1].")


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash(path: str | Path) -> str:
    payload = read_structured(path)
    if not verify_artifact_hash(payload):
        raise ValueError(f"Structured artifact signature is invalid: {path}.")
    return str(payload["artifact_sha256"])


def _source_tree_hash() -> str:
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[1]
    for path in sorted(source_root.rglob("*.py")):
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        repository = Path(__file__).resolve().parents[3]
        return (
            subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "unknown"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _downsample_events(events: EventBatch, *, width: int, height: int) -> EventBatch:
    if width <= 0 or height <= 0:
        raise ValueError("Target event resolution must be positive.")
    if events.num_events == 0:
        return EventBatch.empty(
            width=width,
            height=height,
            sequence_id=events.sequence_id,
            t_start_us=events.t_start_us,
            t_end_us=events.t_end_us,
        )
    x = np.minimum(
        (events.x.astype(np.int64) * width) // events.width,
        width - 1,
    ).astype(np.int32)
    y = np.minimum(
        (events.y.astype(np.int64) * height) // events.height,
        height - 1,
    ).astype(np.int32)
    return EventBatch(
        x=x,
        y=y,
        t_us=events.t_us,
        polarity=events.polarity,
        width=width,
        height=height,
        sequence_id=events.sequence_id,
        t_start_us=events.t_start_us,
        t_end_us=events.t_end_us,
    )


def _base_compatible_voxel(
    events: EventBatch,
    *,
    width: int,
    height: int,
    bins: int,
) -> torch.Tensor:
    resized = _downsample_events(events, width=width, height=height)
    event_voxel = encode_voxel_grid(resized, bins=bins, normalize=True)
    tensor = np.zeros((EVTTC_BASE_INPUT_CHANNELS, height, width), dtype=np.float32)
    tensor[:EVTTC_BASE_EVENT_CHANNELS] = event_voxel
    return torch.from_numpy(tensor)


def _coverage_end_ms(sequence: CarlaLoomingSequence) -> int:
    if sequence.metadata is None or sequence.last_event_ms is None:
        return -1
    return min(
        int(math.floor(sequence.last_event_ms + sequence.metadata.dt_ms)),
        sequence.metadata.t_end_ms,
    )


class CarlaJEPAVoxelDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Build context/future voxel pairs directly from memory-mapped events."""

    def __init__(
        self,
        root: str | Path,
        sequences: list[CarlaLoomingSequence],
        config: CarlaJEPATrainerConfig,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self.samples: list[tuple[CarlaLoomingSequence, int]] = []
        earliest_target_offset = min(config.horizons_ms) + config.future_window_ms
        for sequence in sequences:
            references = carla_window_references_ms(
                sequence,
                context_ms=config.context_ms,
                stride_ms=config.stride_ms,
                max_windows=config.max_windows_per_sequence,
            )
            coverage_end = _coverage_end_ms(sequence)
            self.samples.extend(
                (sequence, int(reference))
                for reference in references
                if int(reference) + earliest_target_offset <= coverage_end
            )
        if not self.samples:
            raise ValueError("No CARLA context/future JEPA pairs satisfy the configuration.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence, reference_ms = self.samples[index]
        sample = build_carla_window_sample(
            self.root,
            sequence,
            reference_ms=reference_ms,
            context_ms=self.config.context_ms,
            horizons_ms=self.config.horizons_ms,
            future_window_ms=self.config.future_window_ms,
        )
        context = _base_compatible_voxel(
            sample.context_events,
            width=self.config.width,
            height=self.config.height,
            bins=self.config.bins,
        )
        targets: list[torch.Tensor] = []
        valid: list[bool] = []
        for horizon in self.config.horizons_ms:
            events = sample.future_events.get(horizon)
            valid.append(events is not None)
            targets.append(
                _base_compatible_voxel(
                    events,
                    width=self.config.width,
                    height=self.config.height,
                    bins=self.config.bins,
                )
                if events is not None
                else torch.zeros_like(context)
            )
        return context, torch.stack(targets), torch.tensor(valid, dtype=torch.bool)


def _sequences_for_role(
    manifest_path: str | Path,
    split_path: str | Path,
    role: str,
) -> list[CarlaLoomingSequence]:
    split = read_structured(split_path)
    if split.get("dataset_id") != CARLA_LOOMING_DATASET_ID:
        raise ValueError(f"Split {split_path} is not CARLA DVS Looming.")
    expected_manifest_hash = split.get("manifest_artifact_sha256")
    if expected_manifest_hash is not None:
        if str(expected_manifest_hash) != _artifact_hash(manifest_path):
            raise ValueError("CARLA manifest artifact does not match the signed split.")
    elif str(split.get("manifest_sha256", "")) != _hash_file(manifest_path):
        raise ValueError("Legacy CARLA manifest file hash does not match the signed split.")
    assignments = split.get("assignments")
    if not isinstance(assignments, dict) or not isinstance(assignments.get(role), list):
        raise ValueError(f"CARLA split does not define role {role!r}.")
    selected_ids = set(str(value) for value in assignments[role])
    by_id = {
        sequence.sequence_id: sequence
        for sequence in read_carla_looming_manifest(manifest_path)
    }
    missing = sorted(selected_ids - set(by_id))
    if missing:
        raise ValueError(f"CARLA split references unknown sequences: {missing[:5]}.")
    return [by_id[sequence_id] for sequence_id in sorted(selected_ids)]


def _bounded_indices(length: int, maximum: int | None) -> list[int]:
    if maximum is None or maximum >= length:
        return list(range(length))
    if maximum <= 0:
        raise ValueError("Sample limits must be positive.")
    return np.linspace(0, length - 1, maximum, dtype=np.int64).tolist()


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def _loader(
    dataset: CarlaJEPAVoxelDataset,
    *,
    maximum: int | None,
    config: CarlaJEPATrainerConfig,
    device: torch.device,
    train: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    indices = _bounded_indices(len(dataset), maximum)
    generator = torch.Generator().manual_seed(config.seed + int(not train))
    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": train,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
        "worker_init_fn": _worker_seed,
        "generator": generator,
    }
    if config.num_workers:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=config.prefetch_factor,
        )
    return DataLoader(Subset(dataset, indices), **kwargs)


def _shutdown_loader(loader: DataLoader[Any]) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if hasattr(loader, "_iterator"):
        loader._iterator = None


def _autocast(device: torch.device, precision: str) -> torch.amp.autocast_mode.autocast:
    enabled = device.type == "cuda" and precision != "fp32"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("CARLA JEPA loader produced no batches.")
    keys = rows[0]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _run_epoch(
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    config: CarlaJEPATrainerConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    horizon_ids: torch.Tensor,
    optimizer_step: int,
    total_optimizer_steps: int,
) -> tuple[dict[str, float], int]:
    train = optimizer is not None
    encoder.train(train)
    predictor.train(train)
    target_encoder.eval()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    rows: list[dict[str, float]] = []
    batch_count = len(loader)
    for batch_index, (context, future, valid) in enumerate(loader):
        context = context.to(device=device, non_blocking=True)
        future = future.to(device=device, non_blocking=True)
        valid = valid.to(device=device, non_blocking=True)
        group_start = (batch_index // config.gradient_accumulation) * config.gradient_accumulation
        group_size = min(config.gradient_accumulation, batch_count - group_start)
        with _autocast(device, config.precision):
            loss, metrics = _jepa_loss(
                encoder,
                target_encoder,
                predictor,
                context,
                future_x=future,
                future_mask=valid,
                horizon_ids=horizon_ids,
                mask_ratio=config.mask_ratio,
                block_count=config.mask_blocks,
                mask_mode="tubelet",
                regularizer="variance",
                variance_weight=config.variance_weight,
                min_std=config.minimum_std,
                visreg_center_weight=0.0,
                visreg_sketch_weight=0.0,
                visreg_projection_count=1,
                temporal_straightening_weight=0.0,
                dense_tokens=True,
                motion_conditioning=False,
                deep_supervision_layers=(),
                bins=config.bins,
                metadata_channels=True,
                navigation_feature_count=0,
                action_feature_mean=None,
                action_feature_std=None,
                context_token_weight=config.context_token_weight,
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"CARLA JEPA produced non-finite loss: {metrics}.")
        if train:
            scaled_loss = loss / group_size
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            end_group = (batch_index + 1) % config.gradient_accumulation == 0
            if end_group or batch_index + 1 == batch_count:
                parameters = [*encoder.parameters(), *predictor.parameters()]
                if scaler is None:
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                optimizer_step += 1
                progress = min(optimizer_step / max(total_optimizer_steps, 1), 1.0)
                momentum = config.ema_end - (
                    config.ema_end - config.ema_start
                ) * (math.cos(math.pi * progress) + 1.0) / 2.0
                metrics["target_encoder_divergence_l2"] = _update_ema(
                    target_encoder,
                    encoder,
                    momentum=momentum,
                )
                metrics["ema_momentum"] = momentum
        rows.append({"loss": float(loss.detach().cpu()), **metrics})
    return _aggregate(rows), optimizer_step


def _optimizer(
    parameters: list[nn.Parameter],
    config: CarlaJEPATrainerConfig,
    device: torch.device,
) -> torch.optim.AdamW:
    kwargs: dict[str, Any] = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameters, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(parameters, **kwargs)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _run_fingerprint(
    *,
    config: CarlaJEPATrainerConfig,
    manifest_path: Path,
    split_path: Path,
    source_tree_sha256: str,
) -> str:
    payload = {
        "config": asdict(config),
        "manifest_artifact_sha256": _artifact_hash(manifest_path),
        "split_artifact_sha256": _artifact_hash(split_path),
        "source_tree_sha256": source_tree_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _rng_state(loader: DataLoader[Any]) -> dict[str, Any]:
    generator = getattr(loader, "generator", None)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": generator.get_state() if generator is not None else None,
    }


def _restore_rng_state(payload: dict[str, Any], loader: DataLoader[Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch"])
    cuda_state = payload.get("cuda", [])
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    generator = getattr(loader, "generator", None)
    loader_state = payload.get("loader_generator")
    if generator is not None and loader_state is not None:
        generator.set_state(loader_state)


def _checkpoint(
    *,
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    epoch: int,
    role: str,
    config: CarlaJEPATrainerConfig,
    manifest_path: Path,
    split_path: Path,
    source_tree_sha256: str,
    git_commit: str,
) -> dict[str, Any]:
    return {
        "model": "event_tubelet_transformer_carla_jepa",
        "model_name": "event-tubelet-transformer",
        "objective": "tubeletmask_dense_temporal_token_multihorizon",
        "encoder_state_dict": encoder.state_dict(),
        "target_encoder_state_dict": target_encoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "epoch": epoch,
        "checkpoint_role": role,
        "checkpoint_selected_by": "validation_loss" if role == "best" else "final_epoch",
        "seed": config.seed,
        "in_channels": EVTTC_BASE_INPUT_CHANNELS,
        "bins": config.bins,
        "pretraining_dataset_id": CARLA_LOOMING_DATASET_ID,
        "external_pretraining": True,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _hash_file(manifest_path),
        "manifest_artifact_sha256": _artifact_hash(manifest_path),
        "split_manifest_path": split_path.as_posix(),
        "split_manifest_sha256": _hash_file(split_path),
        "split_artifact_sha256": _artifact_hash(split_path),
        "pretrain_splits": ["train"],
        "validation_splits": ["validation"],
        "temporal_horizons_ms": list(config.horizons_ms),
        "auxiliary_channel_semantics": (
            "eleven_zero_channels_for_evttc_base_shape_compatibility"
        ),
        "uses_ttc_labels": False,
        "uses_collision_labels": False,
        "uses_velocity_feature": False,
        "uses_object_diameter_feature": False,
        "benchmark10_opened": False,
        "git_commit": git_commit,
        "source_tree_sha256": source_tree_sha256,
        "trainer_config": asdict(config),
    }


def pretrain_carla_jepa(
    *,
    root: str | Path,
    manifest_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    config: CarlaJEPATrainerConfig,
    device_name: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Pretrain the exact 21-channel BASE encoder without TTC supervision."""

    assert_no_sealed_benchmark_paths([root, manifest_path, split_path, output_dir])
    _set_seed(config.seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    manifest = Path(manifest_path)
    split = Path(split_path)
    train_sequences = _sequences_for_role(manifest, split, "train")
    validation_sequences = _sequences_for_role(manifest, split, "validation")
    train_dataset = CarlaJEPAVoxelDataset(root, train_sequences, config)
    validation_dataset = CarlaJEPAVoxelDataset(root, validation_sequences, config)
    train_loader = _loader(
        train_dataset,
        maximum=config.max_train_samples,
        config=config,
        device=device,
        train=True,
    )
    validation_loader = _loader(
        validation_dataset,
        maximum=config.max_validation_samples,
        config=config,
        device=device,
        train=False,
    )

    encoder = build_encoder("event-tubelet-transformer", in_channels=21).to(device)
    target_encoder = build_encoder("event-tubelet-transformer", in_channels=21).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    target_encoder.requires_grad_(False)
    predictor = DenseTemporalJEPAPredictor(
        dim=int(encoder.output_dim),
        horizon_count=len(config.horizons_ms) + int(config.context_token_weight > 0.0),
    ).to(device)
    optimizer = _optimizer([*encoder.parameters(), *predictor.parameters()], config, device)
    steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation)
    total_optimizer_steps = steps_per_epoch * config.epochs
    scheduler = _scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_fraction=config.warmup_fraction,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and config.precision == "fp16"
        else None
    )
    horizon_ids = torch.arange(len(config.horizons_ms), device=device, dtype=torch.long)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "carla_jepa_encoder_best.pt"
    last_path = output / "carla_jepa_encoder_last.pt"
    history_path = output / "history.jsonl"
    metrics_path = output / "metrics.json"
    resume_path = output / "resume.pt"
    source_tree_sha256 = _source_tree_hash()
    git_commit = _git_commit()
    run_fingerprint = _run_fingerprint(
        config=config,
        manifest_path=manifest,
        split_path=split,
        source_tree_sha256=source_tree_sha256,
    )
    start_time = time.perf_counter()
    start_epoch = 1
    optimizer_step = 0
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    collapsed_epochs = 0
    history: list[dict[str, Any]] = []
    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"CARLA JEPA resume checkpoint is missing: {resume_path}.")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("run_fingerprint") != run_fingerprint:
            raise ValueError("CARLA JEPA resume fingerprint differs from the current run.")
        encoder.load_state_dict(state["encoder_state_dict"])
        target_encoder.load_state_dict(state["target_encoder_state_dict"])
        predictor.load_state_dict(state["predictor_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        optimizer_step = int(state["optimizer_step"])
        best_loss = float(state["best_loss"])
        best_epoch = int(state["best_epoch"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        collapsed_epochs = int(state["collapsed_epochs"])
        history = list(state["history"])
        _restore_rng_state(state["rng_state"], train_loader)
        history_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
            encoding="utf-8",
        )
    try:
        with history_path.open("a" if resume else "w", encoding="utf-8") as history_file:
            for epoch in range(start_epoch, config.epochs + 1):
                train_metrics, optimizer_step = _run_epoch(
                    encoder,
                    target_encoder,
                    predictor,
                    train_loader,
                    config=config,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    horizon_ids=horizon_ids,
                    optimizer_step=optimizer_step,
                    total_optimizer_steps=total_optimizer_steps,
                )
                with torch.no_grad():
                    validation_metrics, _ = _run_epoch(
                        encoder,
                        target_encoder,
                        predictor,
                        validation_loader,
                        config=config,
                        device=device,
                        optimizer=None,
                        scheduler=None,
                        scaler=None,
                        horizon_ids=horizon_ids,
                        optimizer_step=optimizer_step,
                        total_optimizer_steps=total_optimizer_steps,
                    )
                row = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                history.append(row)
                history_file.write(json.dumps(row, sort_keys=True) + "\n")
                history_file.flush()
                collapse_fraction = validation_metrics[
                    "context_collapsed_dimension_fraction"
                ]
                collapsed_epochs = (
                    collapsed_epochs + 1
                    if collapse_fraction > config.collapse_dimension_fraction
                    else 0
                )
                if collapsed_epochs >= config.collapse_patience:
                    raise RuntimeError(
                        "CARLA JEPA embedding collapse: "
                        f"{collapse_fraction:.1%} dimensions below 1e-3."
                    )
                score = validation_metrics["loss"]
                if score < best_loss:
                    best_loss = score
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    _atomic_torch_save(
                        _checkpoint(
                            encoder=encoder,
                            target_encoder=target_encoder,
                            predictor=predictor,
                            epoch=epoch,
                            role="best",
                            config=config,
                            manifest_path=manifest,
                            split_path=split,
                            source_tree_sha256=source_tree_sha256,
                            git_commit=git_commit,
                        ),
                        best_path,
                    )
                else:
                    epochs_without_improvement += 1
                _atomic_torch_save(
                    {
                        "run_fingerprint": run_fingerprint,
                        "epoch": epoch,
                        "optimizer_step": optimizer_step,
                        "encoder_state_dict": encoder.state_dict(),
                        "target_encoder_state_dict": target_encoder.state_dict(),
                        "predictor_state_dict": predictor.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                        "best_loss": best_loss,
                        "best_epoch": best_epoch,
                        "epochs_without_improvement": epochs_without_improvement,
                        "collapsed_epochs": collapsed_epochs,
                        "history": history,
                        "rng_state": _rng_state(train_loader),
                    },
                    resume_path,
                )
                if (
                    config.early_stopping_patience
                    and epoch >= config.early_stopping_min_epochs
                    and epochs_without_improvement >= config.early_stopping_patience
                ):
                    break
    finally:
        _shutdown_loader(train_loader)
        _shutdown_loader(validation_loader)

    epochs_completed = len(history)
    _atomic_torch_save(
        _checkpoint(
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            epoch=epochs_completed,
            role="last",
            config=config,
            manifest_path=manifest,
            split_path=split,
            source_tree_sha256=source_tree_sha256,
            git_commit=git_commit,
        ),
        last_path,
    )
    resume_path.unlink(missing_ok=True)
    summary: dict[str, Any] = {
        "artifact_type": "carla_dvs_looming_jepa_pretraining_v1",
        "pretraining_dataset_id": CARLA_LOOMING_DATASET_ID,
        "manifest_sha256": _hash_file(manifest),
        "manifest_artifact_sha256": _artifact_hash(manifest),
        "split_manifest_sha256": _hash_file(split),
        "split_artifact_sha256": _artifact_hash(split),
        "train_sequence_count": len(train_sequences),
        "validation_sequence_count": len(validation_sequences),
        "train_pair_count": len(train_dataset),
        "validation_pair_count": len(validation_dataset),
        "selected_train_pair_count": len(train_loader.dataset),
        "selected_validation_pair_count": len(validation_loader.dataset),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_checkpoint": best_path.as_posix(),
        "best_checkpoint_sha256": _hash_file(best_path),
        "last_checkpoint": last_path.as_posix(),
        "last_checkpoint_sha256": _hash_file(last_path),
        "elapsed_seconds": time.perf_counter() - start_time,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "trainer_config": asdict(config),
        "git_commit": git_commit,
        "source_tree_sha256": source_tree_sha256,
        "run_fingerprint": run_fingerprint,
        "history": history,
        "leakage_audit": {
            "uses_ttc_labels": False,
            "uses_collision_labels": False,
            "uses_velocity_feature": False,
            "uses_object_diameter_feature": False,
            "uses_future_events_as_ssl_targets": True,
            "train_validation_sequences_disjoint": True,
            "benchmark10_opened": False,
        },
    }
    write_structured(metrics_path, summary)
    return summary


def inspect_carla_jepa_pairs(
    *,
    root: str | Path,
    manifest_path: str | Path,
    split_path: str | Path,
    config: CarlaJEPATrainerConfig,
) -> dict[str, Any]:
    """Count on-demand pairs for every role without reading or voxelizing events."""

    assert_no_sealed_benchmark_paths([root, manifest_path, split_path])
    manifest = Path(manifest_path)
    split = Path(split_path)
    roles: dict[str, dict[str, int]] = {}
    for role in ("train", "validation", "test"):
        sequences = _sequences_for_role(manifest, split, role)
        dataset = CarlaJEPAVoxelDataset(root, sequences, config)
        roles[role] = {
            "sequence_count": len(sequences),
            "pair_count": len(dataset),
        }
    return {
        "artifact_type": "carla_dvs_looming_jepa_pair_inspection_v1",
        "manifest_sha256": _hash_file(manifest),
        "manifest_artifact_sha256": _artifact_hash(manifest),
        "split_manifest_sha256": _hash_file(split),
        "split_artifact_sha256": _artifact_hash(split),
        "trainer_config": asdict(config),
        "roles": roles,
        "benchmark10_opened": False,
    }


def evaluate_carla_jepa(
    *,
    root: str | Path,
    manifest_path: str | Path,
    split_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    role: str = "test",
    device_name: str = "auto",
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate the validation-selected CARLA JEPA model on one held-out role."""

    if role not in {"train", "validation", "test"}:
        raise ValueError("CARLA JEPA evaluation role must be train, validation or test.")
    assert_no_sealed_benchmark_paths(
        [root, manifest_path, split_path, checkpoint_path, output_path]
    )
    manifest = Path(manifest_path)
    split = Path(split_path)
    checkpoint_file = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_role") != "best":
        raise ValueError("CARLA holdout evaluation requires the validation-selected best.pt.")
    checkpoint_split_artifact = checkpoint.get("split_artifact_sha256")
    if checkpoint_split_artifact is not None:
        if checkpoint_split_artifact != _artifact_hash(split):
            raise ValueError("CARLA evaluation split differs from checkpoint provenance.")
    elif checkpoint.get("split_manifest_sha256") not in {
        _hash_file(split),
        read_structured(split).get("legacy_file_sha256"),
    }:
        raise ValueError("CARLA evaluation legacy split hash differs from the checkpoint.")
    checkpoint_manifest_artifact = checkpoint.get("manifest_artifact_sha256")
    if checkpoint_manifest_artifact is not None:
        if checkpoint_manifest_artifact != _artifact_hash(manifest):
            raise ValueError("CARLA evaluation manifest differs from checkpoint provenance.")
    elif checkpoint.get("manifest_sha256") != _hash_file(manifest):
        raise ValueError("CARLA evaluation legacy manifest hash differs from the checkpoint.")
    trainer_payload = checkpoint.get("trainer_config")
    if not isinstance(trainer_payload, dict):
        raise ValueError("CARLA JEPA checkpoint has no trainer_config.")
    trainer_payload = dict(trainer_payload)
    trainer_payload["horizons_ms"] = tuple(trainer_payload["horizons_ms"])
    config = CarlaJEPATrainerConfig(**trainer_payload)
    config = replace(
        config,
        batch_size=batch_size or config.batch_size,
        num_workers=config.num_workers if num_workers is None else num_workers,
        max_train_samples=None,
        max_validation_samples=None,
    )
    _set_seed(config.seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    sequences = _sequences_for_role(manifest, split, role)
    dataset = CarlaJEPAVoxelDataset(root, sequences, config)
    loader = _loader(
        dataset,
        maximum=max_samples,
        config=config,
        device=device,
        train=False,
    )
    encoder = build_encoder("event-tubelet-transformer", in_channels=21).to(device)
    target_encoder = build_encoder("event-tubelet-transformer", in_channels=21).to(device)
    predictor = DenseTemporalJEPAPredictor(
        dim=int(encoder.output_dim),
        horizon_count=len(config.horizons_ms) + int(config.context_token_weight > 0.0),
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    target_encoder.load_state_dict(checkpoint["target_encoder_state_dict"])
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    horizon_ids = torch.arange(len(config.horizons_ms), device=device, dtype=torch.long)
    start_time = time.perf_counter()
    try:
        with torch.no_grad():
            metrics, _ = _run_epoch(
                encoder,
                target_encoder,
                predictor,
                loader,
                config=config,
                device=device,
                optimizer=None,
                scheduler=None,
                scaler=None,
                horizon_ids=horizon_ids,
                optimizer_step=0,
                total_optimizer_steps=1,
            )
    finally:
        _shutdown_loader(loader)
    summary: dict[str, Any] = {
        "artifact_type": "carla_dvs_looming_jepa_holdout_evaluation_v1",
        "role": role,
        "sequence_count": len(sequences),
        "pair_count": len(dataset),
        "evaluated_pair_count": len(loader.dataset),
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - start_time,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "checkpoint": checkpoint_file.as_posix(),
        "checkpoint_sha256": _hash_file(checkpoint_file),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_selected_by": checkpoint.get("checkpoint_selected_by"),
        "manifest_sha256": _hash_file(manifest),
        "manifest_artifact_sha256": _artifact_hash(manifest),
        "split_manifest_sha256": _hash_file(split),
        "split_artifact_sha256": _artifact_hash(split),
        "evaluation_config": asdict(config),
        "uses_ttc_labels": False,
        "uses_collision_labels": False,
        "used_for_model_selection": role == "validation",
        "out_of_sample_scope": (
            "synthetic_carla_sequence_holdout" if role == "test" else role
        ),
        "benchmark10_opened": False,
    }
    write_structured(output_path, summary)
    return summary


__all__ = [
    "EVTTC_BASE_EVENT_CHANNELS",
    "EVTTC_BASE_INPUT_CHANNELS",
    "CarlaJEPATrainerConfig",
    "CarlaJEPAVoxelDataset",
    "evaluate_carla_jepa",
    "inspect_carla_jepa_pairs",
    "pretrain_carla_jepa",
]
