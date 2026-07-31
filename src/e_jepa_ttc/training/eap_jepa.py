"""On-demand eAP pretraining for EvTTC-compatible EventTubelet encoders."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash
from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.eap import (
    EAP_IMAGE_SIZE,
    EAPEventReader,
    EAPObjectState,
    load_eap_media_table,
    load_eap_sequence_labels,
    reconstruct_eap_object_states,
)
from e_jepa_ttc.data.eap_geometry_v2 import (
    EAP_GEOMETRY_V1_NAMES,
    EAP_GEOMETRY_V2_DIM,
    EAP_GEOMETRY_V2_NAMES,
    geometry_v2_targets,
)
from e_jepa_ttc.data.eap_representation import base_compatible_voxel, downsample_full_frame
from e_jepa_ttc.models import build_encoder
from e_jepa_ttc.training.carla_jepa import (
    EVTTC_BASE_EVENT_CHANNELS,
    EVTTC_BASE_INPUT_CHANNELS,
    _artifact_hash,
    _atomic_torch_save,
    _autocast,
    _git_commit,
    _hash_file,
    _restore_rng_state,
    _rng_state,
    _scheduler,
    _set_seed,
    _source_tree_hash,
    _worker_seed,
)
from e_jepa_ttc.training.jepa import DenseTemporalJEPAPredictor, _jepa_loss, _update_ema
from e_jepa_ttc.utils.io import read_structured, write_structured

EAP_PRETRAINING_DATASET_ID = "EAP_PUBLIC_TRAIN40"
EAP_GEOMETRY_TARGET_DIM = 6


def _geometry_target_names(version: str) -> tuple[str, ...]:
    if version == "v1":
        return EAP_GEOMETRY_V1_NAMES
    if version == "v2":
        return EAP_GEOMETRY_V2_NAMES
    raise ValueError(f"Unsupported eAP geometry target version: {version!r}.")


@dataclass(frozen=True)
class EAPJEPATrainerConfig:
    """Bounded on-demand eAP pretraining configuration."""

    epochs: int = 30
    batch_size: int = 24
    gradient_accumulation: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    precision: str = "bf16"
    num_workers: int = 8
    prefetch_factor: int = 2
    event_window_ms: int = 100
    horizons_ms: tuple[int, ...] = (100, 250, 500)
    max_windows_per_sequence: int = 512
    width: int = 160
    height: int = 90
    bins: int = 5
    mask_ratio: float = 0.45
    mask_blocks: int = 4
    context_token_weight: float = 0.25
    variance_weight: float = 1.0
    minimum_std: float = 0.05
    geometry_loss_weight: float = 0.0
    geometry_target_version: str = "v1"
    geometry_sampling_strategy: str = "nearest"
    corridor_half_width: float = 0.18
    patch_objectness_weight: float = 0.25
    ttc_loss_weight: float = 0.25
    ema_start: float = 0.99
    ema_end: float = 0.9999
    early_stopping_patience: int = 6
    early_stopping_min_epochs: int = 8
    collapse_patience: int = 3
    collapse_dimension_fraction: float = 0.80
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    seed: int = 42
    expected_garlttc_train_rows: int = 88_744
    allow_garlttc_version_change: bool = False

    def __post_init__(self) -> None:
        integers = (
            self.epochs,
            self.batch_size,
            self.gradient_accumulation,
            self.num_workers + 1,
            self.prefetch_factor,
            self.event_window_ms,
            self.max_windows_per_sequence,
            self.width,
            self.height,
            self.bins,
            self.mask_blocks,
            self.collapse_patience,
        )
        if min(integers) <= 0:
            raise ValueError("eAP JEPA integer controls must be positive.")
        if self.bins * 2 != EVTTC_BASE_EVENT_CHANNELS:
            raise ValueError("EvTTC BASE compatibility requires exactly five event bins.")
        if not self.horizons_ms or any(
            horizon < self.event_window_ms for horizon in self.horizons_ms
        ):
            raise ValueError("eAP horizons must keep future windows disjoint.")
        if tuple(sorted(set(self.horizons_ms))) != self.horizons_ms:
            raise ValueError("eAP horizons must be sorted and unique.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16.")
        if self.geometry_loss_weight < 0.0 or self.patch_objectness_weight < 0.0:
            raise ValueError("eAP geometry weights must be non-negative.")
        _geometry_target_names(self.geometry_target_version)
        if self.geometry_sampling_strategy not in {"nearest", "balanced_tracks"}:
            raise ValueError("geometry_sampling_strategy must be nearest or balanced_tracks.")
        if not 0.0 < self.corridor_half_width <= 0.5:
            raise ValueError("corridor_half_width must lie in (0, 0.5].")
        if not 0.0 <= self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie in [0, 1).")


@dataclass(frozen=True)
class _EAPJEPASample:
    sequence_id: str
    timestamp_us: int
    geometry_state: EAPObjectState | None = None
    previous_geometry_state: EAPObjectState | None = None
    sampling_group: str = "unlabelled"


def _uniform(items: list[int], maximum: int) -> list[int]:
    if len(items) <= maximum:
        return items
    indices = np.linspace(0, len(items) - 1, maximum, dtype=np.int64)
    return [items[int(index)] for index in np.unique(indices)]


def _geometry_targets(
    sample: _EAPJEPASample,
    *,
    patch_height: int,
    patch_width: int,
    config: EAPJEPATrainerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_names = _geometry_target_names(config.geometry_target_version)
    state = sample.geometry_state
    if state is None:
        return (
            torch.zeros(len(target_names), dtype=torch.float32),
            torch.zeros(len(target_names), dtype=torch.bool),
            torch.zeros((patch_height, patch_width), dtype=torch.bool),
        )
    image_width, image_height = EAP_IMAGE_SIZE
    x0, y0, x1, y1 = state.bbox_xyxy
    if config.geometry_target_version == "v2":
        target = geometry_v2_targets(
            state,
            sample.previous_geometry_state,
            corridor_half_width=config.corridor_half_width,
        )
        values = target.values
        valid = target.valid
    else:
        center_x = (x0 + x1) * 0.5 / image_width
        center_y = (y0 + y1) * 0.5 / image_height
        width = (x1 - x0) / image_width
        height = (y1 - y0) / image_height
        closing = -state.depth_velocity_mps / 20.0
        previous = sample.previous_geometry_state
        if previous is None:
            expansion = float("nan")
        else:
            delta_s = (state.timestamp_us - previous.timestamp_us) * 1e-6
            expansion = (
                (
                    math.log(max(state.visible_height_px, 1e-3))
                    - math.log(max(previous.visible_height_px, 1e-3))
                )
                / max(delta_s, 1e-6)
                / 5.0
                if 0.0 < delta_s <= 0.25
                else float("nan")
            )
        values = np.asarray(
            [
                center_x,
                center_y,
                width,
                height,
                np.clip(closing, -1.0, 1.0),
                np.clip(expansion, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        valid = np.isfinite(values)
        values[~valid] = 0.0
    center_x = (x0 + x1) * 0.5 / image_width
    center_y = (y0 + y1) * 0.5 / image_height
    x_centers = (np.arange(patch_width, dtype=np.float32) + 0.5) / patch_width
    y_centers = (np.arange(patch_height, dtype=np.float32) + 0.5) / patch_height
    objectness = (
        (x_centers[None, :] >= x0 / image_width)
        & (x_centers[None, :] <= x1 / image_width)
        & (y_centers[:, None] >= y0 / image_height)
        & (y_centers[:, None] <= y1 / image_height)
    )
    if not np.any(objectness):
        x_index = int(np.clip(center_x * patch_width, 0, patch_width - 1))
        y_index = int(np.clip(center_y * patch_height, 0, patch_height - 1))
        objectness[y_index, x_index] = True
    return (
        torch.from_numpy(values),
        torch.from_numpy(valid),
        torch.from_numpy(objectness),
    )



class EAPOnDemandJEPADataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
):
    """Build full-frame eAP JEPA pairs lazily with per-worker HDF5 readers."""

    def __init__(
        self,
        root: str | Path,
        sequence_ids: list[str],
        config: EAPJEPATrainerConfig,
    ) -> None:
        self.root = Path(root)
        self.config = config
        media = load_eap_media_table(self.root, split="train")
        self.samples: list[_EAPJEPASample] = []
        for sequence_id in sorted(sequence_ids):
            sequence_media = media[media["sequence_id"].astype(str) == sequence_id].copy()
            frame_times = np.rint(
                (
                    sequence_media["rgb_exposure_start_timestamp_us"].to_numpy(dtype=np.int64)
                    + sequence_media["rgb_exposure_end_timestamp_us"].to_numpy(dtype=np.int64)
                )
                * 0.5
            ).astype(np.int64)
            frame_times = np.unique(frame_times)
            earliest = int(frame_times[0]) + config.event_window_ms * 1000
            latest = int(frame_times[-1]) - max(config.horizons_ms) * 1000
            eligible = [
                int(timestamp) for timestamp in frame_times if earliest <= int(timestamp) <= latest
            ]
            selected = _uniform(eligible, config.max_windows_per_sequence)
            if config.geometry_loss_weight <= 0.0:
                self.samples.extend(
                    _EAPJEPASample(sequence_id=sequence_id, timestamp_us=timestamp)
                    for timestamp in selected
                )
                continue
            labels = load_eap_sequence_labels(self.root, sequence_id, split="train")
            states = reconstruct_eap_object_states(sequence_media, labels)
            by_track: dict[str, list[EAPObjectState]] = {}
            for state in states:
                by_track.setdefault(state.track_id, []).append(state)
            previous_by_state: dict[tuple[str, int], EAPObjectState | None] = {}
            for track in by_track.values():
                track.sort(key=lambda value: value.timestamp_us)
                for index, state in enumerate(track):
                    previous_by_state[(state.track_id, state.timestamp_us)] = (
                        track[index - 1] if index else None
                    )
            eligible_set = set(eligible)
            if config.geometry_sampling_strategy == "nearest":
                candidates: dict[int, list[EAPObjectState]] = {}
                for state in states:
                    if state.timestamp_us in eligible_set:
                        candidates.setdefault(state.timestamp_us, []).append(state)
                geometry_samples = []
                for timestamp in selected:
                    timestamp_states = candidates.get(timestamp, [])
                    if not timestamp_states:
                        continue
                    state = min(timestamp_states, key=lambda value: value.nearest_depth_m)
                    previous = previous_by_state[(state.track_id, state.timestamp_us)]
                    geometry_samples.append(
                        _EAPJEPASample(
                            sequence_id=sequence_id,
                            timestamp_us=timestamp,
                            geometry_state=state,
                            previous_geometry_state=previous,
                        )
                    )
            else:
                grouped: dict[str, list[_EAPJEPASample]] = {}
                for state in states:
                    if state.timestamp_us not in eligible_set:
                        continue
                    previous = previous_by_state[(state.track_id, state.timestamp_us)]
                    target = geometry_v2_targets(
                        state,
                        previous,
                        corridor_half_width=config.corridor_half_width,
                    )
                    grouped.setdefault(target.sampling_group, []).append(
                        _EAPJEPASample(
                            sequence_id=sequence_id,
                            timestamp_us=state.timestamp_us,
                            geometry_state=state,
                            previous_geometry_state=previous,
                            sampling_group=target.sampling_group,
                        )
                    )
                geometry_samples = []
                ordered_groups = sorted(grouped)
                cursors = {group: 0 for group in ordered_groups}
                while ordered_groups and len(geometry_samples) < config.max_windows_per_sequence:
                    next_groups: list[str] = []
                    for group in ordered_groups:
                        index = cursors[group]
                        rows = grouped[group]
                        if index < len(rows):
                            geometry_samples.append(rows[index])
                            cursors[group] += 1
                        if cursors[group] < len(rows):
                            next_groups.append(group)
                        if len(geometry_samples) >= config.max_windows_per_sequence:
                            break
                    ordered_groups = next_groups
            self.samples.extend(geometry_samples)
        if not self.samples:
            raise ValueError("No eAP object windows satisfy the on-demand JEPA protocol.")
        self._readers: dict[str, EAPEventReader] = {}
        self._voxel_cache: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _reader(self, sequence_id: str) -> EAPEventReader:
        reader = self._readers.get(sequence_id)
        if reader is None:
            path = self.root / "data" / "train" / sequence_id / "events.h5"
            reader = EAPEventReader(path)
            reader.open()
            self._readers[sequence_id] = reader
        return reader

    def _voxel(self, sequence_id: str, end_us: int) -> torch.Tensor:
        key = (sequence_id, end_us)
        cached = self._voxel_cache.pop(key, None)
        if cached is not None:
            self._voxel_cache[key] = cached
            return cached.clone()
        start_us = end_us - self.config.event_window_ms * 1000
        events = self._reader(sequence_id).read_window(start_us, end_us)
        voxel = base_compatible_voxel(
            downsample_full_frame(
                events,
                sequence_id=sequence_id,
                start_us=start_us,
                end_us=end_us,
                width=self.config.width,
                height=self.config.height,
            ),
            bins=self.config.bins,
        )
        self._voxel_cache[key] = voxel
        while len(self._voxel_cache) > 16:
            self._voxel_cache.popitem(last=False)
        return voxel.clone()

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        window = self.samples[index]
        context = self._voxel(window.sequence_id, window.timestamp_us)
        future: list[torch.Tensor] = []
        for horizon in self.config.horizons_ms:
            future.append(
                self._voxel(
                    window.sequence_id,
                    window.timestamp_us + horizon * 1000,
                )
            )
        geometry, geometry_valid, objectness = _geometry_targets(
            window,
            patch_height=self.config.height // 16,
            patch_width=self.config.width // 16,
            config=self.config,
        )
        return (
            context,
            torch.stack(future),
            torch.ones(len(self.config.horizons_ms), dtype=torch.bool),
            geometry,
            geometry_valid,
            objectness,
        )

    def close(self) -> None:
        readers = getattr(self, "_readers", {})
        for reader in readers.values():
            reader.close()
        readers.clear()
        cache = getattr(self, "_voxel_cache", None)
        if cache is not None:
            cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_readers"] = {}
        state["_voxel_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        self.close()


class _GeometryHeads(nn.Module):
    def __init__(self, dim: int, target_dim: int) -> None:
        super().__init__()
        self.geometry = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, target_dim),
        )
        self.objectness = nn.Linear(dim, 1)


def _bounded_indices(length: int, maximum: int | None) -> list[int]:
    if maximum is None or maximum >= length:
        return list(range(length))
    if maximum <= 0:
        raise ValueError("Sample limits must be positive.")
    return np.linspace(0, length - 1, maximum, dtype=np.int64).tolist()


@dataclass
class EAPJEPAModels:
    online_encoder: nn.Module
    target_encoder: nn.Module
    predictor: nn.Module


@dataclass(frozen=True)
class EAPJEPALossOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


def build_eap_jepa_models(
    *,
    config: EAPJEPATrainerConfig,
    device: torch.device,
) -> EAPJEPAModels:

    encoder = build_encoder("event-tubelet-transformer", in_channels=21).to(device)
    target_encoder = build_encoder("event-tubelet-transformer", in_channels=21).to(device)
    target_encoder.load_state_dict(encoder.state_dict())
    target_encoder.requires_grad_(False)
    target_encoder.eval()
    predictor = DenseTemporalJEPAPredictor(
        dim=int(encoder.output_dim),
        horizon_count=len(config.horizons_ms) + int(config.context_token_weight > 0.0),
    ).to(device)
    return EAPJEPAModels(
        online_encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
    )


def compute_eap_jepa_objective(
    *,
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    context: torch.Tensor,
    futures: torch.Tensor,
    future_valid: torch.Tensor,
    config: EAPJEPATrainerConfig,
) -> EAPJEPALossOutput:
    horizon_ids = torch.arange(
        len(config.horizons_ms),
        device=context.device,
        dtype=torch.long,
    )

    loss, metrics = _jepa_loss(
        online_encoder,
        target_encoder,
        predictor,
        context,
        future_x=futures,
        future_mask=future_valid,
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
    return EAPJEPALossOutput(loss=loss, metrics=metrics)


def update_eap_jepa_ema(
    *,
    target_encoder: nn.Module,
    online_encoder: nn.Module,
    optimizer_step: int,
    total_optimizer_steps: int,
    config: EAPJEPATrainerConfig,
) -> tuple[float, float]:
    """Update target encoder using the shared cosine EMA schedule."""

    if total_optimizer_steps <= 0:
        raise ValueError(f"total_optimizer_steps must be positive, got {total_optimizer_steps}")

    progress = min(
        max(optimizer_step, 0) / float(total_optimizer_steps),
        1.0,
    )

    momentum = (
        config.ema_end
        - (config.ema_end - config.ema_start) * (math.cos(math.pi * progress) + 1.0) / 2.0
    )

    divergence = _update_ema(
        target_encoder,
        online_encoder,
        momentum=momentum,
    )

    return float(divergence), float(momentum)


def _loader(
    dataset: EAPOnDemandJEPADataset,
    *,
    maximum: int | None,
    config: EAPJEPATrainerConfig,
    device: torch.device,
    train: bool,
) -> DataLoader[Any]:
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
        kwargs.update(persistent_workers=True, prefetch_factor=config.prefetch_factor)
    return DataLoader(Subset(dataset, _bounded_indices(len(dataset), maximum)), **kwargs)


def _optimizer(
    parameters: list[nn.Parameter],
    config: EAPJEPATrainerConfig,
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


def _shutdown_loader(loader: DataLoader[Any]) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if hasattr(loader, "_iterator"):
        loader._iterator = None


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("eAP JEPA loader produced no batches.")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _run_epoch(
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    geometry_heads: _GeometryHeads | None,
    loader: DataLoader[Any],
    *,
    config: EAPJEPATrainerConfig,
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
    if geometry_heads is not None:
        geometry_heads.train(train)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    rows: list[dict[str, float]] = []
    batch_count = len(loader)
    for batch_index, batch in enumerate(loader):
        context, future, valid, geometry, geometry_valid, objectness = batch
        context = context.to(device=device, non_blocking=True)
        future = future.to(device=device, non_blocking=True)
        valid = valid.to(device=device, non_blocking=True)
        geometry = geometry.to(device=device, non_blocking=True)
        geometry_valid = geometry_valid.to(device=device, non_blocking=True)
        objectness = objectness.to(device=device, non_blocking=True)
        group_start = (batch_index // config.gradient_accumulation) * config.gradient_accumulation
        group_size = min(config.gradient_accumulation, batch_count - group_start)
        with _autocast(device, config.precision):
            jepa_output = compute_eap_jepa_objective(
                online_encoder=encoder,
                target_encoder=target_encoder,
                predictor=predictor,
                context=context,
                futures=future,
                future_valid=valid,
                config=config,
            )
            loss = jepa_output.loss
            metrics = jepa_output.metrics
            if geometry_heads is not None:
                tokens = encoder.forward_tokens(context)
                predicted_geometry = geometry_heads.geometry(tokens.mean(dim=1))
                per_geometry = torch.nn.functional.smooth_l1_loss(
                    predicted_geometry,
                    geometry,
                    beta=0.1,
                    reduction="none",
                )
                geometry_mask = geometry_valid.to(dtype=per_geometry.dtype)
                geometry_loss = (
                    per_geometry * geometry_mask
                ).sum() / geometry_mask.sum().clamp_min(1)
                spatial_count = objectness.shape[-2] * objectness.shape[-1]
                temporal_count = tokens.shape[1] // spatial_count
                token_target = objectness.flatten(1).repeat(1, temporal_count)
                logits = geometry_heads.objectness(tokens).squeeze(-1)
                positives = token_target.sum().clamp_min(1.0)
                negatives = token_target.numel() - positives
                positive_weight = (negatives / positives).clamp(1.0, 20.0)
                objectness_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    token_target.to(dtype=logits.dtype),
                    pos_weight=positive_weight,
                )
                jepa_loss = loss
                loss = loss + config.geometry_loss_weight * (
                    geometry_loss + config.patch_objectness_weight * objectness_loss
                )
                predicted_mask = logits.detach() >= 0
                true_mask = token_target.bool()
                intersection = (predicted_mask & true_mask).sum().float()
                union = (predicted_mask | true_mask).sum().float().clamp_min(1.0)
                metrics.update(
                    {
                        "jepa_loss": float(jepa_loss.detach().cpu()),
                        "geometry_loss": float(geometry_loss.detach().cpu()),
                        "patch_objectness_loss": float(objectness_loss.detach().cpu()),
                        "patch_objectness_iou": float((intersection / union).cpu()),
                        "geometry_loss_weight": config.geometry_loss_weight,
                    }
                )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"eAP JEPA produced non-finite loss: {metrics}.")
        if train:
            scaled_loss = loss / group_size
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            end_group = (batch_index + 1) % config.gradient_accumulation == 0
            if end_group or batch_index + 1 == batch_count:
                parameters = [*encoder.parameters(), *predictor.parameters()]
                if geometry_heads is not None:
                    parameters.extend(geometry_heads.parameters())
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
                divergence, ema_momentum = update_eap_jepa_ema(
                    target_encoder=target_encoder,
                    online_encoder=encoder,
                    optimizer_step=optimizer_step,
                    total_optimizer_steps=total_optimizer_steps,
                    config=config,
                )

                metrics["target_encoder_divergence_l2"] = divergence
                metrics["ema_momentum"] = ema_momentum
        rows.append({"loss": float(loss.detach().cpu()), **metrics})
    return _aggregate(rows), optimizer_step


def _sequences(split_path: Path, role: str) -> list[str]:
    payload = read_structured(split_path)
    if not verify_artifact_hash(payload):
        raise ValueError("eAP pilot split signature is invalid.")
    assignments = payload.get("assignments")
    if not isinstance(assignments, dict) or not isinstance(assignments.get(role), list):
        raise ValueError(f"eAP pilot split has no role {role!r}.")
    return sorted(str(value) for value in assignments[role])


def _validate_protocol(inventory_path: Path, split_path: Path) -> None:
    inventory = read_structured(inventory_path)
    split = read_structured(split_path)
    if not verify_artifact_hash(inventory):
        raise ValueError("eAP inventory signature is invalid.")
    if not verify_artifact_hash(split):
        raise ValueError("eAP pilot split signature is invalid.")
    if split.get("inventory_artifact_sha256") != inventory.get("artifact_sha256"):
        raise ValueError("eAP split was selected from a different inventory artifact.")
    assignments = split.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("eAP pilot split assignments are missing.")
    train = {str(value) for value in assignments.get("train", [])}
    validation = {str(value) for value in assignments.get("validation", [])}
    if not train or not validation or train & validation:
        raise ValueError("eAP train and validation sequence assignments must be disjoint.")
    selected = {str(value) for value in split.get("selected", [])}
    if selected != train | validation:
        raise ValueError("eAP selected sequences differ from train/validation assignments.")
    inventory_ids = {
        str(row["sequence_id"])
        for row in inventory.get("rows", [])
        if isinstance(row, dict) and "sequence_id" in row
    }
    if not selected <= inventory_ids:
        raise ValueError("eAP pilot split contains sequences absent from its inventory.")


def _run_fingerprint(
    config: EAPJEPATrainerConfig,
    inventory_path: Path,
    split_path: Path,
) -> str:
    payload = {
        "config": asdict(config),
        "inventory_artifact_sha256": _artifact_hash(inventory_path),
        "split_artifact_sha256": _artifact_hash(split_path),
        "source_tree_sha256": _source_tree_hash(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _checkpoint(
    *,
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    geometry_heads: _GeometryHeads | None,
    epoch: int,
    role: str,
    config: EAPJEPATrainerConfig,
    inventory_path: Path,
    split_path: Path,
) -> dict[str, Any]:
    regime = "eap_geo" if geometry_heads is not None else "eap_ssl"
    return {
        "model": "event_tubelet_transformer_eap_jepa",
        "model_name": "event-tubelet-transformer",
        "objective": (
            "dense_temporal_jepa_plus_projected_3d_geometry"
            if geometry_heads is not None
            else "dense_temporal_jepa"
        ),
        "encoder_state_dict": encoder.state_dict(),
        "target_encoder_state_dict": target_encoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "geometry_heads_state_dict": (
            geometry_heads.state_dict() if geometry_heads is not None else None
        ),
        "epoch": epoch,
        "checkpoint_role": role,
        "checkpoint_selected_by": "validation_loss" if role == "best" else "final_epoch",
        "seed": config.seed,
        "in_channels": EVTTC_BASE_INPUT_CHANNELS,
        "bins": config.bins,
        "pretraining_dataset_id": EAP_PRETRAINING_DATASET_ID,
        "external_pretraining": True,
        "pretraining_regime": regime,
        "inventory_artifact_sha256": _artifact_hash(inventory_path),
        "split_artifact_sha256": _artifact_hash(split_path),
        "pretrain_splits": ["train"],
        "validation_splits": ["validation"],
        "uses_ttc_labels": False,
        "uses_collision_labels": False,
        "uses_object_bboxes": geometry_heads is not None,
        "uses_depth_track_derivatives": geometry_heads is not None,
        "uses_labels_for_window_sampling": (
            geometry_heads is not None
            and config.geometry_sampling_strategy == "balanced_tracks"
        ),
        "geometry_target_version": config.geometry_target_version,
        "geometry_target_names": list(_geometry_target_names(config.geometry_target_version)),
        "geometry_sampling_strategy": config.geometry_sampling_strategy,
        "uses_rgb": False,
        "uses_evttc_pretraining_events": False,
        "benchmark10_opened": False,
        "trainer_config": asdict(config),
        "git_commit": _git_commit(),
        "source_tree_sha256": _source_tree_hash(),
    }


def inspect_eap_jepa_windows(
    *,
    root: str | Path,
    inventory_path: str | Path,
    split_path: str | Path,
    config: EAPJEPATrainerConfig,
) -> dict[str, Any]:
    """Inspect exact eAP sample counts without reading RGB or event payload arrays."""

    assert_no_sealed_benchmark_paths((root, inventory_path, split_path))
    inventory = Path(inventory_path)
    split = Path(split_path)
    _validate_protocol(inventory, split)
    train_sequences = _sequences(split, "train")
    validation_sequences = _sequences(split, "validation")
    train_dataset = EAPOnDemandJEPADataset(root, train_sequences, config)
    validation_dataset = EAPOnDemandJEPADataset(root, validation_sequences, config)
    try:
        return {
            "artifact_type": "eap_on_demand_jepa_inspection_v1",
            "inventory_artifact_sha256": _artifact_hash(inventory),
            "split_artifact_sha256": _artifact_hash(split),
            "train_sequences": train_sequences,
            "validation_sequences": validation_sequences,
            "train_window_count": len(train_dataset),
            "validation_window_count": len(validation_dataset),
            "selected_train_samples": len(
                _bounded_indices(len(train_dataset), config.max_train_samples)
            ),
            "selected_validation_samples": len(
                _bounded_indices(
                    len(validation_dataset),
                    config.max_validation_samples,
                )
            ),
            "event_payloads_opened": False,
            "rgb_opened": False,
            "materialized_event_cache": False,
            "benchmark10_opened": False,
        }
    finally:
        train_dataset.close()
        validation_dataset.close()


def pretrain_eap_jepa(
    *,
    root: str | Path,
    inventory_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    config: EAPJEPATrainerConfig,
    device_name: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Run sequence-disjoint eAP SSL or Geo without materializing an event cache."""

    assert_no_sealed_benchmark_paths((root, inventory_path, split_path, output_dir))
    _set_seed(config.seed)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    inventory = Path(inventory_path)
    split = Path(split_path)
    _validate_protocol(inventory, split)
    train_sequences = _sequences(split, "train")
    validation_sequences = _sequences(split, "validation")
    train_dataset = EAPOnDemandJEPADataset(root, train_sequences, config)
    validation_dataset = EAPOnDemandJEPADataset(root, validation_sequences, config)
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
    models = build_eap_jepa_models(config=config, device=device)
    encoder = models.online_encoder
    target_encoder = models.target_encoder
    predictor = models.predictor
    geometry_heads = (
        _GeometryHeads(
            int(encoder.output_dim),
            len(_geometry_target_names(config.geometry_target_version)),
        ).to(device)
        if config.geometry_loss_weight > 0.0
        else None
    )
    parameters = [*encoder.parameters(), *predictor.parameters()]
    if geometry_heads is not None:
        parameters.extend(geometry_heads.parameters())
    optimizer = _optimizer(parameters, config, device)
    steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation)
    total_steps = steps_per_epoch * config.epochs
    scheduler = _scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_fraction=config.warmup_fraction,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and config.precision == "fp16"
        else None
    )
    horizons = torch.arange(len(config.horizons_ms), device=device, dtype=torch.long)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "eap_jepa_encoder_best.pt"
    last_path = output / "eap_jepa_encoder_last.pt"
    resume_path = output / "resume.pt"
    history_path = output / "history.jsonl"
    metrics_path = output / "metrics.json"
    fingerprint = _run_fingerprint(config, inventory, split)
    start_epoch = 1
    optimizer_step = 0
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    collapsed_epochs = 0
    history: list[dict[str, Any]] = []
    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"eAP JEPA resume checkpoint is missing: {resume_path}.")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("run_fingerprint") != fingerprint:
            raise ValueError("eAP JEPA resume fingerprint differs from the current run.")
        encoder.load_state_dict(state["encoder_state_dict"])
        target_encoder.load_state_dict(state["target_encoder_state_dict"])
        predictor.load_state_dict(state["predictor_state_dict"])
        if geometry_heads is not None:
            geometry_heads.load_state_dict(state["geometry_heads_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        optimizer_step = int(state["optimizer_step"])
        best_loss = float(state["best_loss"])
        best_epoch = int(state["best_epoch"])
        no_improvement = int(state["no_improvement"])
        collapsed_epochs = int(state["collapsed_epochs"])
        history = list(state["history"])
        _restore_rng_state(state["rng_state"], train_loader)
    started = time.perf_counter()
    try:
        with history_path.open("a" if resume else "w", encoding="utf-8") as stream:
            for epoch in range(start_epoch, config.epochs + 1):
                train_metrics, optimizer_step = _run_epoch(
                    encoder,
                    target_encoder,
                    predictor,
                    geometry_heads,
                    train_loader,
                    config=config,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    horizon_ids=horizons,
                    optimizer_step=optimizer_step,
                    total_optimizer_steps=total_steps,
                )
                with torch.no_grad():
                    validation_metrics, _ = _run_epoch(
                        encoder,
                        target_encoder,
                        predictor,
                        geometry_heads,
                        validation_loader,
                        config=config,
                        device=device,
                        optimizer=None,
                        scheduler=None,
                        scaler=None,
                        horizon_ids=horizons,
                        optimizer_step=optimizer_step,
                        total_optimizer_steps=total_steps,
                    )
                row = {
                    "epoch": epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
                history.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                collapse = validation_metrics["context_collapsed_dimension_fraction"]
                collapsed_epochs = (
                    collapsed_epochs + 1 if collapse > config.collapse_dimension_fraction else 0
                )
                if collapsed_epochs >= config.collapse_patience:
                    raise RuntimeError("eAP JEPA embedding collapse detected.")
                score = validation_metrics["loss"]
                if score < best_loss:
                    best_loss = score
                    best_epoch = epoch
                    no_improvement = 0
                    _atomic_torch_save(
                        _checkpoint(
                            encoder=encoder,
                            target_encoder=target_encoder,
                            predictor=predictor,
                            geometry_heads=geometry_heads,
                            epoch=epoch,
                            role="best",
                            config=config,
                            inventory_path=inventory,
                            split_path=split,
                        ),
                        best_path,
                    )
                else:
                    no_improvement += 1
                _atomic_torch_save(
                    {
                        "run_fingerprint": fingerprint,
                        "epoch": epoch,
                        "optimizer_step": optimizer_step,
                        "encoder_state_dict": encoder.state_dict(),
                        "target_encoder_state_dict": target_encoder.state_dict(),
                        "predictor_state_dict": predictor.state_dict(),
                        "geometry_heads_state_dict": (
                            geometry_heads.state_dict() if geometry_heads is not None else None
                        ),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_loss": best_loss,
                        "best_epoch": best_epoch,
                        "no_improvement": no_improvement,
                        "collapsed_epochs": collapsed_epochs,
                        "history": history,
                        "rng_state": _rng_state(train_loader),
                    },
                    resume_path,
                )
                if (
                    config.early_stopping_patience
                    and epoch >= config.early_stopping_min_epochs
                    and no_improvement >= config.early_stopping_patience
                ):
                    break
    finally:
        _shutdown_loader(train_loader)
        _shutdown_loader(validation_loader)
        train_dataset.close()
        validation_dataset.close()
    _atomic_torch_save(
        _checkpoint(
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            geometry_heads=geometry_heads,
            epoch=len(history),
            role="last",
            config=config,
            inventory_path=inventory,
            split_path=split,
        ),
        last_path,
    )
    resume_path.unlink(missing_ok=True)
    summary = {
        "artifact_type": (
            "eap_geo_on_demand_pretraining_v1"
            if geometry_heads is not None
            else "eap_ssl_on_demand_pretraining_v1"
        ),
        "pretraining_dataset_id": EAP_PRETRAINING_DATASET_ID,
        "pretraining_regime": "eap_geo" if geometry_heads is not None else "eap_ssl",
        "inventory_artifact_sha256": _artifact_hash(inventory),
        "split_artifact_sha256": _artifact_hash(split),
        "train_sequences": train_sequences,
        "validation_sequences": validation_sequences,
        "train_window_count": len(train_dataset),
        "validation_window_count": len(validation_dataset),
        "selected_train_samples": len(train_loader.dataset),
        "selected_validation_samples": len(validation_loader.dataset),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_checkpoint": best_path.as_posix(),
        "best_checkpoint_sha256": _hash_file(best_path),
        "last_checkpoint": last_path.as_posix(),
        "last_checkpoint_sha256": _hash_file(last_path),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "trainer_config": asdict(config),
        "run_fingerprint": fingerprint,
        "history": history,
        "provenance": {
            "uses_ttc_labels": False,
            "uses_object_bboxes": geometry_heads is not None,
            "uses_depth_track_derivatives": geometry_heads is not None,
            "uses_labels_for_window_sampling": (
                config.geometry_loss_weight > 0.0
                and config.geometry_sampling_strategy == "balanced_tracks"
            ),
            "geometry_target_version": config.geometry_target_version,
            "geometry_target_names": list(_geometry_target_names(config.geometry_target_version)),
            "geometry_sampling_strategy": config.geometry_sampling_strategy,
            "uses_rgb": False,
            "uses_evttc_pretraining_events": False,
            "materialized_event_cache": False,
            "benchmark10_opened": False,
        },
    }
    write_structured(metrics_path, summary)
    return summary


__all__ = [
    "EAP_GEOMETRY_TARGET_DIM",
    "EAP_PRETRAINING_DATASET_ID",
    "EAPJEPATrainerConfig",
    "EAPOnDemandJEPADataset",
    "inspect_eap_jepa_windows",
    "pretrain_eap_jepa",
]
