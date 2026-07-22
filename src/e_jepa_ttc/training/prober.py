"""Frozen latent TTC probers inspired by SkyJEPA."""

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

from e_jepa_ttc.baselines.roi_events import (
    ROI_EVENT_FEATURE_NAMES,
    ROIEventRow,
    _sequence_rows,
)
from e_jepa_ttc.data.annotations import load_measurements_from_manifest
from e_jepa_ttc.data.evttc import NAVIGATION_FEATURE_NAMES, read_manifest
from e_jepa_ttc.data.ml_cache import validate_voxel_cache
from e_jepa_ttc.data.split import read_splits
from e_jepa_ttc.evaluation.metrics import regression_metrics
from e_jepa_ttc.models import build_encoder, build_regressor
from e_jepa_ttc.training.checkpoints import checkpoint_provenance
from e_jepa_ttc.training.jepa import (
    EVENT_MOTION_FEATURE_NAMES,
    DenseTemporalJEPAPredictor,
    DenseTemporalTransformerJEPAPredictor,
    TemporalJEPAPredictor,
    _cache_bool,
    _cache_string_tuple,
    _context_action_features,
    _normalize_action_features,
)
from e_jepa_ttc.utils.io import ensure_parent, write_structured


class LatentProberDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset backed by frozen latent/physics features and log-TTC targets."""

    def __init__(
        self,
        features: np.ndarray,
        prior_log_ttc: np.ndarray,
        target_log_ttc: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.features = features.astype(np.float32, copy=False)
        self.prior_log_ttc = prior_log_ttc.astype(np.float32, copy=False)
        self.target_log_ttc = target_log_ttc.astype(np.float32, copy=False)
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_idx = int(self.indices[idx])
        features = torch.from_numpy(self.features[real_idx])
        prior = torch.tensor(self.prior_log_ttc[real_idx], dtype=torch.float32)
        target = torch.tensor(self.target_log_ttc[real_idx], dtype=torch.float32)
        return features, prior, target


class LatentResidualTTCProber(nn.Module):
    """Lightweight prober that predicts log-TTC residuals from frozen latents."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.05,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, prior_log_ttc: torch.Tensor) -> torch.Tensor:
        pred = self.net(features).squeeze(-1)
        if self.residual:
            return prior_log_ttc + pred
        return pred


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


def _standardize_train(
    train_features: np.ndarray,
    all_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train_features, axis=0, dtype=np.float64)
    std = np.std(train_features, axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((all_features - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def _fit_ridge(features: np.ndarray, targets: np.ndarray, *, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(features.shape[0]), features])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ targets)


def _predict_ridge(features: np.ndarray, beta: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(features.shape[0]), features])
    return design @ beta


def _load_frozen_encoder(
    checkpoint_path: str | Path,
    *,
    in_channels: int,
    device: torch.device,
    model_name: str | None,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_model_name = str(checkpoint.get("model_name", "tiny-cnn"))
    selected_model_name = model_name or checkpoint_model_name
    if selected_model_name != checkpoint_model_name:
        msg = (
            f"Checkpoint encoder is {checkpoint_model_name!r}, but requested model is "
            f"{selected_model_name!r}."
        )
        raise ValueError(msg)
    state = checkpoint.get("encoder_state_dict")
    if state is None:
        msg = f"Checkpoint {checkpoint_path} does not contain encoder_state_dict."
        raise ValueError(msg)
    encoder = build_encoder(selected_model_name, in_channels=in_channels).to(device)
    encoder.load_state_dict(state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    metadata = {
        **checkpoint_provenance(checkpoint_path, checkpoint),
        "source_model": checkpoint.get("model"),
        "source_model_name": checkpoint_model_name,
    }
    return encoder, metadata


def _load_frozen_jepa_rollout_components(
    checkpoint_path: str | Path,
    *,
    in_channels: int,
    device: torch.device,
    model_name: str | None,
) -> tuple[nn.Module, nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_model_name = str(checkpoint.get("model_name", "tiny-cnn"))
    selected_model_name = model_name or checkpoint_model_name
    if selected_model_name != checkpoint_model_name:
        msg = (
            f"Checkpoint encoder is {checkpoint_model_name!r}, but requested model is "
            f"{selected_model_name!r}."
        )
        raise ValueError(msg)
    state = checkpoint.get("encoder_state_dict")
    if state is None:
        msg = f"Checkpoint {checkpoint_path} does not contain encoder_state_dict."
        raise ValueError(msg)
    encoder = build_encoder(selected_model_name, in_channels=in_channels).to(device)
    encoder.load_state_dict(state)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    predictor, predictor_metadata = _build_frozen_jepa_predictor(
        checkpoint,
        encoder=encoder,
        device=device,
    )
    metadata = {
        **checkpoint_provenance(checkpoint_path, checkpoint),
        "source_model": checkpoint.get("model"),
        "source_model_name": checkpoint_model_name,
        "objective": checkpoint.get("objective"),
        "mask_mode": checkpoint.get("mask_mode"),
    }
    return encoder, predictor, checkpoint, {**metadata, **predictor_metadata}


def _encoder_batch_features(
    encoder: nn.Module,
    x: torch.Tensor,
    *,
    token_summary: str,
) -> torch.Tensor:
    if token_summary == "mean-std" and hasattr(encoder, "forward_tokens"):
        tokens = encoder.forward_tokens(x)
        return torch.cat(
            [
                tokens.mean(dim=1),
                tokens.std(dim=1, unbiased=False),
            ],
            dim=1,
        )
    if token_summary == "mean":
        return encoder(x)
    msg = "token_summary must be one of {'mean', 'mean-std'}."
    raise ValueError(msg)


def _extract_latent_features(
    encoder: nn.Module,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    token_summary: str,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size].astype(np.float32, copy=False))
            batch = batch.to(device=device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                features = _encoder_batch_features(
                    encoder,
                    batch,
                    token_summary=token_summary,
                )
            rows.append(features.detach().float().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def _token_summary_features(tokens: torch.Tensor, *, token_summary: str) -> torch.Tensor:
    if token_summary == "mean":
        return tokens.mean(dim=1)
    if token_summary == "mean-std":
        return torch.cat(
            [
                tokens.mean(dim=1),
                tokens.std(dim=1, unbiased=False),
            ],
            dim=1,
        )
    msg = "token_summary must be one of {'mean', 'mean-std'}."
    raise ValueError(msg)


def _compose_rollout_features(
    *,
    context_summary: torch.Tensor,
    pred_summary: torch.Tensor,
    horizons_ms: tuple[int, ...],
    include_context_latent: bool,
    feature_mode: str,
) -> torch.Tensor:
    if feature_mode not in {"flat", "dynamics"}:
        msg = "feature_mode must be one of {'flat', 'dynamics'}."
        raise ValueError(msg)
    pieces: list[torch.Tensor] = []
    if include_context_latent:
        pieces.append(context_summary)
    pieces.append(pred_summary.reshape(pred_summary.shape[0], -1))
    if feature_mode == "flat":
        return torch.cat(pieces, dim=1)

    horizon_seconds = torch.tensor(
        [max(horizon_ms, 1) / 1000.0 for horizon_ms in horizons_ms],
        dtype=pred_summary.dtype,
        device=pred_summary.device,
    ).view(1, -1, 1)
    deltas = pred_summary - context_summary[:, None, :]
    velocities = deltas / horizon_seconds
    dynamics_pieces = [
        deltas.reshape(deltas.shape[0], -1),
        velocities.reshape(velocities.shape[0], -1),
    ]
    if pred_summary.shape[1] > 1:
        step_seconds = torch.diff(horizon_seconds, dim=1).clamp_min(1e-3)
        step_velocity = (pred_summary[:, 1:, :] - pred_summary[:, :-1, :]) / step_seconds
        dynamics_pieces.append(step_velocity.reshape(step_velocity.shape[0], -1))
    scalar_features = [
        torch.linalg.vector_norm(deltas, dim=2),
        torch.linalg.vector_norm(velocities, dim=2),
        torch.nn.functional.cosine_similarity(
            pred_summary,
            context_summary[:, None, :],
            dim=2,
            eps=1e-6,
        ),
    ]
    if pred_summary.shape[1] > 1:
        scalar_features.append(torch.linalg.vector_norm(step_velocity, dim=2))
    dynamics_pieces.append(torch.cat(scalar_features, dim=1))
    pieces.extend(dynamics_pieces)
    return torch.cat(pieces, dim=1)


def _build_frozen_jepa_predictor(
    checkpoint: dict[str, Any],
    *,
    encoder: nn.Module,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    predictor_state = checkpoint.get("predictor_state_dict")
    if predictor_state is None:
        msg = "JEPA checkpoint does not contain predictor_state_dict."
        raise ValueError(msg)
    horizons_ms = tuple(int(value) for value in checkpoint.get("temporal_horizons_ms", []))
    if not horizons_ms:
        msg = "JEPA rollout probing requires a temporal JEPA checkpoint."
        raise ValueError(msg)
    dense_tokens = bool(checkpoint.get("dense_tokens", False))
    action_feature_dim = int(
        checkpoint.get(
            "action_feature_dim",
            checkpoint.get("motion_feature_dim", 0),
        )
    )
    context_token_loss = bool(checkpoint.get("context_token_loss", False))
    horizon_count = len(horizons_ms) + int(context_token_loss)
    if dense_tokens:
        deep_layers = tuple(int(value) for value in checkpoint.get("deep_supervision_layers", []))
        layer_count = max(deep_layers) + 1 if deep_layers else 0
        kwargs = {
            "dim": int(encoder.output_dim),
            "horizon_count": horizon_count,
            "motion_dim": action_feature_dim,
            "layer_count": layer_count,
        }
        dense_predictor = str(checkpoint.get("dense_predictor", "mlp"))
        if dense_predictor == "transformer":
            predictor: nn.Module = DenseTemporalTransformerJEPAPredictor(**kwargs).to(device)
        elif dense_predictor == "mlp":
            predictor = DenseTemporalJEPAPredictor(**kwargs).to(device)
        else:
            msg = f"Unsupported dense_predictor in checkpoint: {dense_predictor!r}."
            raise ValueError(msg)
    else:
        predictor = TemporalJEPAPredictor(
            dim=int(encoder.output_dim),
            horizon_count=len(horizons_ms),
        ).to(device)
        deep_layers = ()
        dense_predictor = "pooled"

    predictor.load_state_dict(predictor_state)
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)
    metadata = {
        "temporal_horizons_ms": list(horizons_ms),
        "dense_tokens": dense_tokens,
        "dense_predictor": dense_predictor,
        "context_token_loss": context_token_loss,
        "action_feature_dim": action_feature_dim,
        "action_feature_names": list(checkpoint.get("action_feature_names", [])),
        "deep_supervision_layers": list(deep_layers),
    }
    return predictor, metadata


def _extract_predicted_rollout_features(
    *,
    encoder: nn.Module,
    predictor: nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
    token_summary: str,
    horizons_ms: tuple[int, ...],
    bins: int,
    metadata_channels: bool,
    navigation_feature_count: int,
    action_feature_mean: torch.Tensor | None,
    action_feature_std: torch.Tensor | None,
    action_feature_dim: int,
    include_context_latent: bool,
    feature_mode: str,
    deep_supervision_layers: tuple[int, ...] = (),
) -> np.ndarray:
    rows: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    horizon_ids = torch.arange(len(horizons_ms), dtype=torch.long, device=device)
    layer_id = deep_supervision_layers[-1] if deep_supervision_layers else None
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size].astype(np.float32, copy=False))
            batch = batch.to(device=device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if hasattr(encoder, "forward_tokens"):
                    if layer_id is not None:
                        if not hasattr(encoder, "forward_intermediate_tokens"):
                            msg = (
                                f"{encoder.__class__.__name__} does not support "
                                "rollout probing from intermediate tokens."
                            )
                            raise ValueError(msg)
                        context_tokens = encoder.forward_intermediate_tokens(batch, (layer_id,))[0]
                    else:
                        context_tokens = encoder.forward_tokens(batch)
                    motion_features = None
                    if action_feature_dim:
                        motion_features = _normalize_action_features(
                            _context_action_features(
                                batch,
                                bins=bins,
                                metadata_channels=metadata_channels,
                                navigation_feature_count=navigation_feature_count,
                            ),
                            mean=action_feature_mean,
                            std=action_feature_std,
                        )
                    pred_tokens = predictor(context_tokens, horizon_ids, motion_features, layer_id)
                    batch_size_actual, horizon_count, token_count, dim = pred_tokens.shape
                    pred_summary = _token_summary_features(
                        pred_tokens.reshape(batch_size_actual * horizon_count, token_count, dim),
                        token_summary=token_summary,
                    ).view(batch_size_actual, horizon_count, -1)
                    context_summary = _token_summary_features(
                        context_tokens,
                        token_summary=token_summary,
                    )
                    features = _compose_rollout_features(
                        context_summary=context_summary,
                        pred_summary=pred_summary,
                        horizons_ms=horizons_ms,
                        include_context_latent=include_context_latent,
                        feature_mode=feature_mode,
                    )
                else:
                    if action_feature_dim:
                        msg = "Pooled JEPA rollout probing does not support action conditioning."
                        raise ValueError(msg)
                    context_z = encoder(batch)
                    pred_z = predictor(context_z, horizon_ids)
                    features = _compose_rollout_features(
                        context_summary=context_z,
                        pred_summary=pred_z,
                        horizons_ms=horizons_ms,
                        include_context_latent=include_context_latent,
                        feature_mode=feature_mode,
                    )
            rows.append(features.detach().float().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def _extract_physics_features(
    x: np.ndarray,
    *,
    bins: int,
    metadata_channels: bool,
    navigation_feature_count: int,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, int(x.shape[0]), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size].astype(np.float32, copy=False))
        features = _context_action_features(
            batch,
            bins=bins,
            metadata_channels=metadata_channels,
            navigation_feature_count=navigation_feature_count,
        )
        rows.append(features.detach().cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    *,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    use_amp = scaler is not None and device.type == "cuda"
    losses: list[float] = []
    for features, prior, target in loader:
        features = features.to(device=device, non_blocking=True)
        prior = prior.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(features, prior)
            loss = loss_fn(pred, target)
        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _evaluate_prober(
    model: nn.Module,
    dataset: LatentProberDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for features, prior, target in loader:
            features = features.to(device=device, non_blocking=True)
            prior = prior.to(device=device, non_blocking=True)
            pred_log = model(features, prior).detach().cpu().numpy()
            y_pred.append(np.exp(pred_log))
            y_true.append(np.exp(target.numpy()))
    elapsed = time.perf_counter() - start
    if not y_true:
        return np.empty(0), np.empty(0), elapsed
    return np.concatenate(y_true), np.concatenate(y_pred), elapsed


def _prior_metrics(
    y_log: np.ndarray,
    prior_log: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    y_true = np.exp(y_log[indices])
    y_pred = np.exp(prior_log[indices])
    return regression_metrics(y_true, y_pred)


def _match_cache_indices(
    *,
    cache_sequence_id: np.ndarray,
    cache_timestamp_us: np.ndarray,
    rows: list[ROIEventRow],
    max_slop_us: int,
) -> tuple[np.ndarray, list[int]]:
    """Match bbox rows to nearest same-sequence cache windows."""

    by_sequence = {
        sequence_id: np.flatnonzero(cache_sequence_id == sequence_id).astype(np.int64)
        for sequence_id in sorted(set(cache_sequence_id.tolist()))
    }
    matched: list[int] = []
    unmatched_positions: list[int] = []
    for row_position, row in enumerate(rows):
        candidates = by_sequence.get(row.sequence_id)
        if candidates is None or candidates.size == 0:
            unmatched_positions.append(row_position)
            continue
        deltas = np.abs(cache_timestamp_us[candidates] - int(row.timestamp_us))
        best_local = int(np.argmin(deltas))
        if int(deltas[best_local]) > max_slop_us:
            unmatched_positions.append(row_position)
            continue
        matched.append(int(candidates[best_local]))
    return np.array(matched, dtype=np.int64), unmatched_positions


def _load_roi_rows(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    split_names: tuple[str, ...],
    context_ms: int,
) -> tuple[list[ROIEventRow], np.ndarray, dict[str, int]]:
    sequences = {sequence.sequence_id: sequence for sequence in read_manifest(manifest_path)}
    measurements_by_sequence = load_measurements_from_manifest(manifest_path)
    splits = read_splits(split_path)
    all_rows: list[ROIEventRow] = []
    row_splits: list[str] = []
    label_count_by_split: dict[str, int] = {}
    for split_name in split_names:
        split_rows: list[ROIEventRow] = []
        total_labels = 0
        for sequence_id in splits.get(split_name, []):
            sequence = sequences.get(sequence_id)
            measurements = measurements_by_sequence.get(sequence_id, [])
            total_labels += len(measurements)
            if sequence is None:
                continue
            split_rows.extend(
                _sequence_rows(
                    sequence,
                    measurements,
                    context_ms=context_ms,
                )
            )
        label_count_by_split[split_name] = total_labels
        all_rows.extend(split_rows)
        row_splits.extend([split_name] * len(split_rows))
    return all_rows, np.array(row_splits, dtype=str), label_count_by_split


def train_latent_ttc_prober(
    *,
    cache_path: str | Path,
    encoder_checkpoint_path: str | Path,
    output_dir: str | Path,
    epochs: int = 80,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device_name: str = "auto",
    model_name: str | None = None,
    token_summary: str = "mean-std",
    hidden_dim: int = 256,
    dropout: float = 0.05,
    physics_prior: str = "ridge",
    ridge_alpha: float = 1.0,
    train_splits: tuple[str, ...] = ("train",),
    validation_splits: tuple[str, ...] = ("validation",),
    evaluation_splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, Any]:
    """Train a frozen-latent residual TTC prober on a materialized voxel cache."""

    _set_seed(seed)
    if epochs <= 0:
        msg = "epochs must be positive."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)
    if physics_prior not in {"none", "ridge"}:
        msg = "physics_prior must be one of {'none', 'ridge'}."
        raise ValueError(msg)
    if token_summary not in {"mean", "mean-std"}:
        msg = "token_summary must be one of {'mean', 'mean-std'}."
        raise ValueError(msg)
    if not train_splits or not validation_splits or not evaluation_splits:
        msg = "train_splits, validation_splits and evaluation_splits must be non-empty."
        raise ValueError(msg)

    cache = np.load(cache_path, allow_pickle=False)
    validate_voxel_cache(cache)
    x = cache["x"]
    y_ttc = cache["y_ttc"].astype(np.float32)
    y_log = np.log(np.clip(y_ttc, 1e-4, None)).astype(np.float32)
    split = cache["split"].astype(str)
    bins = int(cache["bins"]) if "bins" in cache.files else int(x.shape[1] // 2)
    metadata_channels = _cache_bool(cache, "metadata_channels")
    navigation_channels = _cache_bool(cache, "navigation_channels")
    navigation_feature_names = (
        _cache_string_tuple(cache, "navigation_feature_names", NAVIGATION_FEATURE_NAMES)
        if navigation_channels
        else ()
    )
    navigation_feature_count = len(navigation_feature_names)
    available_splits = _available_split_names(split)
    missing_names = sorted(
        (set(train_splits) | set(validation_splits) | set(evaluation_splits)) - available_splits
    )
    if missing_names:
        msg = f"Requested split names are missing from cache: {missing_names}."
        raise ValueError(msg)
    train_idx = _split_indices_any(split, train_splits)
    val_idx = _split_indices_any(split, validation_splits)
    if train_idx.size == 0 or val_idx.size == 0:
        msg = "Requested train and validation splits must be non-empty."
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

    start_time = time.perf_counter()
    encoder, encoder_metadata = _load_frozen_encoder(
        encoder_checkpoint_path,
        in_channels=int(x.shape[1]),
        device=device,
        model_name=model_name,
    )
    latent_features = _extract_latent_features(
        encoder,
        x,
        batch_size=batch_size,
        device=device,
        token_summary=token_summary,
    )
    physics_features = _extract_physics_features(
        x,
        bins=bins,
        metadata_channels=metadata_channels,
        navigation_feature_count=navigation_feature_count,
        batch_size=batch_size,
    )
    physics_scaled, physics_mean, physics_std = _standardize_train(
        physics_features[train_idx],
        physics_features,
    )
    prior_beta: np.ndarray | None = None
    if physics_prior == "ridge":
        prior_beta = _fit_ridge(
            physics_scaled[train_idx].astype(np.float64),
            y_log[train_idx].astype(np.float64),
            alpha=ridge_alpha,
        )
        prior_log = _predict_ridge(physics_scaled.astype(np.float64), prior_beta).astype(np.float32)
    else:
        prior_log = np.zeros_like(y_log, dtype=np.float32)
    raw_features = np.concatenate(
        [
            latent_features,
            physics_scaled,
            prior_log[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    features, feature_mean, feature_std = _standardize_train(
        raw_features[train_idx],
        raw_features,
    )

    train_dataset = LatentProberDataset(features, prior_log, y_log, train_idx)
    val_dataset = LatentProberDataset(features, prior_log, y_log, val_idx)
    datasets = {
        split_name: LatentProberDataset(features, prior_log, y_log, indices)
        for split_name, indices in split_indices.items()
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    prober = LatentResidualTTCProber(
        int(features.shape[1]),
        hidden_dim=hidden_dim,
        dropout=dropout,
        residual=physics_prior != "none",
    ).to(device)
    optimizer = torch.optim.AdamW(prober.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "latent_prober_best.pt"
    last_path = output / "latent_prober_last.pt"
    metrics_path = output / "metrics.json"
    history_path = output / "history.jsonl"
    best_val_mae = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    checkpoint_common = {
        "model": "latent_ttc_prober",
        "encoder_checkpoint": encoder_metadata,
        "cache_path": str(cache_path),
        "seed": seed,
        "input_dim": int(features.shape[1]),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "token_summary": token_summary,
        "physics_prior": physics_prior,
        "ridge_alpha": ridge_alpha,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "physics_feature_mean": physics_mean.tolist(),
        "physics_feature_std": physics_std.tolist(),
        "prior_beta": prior_beta.tolist() if prior_beta is not None else [],
        "event_motion_feature_names": list(EVENT_MOTION_FEATURE_NAMES),
        "navigation_feature_names": list(navigation_feature_names),
    }

    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, epochs + 1):
            train_loss = _train_one_epoch(
                prober,
                train_loader,
                optimizer,
                loss_fn,
                device=device,
                scaler=scaler,
            )
            val_true, val_pred, val_seconds = _evaluate_prober(
                prober,
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
                        **checkpoint_common,
                        "epoch": epoch,
                        "prober_state_dict": prober.state_dict(),
                    },
                    best_path,
                )

    torch.save(
        {
            **checkpoint_common,
            "epoch": epochs,
            "prober_state_dict": prober.state_dict(),
        },
        last_path,
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    prober.load_state_dict(checkpoint["prober_state_dict"])

    split_results: dict[str, Any] = {}
    predictions: dict[str, list[float]] = {}
    targets: dict[str, list[float]] = {}
    for split_name in evaluation_splits:
        dataset = datasets[split_name]
        y_true, y_pred, seconds = _evaluate_prober(
            prober,
            dataset,
            device=device,
            batch_size=batch_size,
        )
        split_results[split_name] = {
            "count": int(y_true.shape[0]),
            "metrics": regression_metrics(y_true, y_pred),
            "prior_metrics": _prior_metrics(y_log, prior_log, split_indices[split_name]),
            "seconds": seconds,
        }
        predictions[split_name] = y_pred.astype(float).tolist()
        targets[split_name] = y_true.astype(float).tolist()

    summary: dict[str, Any] = {
        "model": "latent_ttc_prober",
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
        "encoder_checkpoint": encoder_metadata,
        "token_summary": token_summary,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "physics_prior": physics_prior,
        "ridge_alpha": ridge_alpha,
        "train_splits": list(train_splits),
        "validation_splits": list(validation_splits),
        "evaluation_splits": list(evaluation_splits),
        "train_count": int(train_idx.size),
        "validation_count": int(val_idx.size),
        "latent_feature_dim": int(latent_features.shape[1]),
        "physics_feature_dim": int(physics_features.shape[1]),
        "input_dim": int(features.shape[1]),
        "best_epoch": best_epoch,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "last": history[-1] if history else None,
        "feature_names": {
            "latent": f"frozen_encoder_{token_summary}",
            "physics": [*EVENT_MOTION_FEATURE_NAMES, *navigation_feature_names],
            "prior": physics_prior,
        },
        "leakage_audit": {
            "encoder_frozen": True,
            "encoder_checkpoint_selected_before_prober": True,
            "uses_ttc_labels_for_encoder": False,
            "uses_ttc_labels_for_prober": True,
            "uses_validation_or_test_ttc_for_prior_fit": False,
            "uses_validation_or_test_ttc_for_feature_scaling": False,
            "uses_future_navigation": False,
            "physics_features_use_context_only": True,
            "test_evaluated_only_if_requested": "test" in evaluation_splits,
        },
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


def train_roi_latent_ttc_prober(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    cache_path: str | Path,
    encoder_checkpoint_path: str | Path,
    output_dir: str | Path,
    context_ms: int = 100,
    max_cache_slop_ms: int = 12,
    epochs: int = 160,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device_name: str = "auto",
    model_name: str | None = None,
    token_summary: str = "mean-std",
    hidden_dim: int = 128,
    dropout: float = 0.05,
    physics_prior: str = "ridge",
    ridge_alpha: float = 1.0,
    train_splits: tuple[str, ...] = ("train",),
    validation_splits: tuple[str, ...] = ("validation",),
    evaluation_splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, Any]:
    """Train a detection-assisted frozen-latent bbox/ROI TTC prober."""

    _set_seed(seed)
    if epochs <= 0:
        msg = "epochs must be positive."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)
    if context_ms <= 0 or max_cache_slop_ms < 0:
        msg = "context_ms must be positive and max_cache_slop_ms must be non-negative."
        raise ValueError(msg)
    if physics_prior not in {"none", "ridge"}:
        msg = "physics_prior must be one of {'none', 'ridge'}."
        raise ValueError(msg)
    if token_summary not in {"mean", "mean-std"}:
        msg = "token_summary must be one of {'mean', 'mean-std'}."
        raise ValueError(msg)
    selected_split_names = tuple(
        dict.fromkeys((*train_splits, *validation_splits, *evaluation_splits))
    )
    all_rows, row_splits, label_count_by_split = _load_roi_rows(
        manifest_path=manifest_path,
        split_path=split_path,
        split_names=selected_split_names,
        context_ms=context_ms,
    )
    if not all_rows:
        msg = "No bbox/ROI rows could be loaded for the requested splits."
        raise ValueError(msg)

    cache = np.load(cache_path, allow_pickle=False)
    validate_voxel_cache(cache)
    x = cache["x"]
    cache_sequence_id = cache["sequence_id"].astype(str)
    cache_timestamp_us = cache["timestamp_us"].astype(np.int64)
    cache_indices, unmatched_positions = _match_cache_indices(
        cache_sequence_id=cache_sequence_id,
        cache_timestamp_us=cache_timestamp_us,
        rows=all_rows,
        max_slop_us=int(max_cache_slop_ms * 1000),
    )
    unmatched = set(unmatched_positions)
    matched_rows = [row for idx, row in enumerate(all_rows) if idx not in unmatched]
    matched_splits = np.array(
        [split_name for idx, split_name in enumerate(row_splits.tolist()) if idx not in unmatched],
        dtype=str,
    )
    if not matched_rows:
        msg = "No bbox/ROI rows matched cache windows; increase max_cache_slop_ms."
        raise ValueError(msg)
    if cache_indices.shape[0] != len(matched_rows):
        msg = "Internal row/cache matching mismatch."
        raise RuntimeError(msg)

    train_idx = _split_indices_any(matched_splits, train_splits)
    val_idx = _split_indices_any(matched_splits, validation_splits)
    if train_idx.size == 0 or val_idx.size == 0:
        msg = "Requested train and validation splits must have matched ROI rows."
        raise ValueError(msg)
    split_indices = {
        split_name: _split_indices(matched_splits, split_name)
        for split_name in set(matched_splits.tolist())
    }
    missing_eval_splits = [
        split_name
        for split_name in evaluation_splits
        if split_name not in split_indices or split_indices[split_name].size == 0
    ]
    if missing_eval_splits:
        msg = f"Requested evaluation splits have no matched ROI rows: {missing_eval_splits}."
        raise ValueError(msg)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    start_time = time.perf_counter()
    encoder, encoder_metadata = _load_frozen_encoder(
        encoder_checkpoint_path,
        in_channels=int(x.shape[1]),
        device=device,
        model_name=model_name,
    )
    all_latents = _extract_latent_features(
        encoder,
        x,
        batch_size=batch_size,
        device=device,
        token_summary=token_summary,
    )
    latent_features = all_latents[cache_indices]
    roi_features = np.array([row.features for row in matched_rows], dtype=np.float32)
    roi_scaled, roi_mean, roi_std = _standardize_train(roi_features[train_idx], roi_features)
    y_ttc = np.array([row.ttc_seconds for row in matched_rows], dtype=np.float32)
    y_log = np.log(np.clip(y_ttc, 1e-4, None)).astype(np.float32)
    prior_beta: np.ndarray | None = None
    if physics_prior == "ridge":
        prior_beta = _fit_ridge(
            roi_scaled[train_idx].astype(np.float64),
            y_log[train_idx].astype(np.float64),
            alpha=ridge_alpha,
        )
        prior_log = _predict_ridge(roi_scaled.astype(np.float64), prior_beta).astype(np.float32)
    else:
        prior_log = np.zeros_like(y_log, dtype=np.float32)
    raw_features = np.concatenate(
        [
            latent_features,
            roi_scaled,
            prior_log[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    features, feature_mean, feature_std = _standardize_train(raw_features[train_idx], raw_features)

    train_dataset = LatentProberDataset(features, prior_log, y_log, train_idx)
    val_dataset = LatentProberDataset(features, prior_log, y_log, val_idx)
    datasets = {
        split_name: LatentProberDataset(features, prior_log, y_log, indices)
        for split_name, indices in split_indices.items()
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    prober = LatentResidualTTCProber(
        int(features.shape[1]),
        hidden_dim=hidden_dim,
        dropout=dropout,
        residual=physics_prior != "none",
    ).to(device)
    optimizer = torch.optim.AdamW(prober.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "roi_latent_prober_best.pt"
    last_path = output / "roi_latent_prober_last.pt"
    metrics_path = output / "metrics.json"
    history_path = output / "history.jsonl"
    best_val_mae = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    checkpoint_common = {
        "model": "roi_latent_ttc_prober",
        "encoder_checkpoint": encoder_metadata,
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
        "cache_path": str(cache_path),
        "seed": seed,
        "input_dim": int(features.shape[1]),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "token_summary": token_summary,
        "physics_prior": physics_prior,
        "ridge_alpha": ridge_alpha,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "roi_feature_mean": roi_mean.tolist(),
        "roi_feature_std": roi_std.tolist(),
        "prior_beta": prior_beta.tolist() if prior_beta is not None else [],
        "roi_feature_names": list(ROI_EVENT_FEATURE_NAMES),
    }

    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, epochs + 1):
            train_loss = _train_one_epoch(
                prober,
                train_loader,
                optimizer,
                loss_fn,
                device=device,
                scaler=scaler,
            )
            val_true, val_pred, val_seconds = _evaluate_prober(
                prober,
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
                        **checkpoint_common,
                        "epoch": epoch,
                        "prober_state_dict": prober.state_dict(),
                    },
                    best_path,
                )

    torch.save(
        {
            **checkpoint_common,
            "epoch": epochs,
            "prober_state_dict": prober.state_dict(),
        },
        last_path,
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    prober.load_state_dict(checkpoint["prober_state_dict"])

    split_results: dict[str, Any] = {}
    predictions: dict[str, list[float]] = {}
    targets: dict[str, list[float]] = {}
    for split_name in evaluation_splits:
        indices = split_indices[split_name]
        dataset = datasets[split_name]
        y_true, y_pred, seconds = _evaluate_prober(
            prober,
            dataset,
            device=device,
            batch_size=batch_size,
        )
        split_results[split_name] = {
            "label_count": int(label_count_by_split.get(split_name, 0)),
            "matched_count": int(indices.shape[0]),
            "unmatched_count": int(
                max(label_count_by_split.get(split_name, 0) - indices.shape[0], 0)
            ),
            "metrics": regression_metrics(y_true, y_pred),
            "prior_metrics": _prior_metrics(y_log, prior_log, indices),
            "seconds": seconds,
        }
        predictions[split_name] = y_pred.astype(float).tolist()
        targets[split_name] = y_true.astype(float).tolist()

    summary: dict[str, Any] = {
        "model": "roi_latent_ttc_prober",
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
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
        "encoder_checkpoint": encoder_metadata,
        "context_ms": context_ms,
        "max_cache_slop_ms": max_cache_slop_ms,
        "token_summary": token_summary,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "physics_prior": physics_prior,
        "ridge_alpha": ridge_alpha,
        "train_splits": list(train_splits),
        "validation_splits": list(validation_splits),
        "evaluation_splits": list(evaluation_splits),
        "loaded_roi_rows": int(len(all_rows)),
        "matched_roi_rows": int(len(matched_rows)),
        "unmatched_roi_rows": int(len(unmatched_positions)),
        "train_count": int(train_idx.size),
        "validation_count": int(val_idx.size),
        "latent_feature_dim": int(latent_features.shape[1]),
        "roi_feature_dim": int(roi_features.shape[1]),
        "input_dim": int(features.shape[1]),
        "best_epoch": best_epoch,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "last": history[-1] if history else None,
        "feature_names": {
            "latent": f"frozen_encoder_{token_summary}",
            "roi": list(ROI_EVENT_FEATURE_NAMES),
            "prior": physics_prior,
        },
        "leakage_audit": {
            "encoder_frozen": True,
            "encoder_checkpoint_selected_before_prober": True,
            "uses_ttc_labels_for_encoder": False,
            "uses_ttc_labels_for_prober": True,
            "uses_validation_or_test_ttc_for_prior_fit": False,
            "uses_validation_or_test_ttc_for_feature_scaling": False,
            "uses_future_events": False,
            "uses_current_bbox": True,
            "uses_future_bboxes": False,
            "uses_future_navigation": False,
            "roi_event_window": "[timestamp - context_ms, timestamp] only",
            "test_evaluated_only_if_requested": "test" in evaluation_splits,
        },
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


def train_roi_rollout_ttc_prober(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    cache_path: str | Path,
    jepa_checkpoint_path: str | Path,
    output_dir: str | Path,
    context_ms: int = 100,
    max_cache_slop_ms: int = 12,
    epochs: int = 160,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-3,
    seed: int = 42,
    device_name: str = "auto",
    model_name: str | None = None,
    rollout_token_summary: str = "mean-std",
    rollout_include_context: bool = True,
    rollout_feature_mode: str = "flat",
    hidden_dim: int = 128,
    dropout: float = 0.10,
    physics_prior: str = "ridge",
    ridge_alpha: float = 1.0,
    train_splits: tuple[str, ...] = ("train",),
    validation_splits: tuple[str, ...] = ("validation",),
    evaluation_splits: tuple[str, ...] = ("train", "validation"),
) -> dict[str, Any]:
    """Train a bbox/ROI TTC prober on frozen JEPA-predicted latent rollouts."""

    _set_seed(seed)
    if epochs <= 0:
        msg = "epochs must be positive."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)
    if context_ms <= 0 or max_cache_slop_ms < 0:
        msg = "context_ms must be positive and max_cache_slop_ms must be non-negative."
        raise ValueError(msg)
    if physics_prior not in {"none", "ridge"}:
        msg = "physics_prior must be one of {'none', 'ridge'}."
        raise ValueError(msg)
    if rollout_token_summary not in {"mean", "mean-std"}:
        msg = "rollout_token_summary must be one of {'mean', 'mean-std'}."
        raise ValueError(msg)
    if rollout_feature_mode not in {"flat", "dynamics"}:
        msg = "rollout_feature_mode must be one of {'flat', 'dynamics'}."
        raise ValueError(msg)
    selected_split_names = tuple(
        dict.fromkeys((*train_splits, *validation_splits, *evaluation_splits))
    )
    all_rows, row_splits, label_count_by_split = _load_roi_rows(
        manifest_path=manifest_path,
        split_path=split_path,
        split_names=selected_split_names,
        context_ms=context_ms,
    )
    if not all_rows:
        msg = "No bbox/ROI rows could be loaded for the requested splits."
        raise ValueError(msg)

    cache = np.load(cache_path, allow_pickle=False)
    validate_voxel_cache(cache)
    x = cache["x"]
    bins = int(cache["bins"]) if "bins" in cache.files else int(x.shape[1] // 2)
    metadata_channels = _cache_bool(cache, "metadata_channels")
    navigation_channels = _cache_bool(cache, "navigation_channels")
    navigation_feature_names = (
        _cache_string_tuple(cache, "navigation_feature_names", NAVIGATION_FEATURE_NAMES)
        if navigation_channels
        else ()
    )
    navigation_feature_count = len(navigation_feature_names)
    cache_sequence_id = cache["sequence_id"].astype(str)
    cache_timestamp_us = cache["timestamp_us"].astype(np.int64)
    cache_indices, unmatched_positions = _match_cache_indices(
        cache_sequence_id=cache_sequence_id,
        cache_timestamp_us=cache_timestamp_us,
        rows=all_rows,
        max_slop_us=int(max_cache_slop_ms * 1000),
    )
    unmatched = set(unmatched_positions)
    matched_rows = [row for idx, row in enumerate(all_rows) if idx not in unmatched]
    matched_splits = np.array(
        [split_name for idx, split_name in enumerate(row_splits.tolist()) if idx not in unmatched],
        dtype=str,
    )
    if not matched_rows:
        msg = "No bbox/ROI rows matched cache windows; increase max_cache_slop_ms."
        raise ValueError(msg)
    if cache_indices.shape[0] != len(matched_rows):
        msg = "Internal row/cache matching mismatch."
        raise RuntimeError(msg)

    train_idx = _split_indices_any(matched_splits, train_splits)
    val_idx = _split_indices_any(matched_splits, validation_splits)
    if train_idx.size == 0 or val_idx.size == 0:
        msg = "Requested train and validation splits must have matched ROI rows."
        raise ValueError(msg)
    split_indices = {
        split_name: _split_indices(matched_splits, split_name)
        for split_name in set(matched_splits.tolist())
    }
    missing_eval_splits = [
        split_name
        for split_name in evaluation_splits
        if split_name not in split_indices or split_indices[split_name].size == 0
    ]
    if missing_eval_splits:
        msg = f"Requested evaluation splits have no matched ROI rows: {missing_eval_splits}."
        raise ValueError(msg)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    start_time = time.perf_counter()
    encoder, predictor, jepa_checkpoint, jepa_metadata = _load_frozen_jepa_rollout_components(
        jepa_checkpoint_path,
        in_channels=int(x.shape[1]),
        device=device,
        model_name=model_name,
    )
    horizons_ms = tuple(int(value) for value in jepa_metadata["temporal_horizons_ms"])
    action_feature_dim = int(jepa_metadata["action_feature_dim"])
    action_feature_mean = (
        torch.tensor(jepa_checkpoint.get("action_feature_mean", []), dtype=torch.float32)
        if action_feature_dim
        else None
    )
    action_feature_std = (
        torch.tensor(jepa_checkpoint.get("action_feature_std", []), dtype=torch.float32)
        if action_feature_dim
        else None
    )
    if action_feature_dim and (
        action_feature_mean is None
        or action_feature_std is None
        or action_feature_mean.numel() != action_feature_dim
        or action_feature_std.numel() != action_feature_dim
    ):
        msg = "JEPA checkpoint is missing valid train-fitted action feature statistics."
        raise ValueError(msg)

    all_rollout_features = _extract_predicted_rollout_features(
        encoder=encoder,
        predictor=predictor,
        x=x,
        batch_size=batch_size,
        device=device,
        token_summary=rollout_token_summary,
        horizons_ms=horizons_ms,
        bins=bins,
        metadata_channels=metadata_channels,
        navigation_feature_count=navigation_feature_count,
        action_feature_mean=action_feature_mean,
        action_feature_std=action_feature_std,
        action_feature_dim=action_feature_dim,
        include_context_latent=rollout_include_context,
        feature_mode=rollout_feature_mode,
        deep_supervision_layers=tuple(jepa_metadata["deep_supervision_layers"]),
    )
    rollout_features = all_rollout_features[cache_indices]
    roi_features = np.array([row.features for row in matched_rows], dtype=np.float32)
    roi_scaled, roi_mean, roi_std = _standardize_train(roi_features[train_idx], roi_features)
    y_ttc = np.array([row.ttc_seconds for row in matched_rows], dtype=np.float32)
    y_log = np.log(np.clip(y_ttc, 1e-4, None)).astype(np.float32)
    prior_beta: np.ndarray | None = None
    if physics_prior == "ridge":
        prior_beta = _fit_ridge(
            roi_scaled[train_idx].astype(np.float64),
            y_log[train_idx].astype(np.float64),
            alpha=ridge_alpha,
        )
        prior_log = _predict_ridge(roi_scaled.astype(np.float64), prior_beta).astype(np.float32)
    else:
        prior_log = np.zeros_like(y_log, dtype=np.float32)

    raw_features = np.concatenate(
        [
            rollout_features,
            roi_scaled,
            prior_log[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    features, feature_mean, feature_std = _standardize_train(raw_features[train_idx], raw_features)

    train_dataset = LatentProberDataset(features, prior_log, y_log, train_idx)
    val_dataset = LatentProberDataset(features, prior_log, y_log, val_idx)
    datasets = {
        split_name: LatentProberDataset(features, prior_log, y_log, indices)
        for split_name, indices in split_indices.items()
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    prober = LatentResidualTTCProber(
        int(features.shape[1]),
        hidden_dim=hidden_dim,
        dropout=dropout,
        residual=physics_prior != "none",
    ).to(device)
    optimizer = torch.optim.AdamW(prober.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "roi_rollout_prober_best.pt"
    last_path = output / "roi_rollout_prober_last.pt"
    metrics_path = output / "metrics.json"
    history_path = output / "history.jsonl"
    best_val_mae = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    checkpoint_common = {
        "model": "roi_rollout_ttc_prober",
        "jepa_checkpoint": jepa_metadata,
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
        "cache_path": str(cache_path),
        "seed": seed,
        "input_dim": int(features.shape[1]),
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "rollout_token_summary": rollout_token_summary,
        "rollout_include_context": rollout_include_context,
        "rollout_feature_mode": rollout_feature_mode,
        "physics_prior": physics_prior,
        "ridge_alpha": ridge_alpha,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "roi_feature_mean": roi_mean.tolist(),
        "roi_feature_std": roi_std.tolist(),
        "prior_beta": prior_beta.tolist() if prior_beta is not None else [],
        "roi_feature_names": list(ROI_EVENT_FEATURE_NAMES),
    }

    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, epochs + 1):
            train_loss = _train_one_epoch(
                prober,
                train_loader,
                optimizer,
                loss_fn,
                device=device,
                scaler=scaler,
            )
            val_true, val_pred, val_seconds = _evaluate_prober(
                prober,
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
                        **checkpoint_common,
                        "epoch": epoch,
                        "prober_state_dict": prober.state_dict(),
                    },
                    best_path,
                )

    torch.save(
        {
            **checkpoint_common,
            "epoch": epochs,
            "prober_state_dict": prober.state_dict(),
        },
        last_path,
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    prober.load_state_dict(checkpoint["prober_state_dict"])

    split_results: dict[str, Any] = {}
    predictions: dict[str, list[float]] = {}
    targets: dict[str, list[float]] = {}
    for split_name in evaluation_splits:
        indices = split_indices[split_name]
        dataset = datasets[split_name]
        y_true, y_pred, seconds = _evaluate_prober(
            prober,
            dataset,
            device=device,
            batch_size=batch_size,
        )
        split_results[split_name] = {
            "label_count": int(label_count_by_split.get(split_name, 0)),
            "matched_count": int(indices.shape[0]),
            "unmatched_count": int(
                max(label_count_by_split.get(split_name, 0) - indices.shape[0], 0)
            ),
            "metrics": regression_metrics(y_true, y_pred),
            "prior_metrics": _prior_metrics(y_log, prior_log, indices),
            "seconds": seconds,
        }
        predictions[split_name] = y_pred.astype(float).tolist()
        targets[split_name] = y_true.astype(float).tolist()

    summary: dict[str, Any] = {
        "model": "roi_rollout_ttc_prober",
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
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
        "jepa_checkpoint": jepa_metadata,
        "context_ms": context_ms,
        "max_cache_slop_ms": max_cache_slop_ms,
        "rollout_token_summary": rollout_token_summary,
        "rollout_include_context": rollout_include_context,
        "rollout_feature_mode": rollout_feature_mode,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "physics_prior": physics_prior,
        "ridge_alpha": ridge_alpha,
        "train_splits": list(train_splits),
        "validation_splits": list(validation_splits),
        "evaluation_splits": list(evaluation_splits),
        "loaded_roi_rows": int(len(all_rows)),
        "matched_roi_rows": int(len(matched_rows)),
        "unmatched_roi_rows": int(len(unmatched_positions)),
        "train_count": int(train_idx.size),
        "validation_count": int(val_idx.size),
        "rollout_feature_dim": int(rollout_features.shape[1]),
        "roi_feature_dim": int(roi_features.shape[1]),
        "input_dim": int(features.shape[1]),
        "best_epoch": best_epoch,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
        "elapsed_seconds": time.perf_counter() - start_time,
        "last": history[-1] if history else None,
        "feature_names": {
            "rollout": f"frozen_jepa_predicted_{rollout_token_summary}_{rollout_feature_mode}",
            "roi": list(ROI_EVENT_FEATURE_NAMES),
            "prior": physics_prior,
        },
        "leakage_audit": {
            "encoder_frozen": True,
            "predictor_frozen": True,
            "jepa_checkpoint_selected_before_prober": True,
            "uses_ttc_labels_for_jepa": False,
            "uses_ttc_labels_for_prober": True,
            "uses_validation_or_test_ttc_for_prior_fit": False,
            "uses_validation_or_test_ttc_for_feature_scaling": False,
            "uses_future_events": False,
            "uses_predicted_future_latents": True,
            "uses_current_bbox": True,
            "uses_future_bboxes": False,
            "uses_future_navigation": False,
            "roi_event_window": "[timestamp - context_ms, timestamp] only",
            "test_evaluated_only_if_requested": "test" in evaluation_splits,
        },
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


def evaluate_roi_latent_ttc_prober_checkpoint(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    cache_path: str | Path,
    prober_checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    context_ms: int = 100,
    max_cache_slop_ms: int = 12,
    batch_size: int = 64,
    device_name: str = "auto",
    model_name: str | None = None,
    evaluation_splits: tuple[str, ...] = ("test",),
) -> dict[str, Any]:
    """Evaluate a saved detection-assisted ROI latent prober without retraining."""

    if not evaluation_splits:
        msg = "evaluation_splits must be non-empty."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)
    if context_ms <= 0 or max_cache_slop_ms < 0:
        msg = "context_ms must be positive and max_cache_slop_ms must be non-negative."
        raise ValueError(msg)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    checkpoint = torch.load(prober_checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model") != "roi_latent_ttc_prober":
        msg = f"Checkpoint {prober_checkpoint_path} is not a roi_latent_ttc_prober checkpoint."
        raise ValueError(msg)

    all_rows, row_splits, label_count_by_split = _load_roi_rows(
        manifest_path=manifest_path,
        split_path=split_path,
        split_names=evaluation_splits,
        context_ms=context_ms,
    )
    if not all_rows:
        msg = "No bbox/ROI rows could be loaded for the requested evaluation splits."
        raise ValueError(msg)

    cache = np.load(cache_path, allow_pickle=False)
    validate_voxel_cache(cache)
    x = cache["x"]
    cache_sequence_id = cache["sequence_id"].astype(str)
    cache_timestamp_us = cache["timestamp_us"].astype(np.int64)
    cache_indices, unmatched_positions = _match_cache_indices(
        cache_sequence_id=cache_sequence_id,
        cache_timestamp_us=cache_timestamp_us,
        rows=all_rows,
        max_slop_us=int(max_cache_slop_ms * 1000),
    )
    unmatched = set(unmatched_positions)
    matched_rows = [row for idx, row in enumerate(all_rows) if idx not in unmatched]
    matched_splits = np.array(
        [split_name for idx, split_name in enumerate(row_splits.tolist()) if idx not in unmatched],
        dtype=str,
    )
    if not matched_rows:
        msg = "No bbox/ROI rows matched cache windows; increase max_cache_slop_ms."
        raise ValueError(msg)
    split_indices = {
        split_name: _split_indices(matched_splits, split_name)
        for split_name in set(matched_splits.tolist())
    }
    missing_eval_splits = [
        split_name
        for split_name in evaluation_splits
        if split_name not in split_indices or split_indices[split_name].size == 0
    ]
    if missing_eval_splits:
        msg = f"Requested evaluation splits have no matched ROI rows: {missing_eval_splits}."
        raise ValueError(msg)

    encoder_info = checkpoint.get("encoder_checkpoint")
    if not isinstance(encoder_info, dict) or "path" not in encoder_info:
        msg = "ROI prober checkpoint does not record an encoder_checkpoint.path."
        raise ValueError(msg)
    token_summary = str(checkpoint.get("token_summary", "mean-std"))
    encoder, encoder_metadata = _load_frozen_encoder(
        encoder_info["path"],
        in_channels=int(x.shape[1]),
        device=device,
        model_name=model_name,
    )
    all_latents = _extract_latent_features(
        encoder,
        x,
        batch_size=batch_size,
        device=device,
        token_summary=token_summary,
    )
    latent_features = all_latents[cache_indices]
    roi_features = np.array([row.features for row in matched_rows], dtype=np.float32)
    roi_mean = np.asarray(checkpoint["roi_feature_mean"], dtype=np.float32)
    roi_std = np.asarray(checkpoint["roi_feature_std"], dtype=np.float32)
    roi_scaled = ((roi_features - roi_mean) / roi_std).astype(np.float32)
    y_ttc = np.array([row.ttc_seconds for row in matched_rows], dtype=np.float32)
    y_log = np.log(np.clip(y_ttc, 1e-4, None)).astype(np.float32)

    physics_prior = str(checkpoint.get("physics_prior", "ridge"))
    prior_beta = np.asarray(checkpoint.get("prior_beta", []), dtype=np.float64)
    if physics_prior == "ridge":
        if prior_beta.size == 0:
            msg = "Ridge ROI prober checkpoint is missing prior_beta."
            raise ValueError(msg)
        prior_log = _predict_ridge(roi_scaled.astype(np.float64), prior_beta).astype(np.float32)
    elif physics_prior == "none":
        prior_log = np.zeros_like(y_log, dtype=np.float32)
    else:
        msg = f"Unsupported physics_prior in checkpoint: {physics_prior!r}."
        raise ValueError(msg)

    raw_features = np.concatenate(
        [
            latent_features,
            roi_scaled,
            prior_log[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    features = ((raw_features - feature_mean) / feature_std).astype(np.float32)
    input_dim = int(checkpoint.get("input_dim", features.shape[1]))
    if features.shape[1] != input_dim:
        msg = (
            f"Feature dimension mismatch: built {features.shape[1]}, "
            f"checkpoint expects {input_dim}."
        )
        raise ValueError(msg)

    prober = LatentResidualTTCProber(
        input_dim,
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        dropout=float(checkpoint.get("dropout", 0.05)),
        residual=physics_prior != "none",
    ).to(device)
    prober.load_state_dict(checkpoint["prober_state_dict"])

    split_results: dict[str, Any] = {}
    predictions: dict[str, list[float]] = {}
    targets: dict[str, list[float]] = {}
    for split_name in evaluation_splits:
        indices = split_indices[split_name]
        dataset = LatentProberDataset(features, prior_log, y_log, indices)
        y_true, y_pred, seconds = _evaluate_prober(
            prober,
            dataset,
            device=device,
            batch_size=batch_size,
        )
        split_results[split_name] = {
            "label_count": int(label_count_by_split.get(split_name, 0)),
            "matched_count": int(indices.shape[0]),
            "unmatched_count": int(
                max(label_count_by_split.get(split_name, 0) - indices.shape[0], 0)
            ),
            "metrics": regression_metrics(y_true, y_pred),
            "prior_metrics": _prior_metrics(y_log, prior_log, indices),
            "seconds": seconds,
        }
        predictions[split_name] = y_pred.astype(float).tolist()
        targets[split_name] = y_true.astype(float).tolist()

    summary: dict[str, Any] = {
        "model": "roi_latent_ttc_prober",
        "checkpoint": Path(prober_checkpoint_path).as_posix(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_seed": checkpoint.get("seed"),
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
        "cache": str(cache_path),
        "checkpoint_cache": checkpoint.get("cache_path"),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": batch_size,
        "encoder_checkpoint": encoder_metadata,
        "context_ms": context_ms,
        "max_cache_slop_ms": max_cache_slop_ms,
        "token_summary": token_summary,
        "physics_prior": physics_prior,
        "evaluation_splits": list(evaluation_splits),
        "loaded_roi_rows": int(len(all_rows)),
        "matched_roi_rows": int(len(matched_rows)),
        "unmatched_roi_rows": int(len(unmatched_positions)),
        "latent_feature_dim": int(latent_features.shape[1]),
        "roi_feature_dim": int(roi_features.shape[1]),
        "input_dim": input_dim,
        "leakage_audit": {
            "encoder_frozen": True,
            "prober_checkpoint_frozen": True,
            "uses_ttc_labels_for_encoder": False,
            "uses_ttc_labels_for_prober_training": True,
            "uses_validation_or_test_ttc_for_feature_scaling": False,
            "uses_validation_or_test_ttc_for_prior_fit": False,
            "uses_future_events": False,
            "uses_current_bbox": True,
            "uses_future_bboxes": False,
            "uses_future_navigation": False,
            "roi_event_window": "[timestamp - context_ms, timestamp] only",
            "retrained_during_evaluation": False,
        },
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
        np.savez(ensure_parent(output.with_suffix(".predictions.npz")), **prediction_arrays)
    return summary


def evaluate_roi_rollout_ttc_prober_checkpoint(
    *,
    manifest_path: str | Path,
    split_path: str | Path,
    cache_path: str | Path,
    prober_checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    context_ms: int = 100,
    max_cache_slop_ms: int = 12,
    batch_size: int = 64,
    device_name: str = "auto",
    model_name: str | None = None,
    evaluation_splits: tuple[str, ...] = ("test",),
) -> dict[str, Any]:
    """Evaluate a saved ROI rollout prober without retraining."""

    if not evaluation_splits:
        msg = "evaluation_splits must be non-empty."
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)
    if context_ms <= 0 or max_cache_slop_ms < 0:
        msg = "context_ms must be positive and max_cache_slop_ms must be non-negative."
        raise ValueError(msg)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    checkpoint = torch.load(prober_checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model") != "roi_rollout_ttc_prober":
        msg = f"Checkpoint {prober_checkpoint_path} is not a roi_rollout_ttc_prober checkpoint."
        raise ValueError(msg)

    all_rows, row_splits, label_count_by_split = _load_roi_rows(
        manifest_path=manifest_path,
        split_path=split_path,
        split_names=evaluation_splits,
        context_ms=context_ms,
    )
    if not all_rows:
        msg = "No bbox/ROI rows could be loaded for the requested evaluation splits."
        raise ValueError(msg)

    cache = np.load(cache_path, allow_pickle=False)
    validate_voxel_cache(cache)
    x = cache["x"]
    bins = int(cache["bins"]) if "bins" in cache.files else int(x.shape[1] // 2)
    metadata_channels = _cache_bool(cache, "metadata_channels")
    navigation_channels = _cache_bool(cache, "navigation_channels")
    navigation_feature_names = (
        _cache_string_tuple(cache, "navigation_feature_names", NAVIGATION_FEATURE_NAMES)
        if navigation_channels
        else ()
    )
    navigation_feature_count = len(navigation_feature_names)
    cache_sequence_id = cache["sequence_id"].astype(str)
    cache_timestamp_us = cache["timestamp_us"].astype(np.int64)
    cache_indices, unmatched_positions = _match_cache_indices(
        cache_sequence_id=cache_sequence_id,
        cache_timestamp_us=cache_timestamp_us,
        rows=all_rows,
        max_slop_us=int(max_cache_slop_ms * 1000),
    )
    unmatched = set(unmatched_positions)
    matched_rows = [row for idx, row in enumerate(all_rows) if idx not in unmatched]
    matched_splits = np.array(
        [split_name for idx, split_name in enumerate(row_splits.tolist()) if idx not in unmatched],
        dtype=str,
    )
    if not matched_rows:
        msg = "No bbox/ROI rows matched cache windows; increase max_cache_slop_ms."
        raise ValueError(msg)
    split_indices = {
        split_name: _split_indices(matched_splits, split_name)
        for split_name in set(matched_splits.tolist())
    }
    missing_eval_splits = [
        split_name
        for split_name in evaluation_splits
        if split_name not in split_indices or split_indices[split_name].size == 0
    ]
    if missing_eval_splits:
        msg = f"Requested evaluation splits have no matched ROI rows: {missing_eval_splits}."
        raise ValueError(msg)

    jepa_info = checkpoint.get("jepa_checkpoint")
    if not isinstance(jepa_info, dict) or "path" not in jepa_info:
        msg = "ROI rollout prober checkpoint does not record a jepa_checkpoint.path."
        raise ValueError(msg)
    encoder, predictor, jepa_checkpoint, jepa_metadata = _load_frozen_jepa_rollout_components(
        jepa_info["path"],
        in_channels=int(x.shape[1]),
        device=device,
        model_name=model_name,
    )
    horizons_ms = tuple(int(value) for value in jepa_metadata["temporal_horizons_ms"])
    action_feature_dim = int(jepa_metadata["action_feature_dim"])
    action_feature_mean = (
        torch.tensor(jepa_checkpoint.get("action_feature_mean", []), dtype=torch.float32)
        if action_feature_dim
        else None
    )
    action_feature_std = (
        torch.tensor(jepa_checkpoint.get("action_feature_std", []), dtype=torch.float32)
        if action_feature_dim
        else None
    )
    rollout_token_summary = str(checkpoint.get("rollout_token_summary", "mean-std"))
    rollout_include_context = bool(checkpoint.get("rollout_include_context", True))
    rollout_feature_mode = str(checkpoint.get("rollout_feature_mode", "flat"))
    if rollout_feature_mode not in {"flat", "dynamics"}:
        msg = f"Unsupported rollout_feature_mode in checkpoint: {rollout_feature_mode!r}."
        raise ValueError(msg)
    all_rollout_features = _extract_predicted_rollout_features(
        encoder=encoder,
        predictor=predictor,
        x=x,
        batch_size=batch_size,
        device=device,
        token_summary=rollout_token_summary,
        horizons_ms=horizons_ms,
        bins=bins,
        metadata_channels=metadata_channels,
        navigation_feature_count=navigation_feature_count,
        action_feature_mean=action_feature_mean,
        action_feature_std=action_feature_std,
        action_feature_dim=action_feature_dim,
        include_context_latent=rollout_include_context,
        feature_mode=rollout_feature_mode,
        deep_supervision_layers=tuple(jepa_metadata["deep_supervision_layers"]),
    )
    rollout_features = all_rollout_features[cache_indices]
    roi_features = np.array([row.features for row in matched_rows], dtype=np.float32)
    roi_mean = np.asarray(checkpoint["roi_feature_mean"], dtype=np.float32)
    roi_std = np.asarray(checkpoint["roi_feature_std"], dtype=np.float32)
    roi_scaled = ((roi_features - roi_mean) / roi_std).astype(np.float32)
    y_ttc = np.array([row.ttc_seconds for row in matched_rows], dtype=np.float32)
    y_log = np.log(np.clip(y_ttc, 1e-4, None)).astype(np.float32)

    physics_prior = str(checkpoint.get("physics_prior", "ridge"))
    prior_beta = np.asarray(checkpoint.get("prior_beta", []), dtype=np.float64)
    if physics_prior == "ridge":
        if prior_beta.size == 0:
            msg = "Ridge ROI rollout prober checkpoint is missing prior_beta."
            raise ValueError(msg)
        prior_log = _predict_ridge(roi_scaled.astype(np.float64), prior_beta).astype(np.float32)
    elif physics_prior == "none":
        prior_log = np.zeros_like(y_log, dtype=np.float32)
    else:
        msg = f"Unsupported physics_prior in checkpoint: {physics_prior!r}."
        raise ValueError(msg)

    raw_features = np.concatenate(
        [
            rollout_features,
            roi_scaled,
            prior_log[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
    features = ((raw_features - feature_mean) / feature_std).astype(np.float32)
    input_dim = int(checkpoint.get("input_dim", features.shape[1]))
    if features.shape[1] != input_dim:
        msg = (
            f"Feature dimension mismatch: built {features.shape[1]}, "
            f"checkpoint expects {input_dim}."
        )
        raise ValueError(msg)

    prober = LatentResidualTTCProber(
        input_dim,
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        dropout=float(checkpoint.get("dropout", 0.10)),
        residual=physics_prior != "none",
    ).to(device)
    prober.load_state_dict(checkpoint["prober_state_dict"])

    split_results: dict[str, Any] = {}
    predictions: dict[str, list[float]] = {}
    targets: dict[str, list[float]] = {}
    for split_name in evaluation_splits:
        indices = split_indices[split_name]
        dataset = LatentProberDataset(features, prior_log, y_log, indices)
        y_true, y_pred, seconds = _evaluate_prober(
            prober,
            dataset,
            device=device,
            batch_size=batch_size,
        )
        split_results[split_name] = {
            "label_count": int(label_count_by_split.get(split_name, 0)),
            "matched_count": int(indices.shape[0]),
            "unmatched_count": int(
                max(label_count_by_split.get(split_name, 0) - indices.shape[0], 0)
            ),
            "metrics": regression_metrics(y_true, y_pred),
            "prior_metrics": _prior_metrics(y_log, prior_log, indices),
            "seconds": seconds,
        }
        predictions[split_name] = y_pred.astype(float).tolist()
        targets[split_name] = y_true.astype(float).tolist()

    summary: dict[str, Any] = {
        "model": "roi_rollout_ttc_prober",
        "checkpoint": Path(prober_checkpoint_path).as_posix(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_seed": checkpoint.get("seed"),
        "manifest": Path(manifest_path).as_posix(),
        "split": Path(split_path).as_posix(),
        "cache": str(cache_path),
        "checkpoint_cache": checkpoint.get("cache_path"),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": batch_size,
        "jepa_checkpoint": jepa_metadata,
        "context_ms": context_ms,
        "max_cache_slop_ms": max_cache_slop_ms,
        "rollout_token_summary": rollout_token_summary,
        "rollout_include_context": rollout_include_context,
        "rollout_feature_mode": rollout_feature_mode,
        "physics_prior": physics_prior,
        "evaluation_splits": list(evaluation_splits),
        "loaded_roi_rows": int(len(all_rows)),
        "matched_roi_rows": int(len(matched_rows)),
        "unmatched_roi_rows": int(len(unmatched_positions)),
        "rollout_feature_dim": int(rollout_features.shape[1]),
        "roi_feature_dim": int(roi_features.shape[1]),
        "input_dim": input_dim,
        "leakage_audit": {
            "encoder_frozen": True,
            "predictor_frozen": True,
            "prober_checkpoint_frozen": True,
            "uses_ttc_labels_for_jepa": False,
            "uses_ttc_labels_for_prober_training": True,
            "uses_validation_or_test_ttc_for_feature_scaling": False,
            "uses_validation_or_test_ttc_for_prior_fit": False,
            "uses_future_events": False,
            "uses_predicted_future_latents": True,
            "uses_current_bbox": True,
            "uses_future_bboxes": False,
            "uses_future_navigation": False,
            "roi_event_window": "[timestamp - context_ms, timestamp] only",
            "retrained_during_evaluation": False,
        },
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
        np.savez(ensure_parent(output.with_suffix(".predictions.npz")), **prediction_arrays)
    return summary
