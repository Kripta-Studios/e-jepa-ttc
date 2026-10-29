"""Train and zero-shot evaluate the leakage-safe official-label LHR object-JEPA."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Iterator, Sized
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset
from e_jepa_ttc.data.garl_input_contract import validate_cache_manifest_input_schema
from e_jepa_ttc.data.garlttc_lhr_cache import (
    FORBIDDEN_MODEL_INPUT_KEYS,
    GarlTTCLHRCacheDataset,
    observable_motion_from_boxes_torch,
)
from e_jepa_ttc.evaluation.garl_ttc_protocol import sequence_macro_signed_metrics
from e_jepa_ttc.evaluation.object_ttc import object_ttc_metrics
from e_jepa_ttc.models.eap_lhr_jepa_ttc import (
    EAPLHRJEPATTC,
    EAPLHRJEPATTCConfig,
    EAPLHRJEPATTCOutput,
)
from e_jepa_ttc.reproducibility import cuda_is_usable, resolve_device
from e_jepa_ttc.utils.io import read_structured, write_structured


@dataclass(frozen=True)
class EAPLHRTrainerConfig:
    epochs: int = 30
    batch_size: int = 24
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 8
    precision: str = "bf16"
    ema_momentum: float = 0.996
    height_loss_weight: float = 0.5
    ratio_loss_weight: float = 1.0
    ttc_loss_weight: float = 0.25
    jepa_loss_weight: float = 0.1
    geometry_loss_weight: float = 0.1
    category_loss_weight: float = 0.05
    foreground_loss_weight: float = 0.0
    early_stopping_patience: int = 5
    minimum_epochs: int = 8
    seed: int = 42
    balanced_sampling: bool = True
    max_train_samples: int | None = None
    max_validation_samples: int | None = None

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size, self.num_workers + 1) <= 0:
            raise ValueError("Trainer integer controls must be positive.")
        for name, value in (
            ("max_train_samples", self.max_train_samples),
            ("max_validation_samples", self.max_validation_samples),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16.")
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("ema_momentum must lie in [0,1).")
        weights = (
            self.height_loss_weight,
            self.ratio_loss_weight,
            self.ttc_loss_weight,
            self.jepa_loss_weight,
            self.geometry_loss_weight,
            self.category_loss_weight,
            self.foreground_loss_weight,
        )
        if min(weights) < 0:
            raise ValueError("Loss weights must be non-negative.")


def _device(name: str) -> torch.device:
    return resolve_device(name)


def _autocast(device: torch.device, precision: str) -> torch.autocast:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


def _tensor(
    batch: dict[str, Any],
    key: str,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must collate to a tensor.")
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _assert_model_inputs_are_causal(keys: set[str]) -> None:
    leaked = sorted(keys & FORBIDDEN_MODEL_INPUT_KEYS)
    if leaked:
        raise RuntimeError(f"Privileged supervision attempted as model input: {leaked}")


def _model_inputs(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    """Return only fields allowed to enter the estimator."""

    used_keys = {"garl_event_roi", "garl_delta_t_s"}
    full_key = "full_frame_events" if "full_frame_events" in batch else "context_events"
    used_keys.add(full_key)
    if "observable_motion" in batch:
        motion = _tensor(batch, "observable_motion", device, dtype=torch.float32)
        used_keys.add("observable_motion")
    else:
        boxes = _tensor(batch, "context_boxes", device, dtype=torch.float32)
        delta = _tensor(batch, "garl_delta_t_s", device, dtype=torch.float32)
        motion = observable_motion_from_boxes_torch(boxes, delta)
        used_keys.add("context_boxes")
    jepa_context_motion = (
        _tensor(batch, "jepa_context_motion", device, dtype=torch.float32)
        if "jepa_context_motion" in batch
        else torch.zeros_like(motion)
    )
    rgb = None
    if "garl_rgb_pair" in batch:
        rgb = _tensor(batch, "garl_rgb_pair", device, dtype=torch.float32)
        used_keys.add("garl_rgb_pair")
    _assert_model_inputs_are_causal(used_keys)
    return {
        "full_frame_events": _tensor(batch, full_key, device, dtype=torch.float32),
        "event_roi_pair": _tensor(batch, "garl_event_roi", device, dtype=torch.float32),
        "delta_t_s": _tensor(batch, "garl_delta_t_s", device, dtype=torch.float32),
        "observable_motion": motion,
        "jepa_context_motion": jepa_context_motion,
        "rgb_pair": rgb,
    }


def _forward(
    model: EAPLHRJEPATTC,
    batch: dict[str, Any],
    device: torch.device,
) -> EAPLHRJEPATTCOutput:
    return model(**_model_inputs(batch, device))


def _signed_log1p(value: torch.Tensor) -> torch.Tensor:
    return value.sign() * torch.log1p(value.abs())


def _losses(
    output: EAPLHRJEPATTCOutput,
    batch: dict[str, Any],
    device: torch.device,
    config: EAPLHRTrainerConfig,
) -> dict[str, torch.Tensor]:
    """Compute supervision losses without feeding any target back into the model."""

    truth = _tensor(batch, "ttc_s", device, dtype=torch.float32).reshape(-1)
    heights = _tensor(batch, "garl_visible_heights_px", device, dtype=torch.float32)
    if heights.ndim != 2 or heights.shape[1] != 2:
        raise ValueError("garl_visible_heights_px must have shape [B,2].")
    target_ratio = heights[:, 0] / heights[:, 1].clamp_min(1e-6)
    valid_ratio = torch.isfinite(target_ratio) & (target_ratio > 0.0)

    height_loss = functional.smooth_l1_loss(
        output.predicted_heights.clamp_min(1e-3).log(),
        heights.clamp_min(1e-3).log(),
        beta=0.1,
    )
    ratio_loss = truth.new_zeros(())
    if valid_ratio.any():
        ratio_loss = functional.smooth_l1_loss(
            output.predicted_height_ratio[valid_ratio].clamp_min(1e-4).log(),
            target_ratio[valid_ratio].log(),
            beta=0.05,
        )
    ttc_loss = functional.smooth_l1_loss(
        _signed_log1p(output.ttc_seconds),
        _signed_log1p(truth),
        beta=0.05,
    )
    jepa_pair_valid = (
        _tensor(batch, "jepa_pair_valid", device).bool().reshape(-1)
        if "jepa_pair_valid" in batch
        else torch.ones_like(truth, dtype=torch.bool)
    )
    jepa_loss = truth.new_zeros(())
    if jepa_pair_valid.any():
        jepa_loss = functional.smooth_l1_loss(
            functional.normalize(output.jepa_prediction[jepa_pair_valid].float(), dim=-1),
            functional.normalize(output.jepa_target[jepa_pair_valid].float(), dim=-1),
            beta=0.1,
        )

    geometry_target = _tensor(batch, "geometry_v2_target", device, dtype=torch.float32)
    geometry_valid = _tensor(batch, "geometry_v2_valid", device).bool()
    per_geometry = functional.smooth_l1_loss(
        output.geometry_prediction,
        geometry_target,
        beta=0.1,
        reduction="none",
    )
    geometry_loss = (per_geometry * geometry_valid).sum() / geometry_valid.sum().clamp_min(1)

    category_loss = truth.new_zeros(())
    if "category_index" in batch:
        category_valid = (
            _tensor(batch, "category_valid", device).bool().reshape(-1)
            if "category_valid" in batch
            else torch.ones_like(truth, dtype=torch.bool)
        )
        if category_valid.any():
            category_loss = functional.cross_entropy(
                output.category_logits[category_valid],
                _tensor(batch, "category_index", device).long().reshape(-1)[category_valid],
            )

    foreground_loss = truth.new_zeros(())
    if config.foreground_loss_weight > 0.0:
        if "garl_foreground_mask" not in batch or "foreground_valid" not in batch:
            raise ValueError(
                "Foreground loss requested but no official/teacher mask target is present. "
                "Weak rectangular bbox masks are intentionally forbidden in v2."
            )
        valid = _tensor(batch, "foreground_valid", device).bool().reshape(-1)
        if valid.any():
            target_mask = _tensor(batch, "garl_foreground_mask", device).long()[valid]
            foreground_loss = functional.cross_entropy(output.foreground_logits[valid], target_mask)

    raw = {
        "height": height_loss,
        "ratio": ratio_loss,
        "ttc_log": ttc_loss,
        "jepa": jepa_loss,
        "geometry": geometry_loss,
        "category": category_loss,
        "foreground": foreground_loss,
    }
    weighted = {
        "height": config.height_loss_weight * raw["height"],
        "ratio": config.ratio_loss_weight * raw["ratio"],
        "ttc_log": config.ttc_loss_weight * raw["ttc_log"],
        "jepa": config.jepa_loss_weight * raw["jepa"],
        "geometry": config.geometry_loss_weight * raw["geometry"],
        "category": config.category_loss_weight * raw["category"],
        "foreground": config.foreground_loss_weight * raw["foreground"],
    }
    weighted["total"] = torch.stack(list(weighted.values())).sum()
    for key, value in raw.items():
        weighted[f"raw_{key}"] = value
    return weighted


def _source_shard_key(dataset: Dataset[dict[str, Any]], index: int) -> str:
    """Resolve a local index to its compressed shard without loading it."""

    current: Dataset[dict[str, Any]] = dataset
    source_index = index
    while isinstance(current, Subset):
        source_index = int(current.indices[source_index])
        current = cast(Dataset[dict[str, Any]], current.dataset)
    entries = getattr(current, "entries", None)
    if isinstance(entries, list) and 0 <= source_index < len(entries):
        entry = entries[source_index]
        if isinstance(entry, tuple) and entry:
            return str(entry[0])
    return "__single_shard__"


def _balanced_sampling_metadata(
    dataset: Dataset[dict[str, Any]],
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Extract lightweight metadata once and reuse it across training epochs."""

    base: Dataset[dict[str, Any]] = dataset
    source_indices = list(range(len(cast(Sized, dataset))))
    while isinstance(base, Subset):
        source_indices = [int(base.indices[index]) for index in source_indices]
        base = cast(Dataset[dict[str, Any]], base.dataset)
    cache_key = tuple(source_indices)
    cache: dict[tuple[int, ...], tuple[list[str], list[tuple[str, str]], list[str]]] = getattr(
        base, "_e_jepa_ttc_balanced_metadata", {}
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    groups: list[str] = []
    tracks: list[tuple[str, str]] = []
    shard_keys: list[str] = []
    for local_index, source_index in enumerate(source_indices):
        row = base[source_index]
        groups.append(str(row.get("sampling_group", "unlabelled")))
        tracks.append((str(row.get("sequence_id", "")), str(row.get("track_id", ""))))
        shard_keys.append(_source_shard_key(dataset, local_index))
        del row
    cached = (groups, tracks, shard_keys)
    cache[cache_key] = cached
    base.__dict__["_e_jepa_ttc_balanced_metadata"] = cache
    return cached


class _ShardLocalWeightedBatchSampler(Sampler[list[int]]):
    """Hierarchically balance rows while keeping compressed-shard locality."""

    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dataset_length = len(cast(Sized, dataset))
        if dataset_length <= 0:
            raise ValueError("Balanced sampling requires a non-empty dataset")
        # Do not retain sample dictionaries: one sample can contain several
        # high-resolution tensors and retaining all rows recreates the full cache in RAM.
        groups, tracks, shard_keys = _balanced_sampling_metadata(dataset)

        group_tracks: dict[str, set[tuple[str, str]]] = {}
        track_counts: dict[tuple[str, tuple[str, str]], int] = {}
        for group, track in zip(groups, tracks, strict=True):
            group_tracks.setdefault(group, set()).add(track)
            key = (group, track)
            track_counts[key] = track_counts.get(key, 0) + 1
        group_count = max(len(group_tracks), 1)
        weights = [
            1.0
            / (
                group_count
                * max(len(group_tracks[group]), 1)
                * max(track_counts[(group, track)], 1)
            )
            for group, track in zip(groups, tracks, strict=True)
        ]
        generator = torch.Generator().manual_seed(seed)
        sampled = torch.multinomial(
            torch.tensor(weights, dtype=torch.float64),
            dataset_length,
            replacement=True,
            generator=generator,
        ).tolist()
        by_shard: dict[str, list[int]] = {}
        for index in sampled:
            by_shard.setdefault(shard_keys[index], []).append(index)
        shard_order = list(by_shard)
        permutation = torch.randperm(len(shard_order), generator=generator).tolist()
        self._batches = [
            indices[start : start + batch_size]
            for shard_position in permutation
            for indices in (by_shard[shard_order[shard_position]],)
            for start in range(0, len(indices), batch_size)
        ]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def _balanced_sampler(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
) -> _ShardLocalWeightedBatchSampler:
    """Balance strata/tracks/states without random access across shards."""

    return _ShardLocalWeightedBatchSampler(dataset, batch_size=batch_size, seed=seed)


def _deterministic_indices(length: int, maximum: int | None) -> list[int] | None:
    """Select a bounded prefix to preserve locality in shard-cached datasets."""

    if maximum is None or maximum >= length:
        return None
    if maximum <= 0:
        raise ValueError("maximum must be positive when provided.")
    return list(range(maximum))


def _loader(
    dataset: Dataset[dict[str, Any]],
    config: EAPLHRTrainerConfig,
    *,
    train: bool,
    epoch: int = 0,
    indices: list[int] | None = None,
) -> DataLoader[dict[str, Any]]:
    selected_dataset: Dataset[dict[str, Any]] = (
        dataset if indices is None else Subset(dataset, indices)
    )
    batch_sampler = (
        _balanced_sampler(
            selected_dataset,
            batch_size=config.batch_size,
            seed=config.seed + epoch,
        )
        if train and config.balanced_sampling
        else None
    )
    common = {
        "num_workers": config.num_workers,
        "pin_memory": cuda_is_usable(),
        "persistent_workers": config.num_workers > 0,
    }
    if batch_sampler is not None:
        return DataLoader(selected_dataset, batch_sampler=batch_sampler, **common)
    return DataLoader(
        selected_dataset,
        batch_size=config.batch_size,
        shuffle=train,
        **common,
    )


def _evaluate(
    model: EAPLHRJEPATTC,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    truth_rows: list[np.ndarray] = []
    prediction_rows: list[np.ndarray] = []
    sequences: list[str] = []
    sample_tokens: list[str] = []
    track_ids: list[str] = []
    timestamps: list[int] = []
    categories: list[str] = []
    sampling_groups: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            output = _forward(model, batch, device)
            truth_rows.append(_tensor(batch, "ttc_s", device).reshape(-1).cpu().numpy())
            prediction_rows.append(output.ttc_seconds.cpu().numpy())
            sequence_values = batch["sequence_id"]
            if not isinstance(sequence_values, list):
                raise TypeError("sequence_id must collate to a list.")
            sequences.extend(str(value) for value in sequence_values)
            sample_tokens.extend(
                str(value) for value in batch.get("sample_token", [""] * len(sequence_values))
            )
            track_ids.extend(
                str(value) for value in batch.get("track_id", [""] * len(sequence_values))
            )
            timestamps.extend(
                int(value) for value in batch.get("timestamp_us", [0] * len(sequence_values))
            )
            categories.extend(
                str(value) for value in batch.get("category", ["unknown"] * len(sequence_values))
            )
            sampling_groups.extend(
                str(value)
                for value in batch.get("sampling_group", ["unknown"] * len(sequence_values))
            )
    truth = np.concatenate(truth_rows)
    prediction = np.concatenate(prediction_rows)
    sequence_array = np.asarray(sequences)
    metrics = object_ttc_metrics(truth, prediction)
    metrics.update(sequence_macro_signed_metrics(truth, prediction, sequence_array))
    return metrics, {
        "ttc_true": truth,
        "ttc_pred": prediction,
        "sequence_id": sequence_array,
        "sample_token": np.asarray(sample_tokens),
        "track_id": np.asarray(track_ids),
        "timestamp_us": np.asarray(timestamps, dtype=np.int64),
        "category": np.asarray(categories),
        "sampling_group": np.asarray(sampling_groups),
    }


def _checkpoint_payload(
    *,
    model: EAPLHRJEPATTC,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: EAPLHRJEPATTCConfig,
    trainer_config: EAPLHRTrainerConfig,
    manifest: dict[str, Any],
    scaler: Any | None = None,  # noqa: ANN401
    stale: int = 0,
    best_score: float = math.inf,
    best_epoch: int = 0,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if cuda_is_usable() else None,
        "early_stopping_stale": stale,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "epoch": epoch,
        "selected_by": "validation_sequence_macro_paper_MiD_overall_signed_v1",
        "model_config": asdict(model_config),
        "trainer_config": asdict(trainer_config),
        "uses_reconstructed_public_eap_ttc": False,
        "uses_official_garl_ttc_labels": True,
        "official_garlttc_annotations_sha256": manifest["garlttc_annotations_sha256"],
        "official_garlttc_data_sha256": manifest["garlttc_data_sha256"],
        "official_garlttc_join_keys_sha256": manifest["garlttc_join_keys_sha256"],
        "ttc_head_transferable_to_evttc": True,
        "privileged_inputs_forbidden": sorted(FORBIDDEN_MODEL_INPUT_KEYS),
        "jepa_predictor_is_strictly_causal": True,
        "hierarchical_track_balancing": trainer_config.balanced_sampling,
    }


def train_eap_lhr_jepa_ttc(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    geo_checkpoint: str | Path | None,
    model_config: EAPLHRJEPATTCConfig,
    trainer_config: EAPLHRTrainerConfig,
    device_name: str = "auto",
    resume: bool = False,
) -> dict[str, Any]:
    """Train on official GarlTTC labels while preserving the complete TTC head."""

    random.seed(trainer_config.seed)
    torch.manual_seed(trainer_config.seed)
    np.random.seed(trainer_config.seed)
    device = _device(device_name)
    manifest = read_structured(manifest_path)
    if manifest.get("uses_official_garl_ttc_labels") is not True:
        raise ValueError("Refusing to train LHR-v2 without official GarlTTC labels.")
    if manifest.get("no_label_fallback") is not True:
        raise ValueError("Cache must explicitly prohibit reconstructed-label fallback.")

    train_dataset = GarlTTCLHRCacheDataset(manifest_path, splits=("train",))
    validation_dataset = GarlTTCLHRCacheDataset(manifest_path, splits=("validation",))
    train_indices = _deterministic_indices(len(train_dataset), trainer_config.max_train_samples)
    validation_indices = _deterministic_indices(
        len(validation_dataset), trainer_config.max_validation_samples
    )
    validation_loader = _loader(
        validation_dataset,
        trainer_config,
        train=False,
        indices=validation_indices,
    )
    model = EAPLHRJEPATTC(model_config).to(device)
    if geo_checkpoint is not None:
        checkpoint = torch.load(geo_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise TypeError("eAP-Geo checkpoint must be a mapping.")
        model.load_geo_encoder(checkpoint)

    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("target_roi_encoder.")
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
    )
    scaler = torch.amp.GradScaler(  # type: ignore[reportPrivateImportUsage]
        "cuda", enabled=(device.type == "cuda" and trainer_config.precision == "fp16")
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.jsonl"
    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
    stale = 0
    if resume and (output / "last.pt").exists():
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if saved.get("scaler_state_dict") is not None:
            scaler.load_state_dict(saved["scaler_state_dict"])
        if saved.get("python_random_state") is not None:
            random.setstate(saved["python_random_state"])
        if saved.get("numpy_random_state") is not None:
            np.random.set_state(saved["numpy_random_state"])
        if saved.get("torch_rng_state") is not None:
            torch.set_rng_state(saved["torch_rng_state"])
        if cuda_is_usable() and saved.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        stale = int(saved.get("early_stopping_stale", 0))
        best_score = float(saved.get("best_score", best_score))
        best_epoch = int(saved.get("best_epoch", best_epoch))
        start_epoch = int(saved["epoch"]) + 1
        if (output / "best.pt").exists():
            best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            best_score = float(best.get("validation_score", math.inf))
            best_epoch = int(best.get("epoch", 0))

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(start_epoch, trainer_config.epochs + 1):
        train_loader = _loader(
            train_dataset,
            trainer_config,
            train=True,
            epoch=epoch,
            indices=train_indices,
        )
        model.train()
        model.target_roi_encoder.eval()
        train_rows: list[dict[str, float]] = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, trainer_config.precision):
                prediction = _forward(model, batch, device)
                losses = _losses(prediction, batch, device, trainer_config)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            model.update_target(trainer_config.ema_momentum)
            train_rows.append({key: float(value.detach().cpu()) for key, value in losses.items()})
        validation, _arrays = _evaluate(model, validation_loader, device)
        score = float(validation["sequence_macro_paper_MiD_overall"])
        row = {
            "epoch": epoch,
            "train": {
                key: float(np.mean([values[key] for values in train_rows])) for key in train_rows[0]
            },
            "validation": validation,
        }
        history.append(row)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            model_config=model_config,
            trainer_config=trainer_config,
            manifest=manifest,
            scaler=scaler,
            stale=(0 if score < best_score else stale + 1),
            best_score=min(best_score, score),
            best_epoch=(epoch if score < best_score else best_epoch),
        )
        payload["validation_score"] = score
        torch.save(payload, output / "last.pt")
        if score < best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(payload, output / "best.pt")
        else:
            stale += 1
        if (
            epoch >= trainer_config.minimum_epochs
            and stale >= trainer_config.early_stopping_patience
        ):
            break

    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    torch.save(
        {
            "model_state_dict": model.inference_state_dict(),
            "model_config": asdict(model_config),
            "role": "weights_only",
            "selected_by": "validation_sequence_macro_paper_MiD_overall_signed_v1",
            "uses_reconstructed_public_eap_ttc": False,
            "uses_official_garl_ttc_labels": True,
            "official_garlttc_annotations_sha256": manifest["garlttc_annotations_sha256"],
            "official_garlttc_data_sha256": manifest["garlttc_data_sha256"],
            "official_garlttc_join_keys_sha256": manifest["garlttc_join_keys_sha256"],
            "transferred_components": [
                "full_encoder",
                "full_projection",
                "roi_encoder",
                "motion_encoder",
                "delta_encoder",
                "fusion",
                "height_head",
                "ttc_residual_head",
                "geometry_head",
                "category_head",
            ],
            "discarded_training_only_components": [
                "target_roi_encoder",
                "jepa_predictor",
            ],
            "ttc_head_transferable_to_evttc": True,
            "same_head_required_for_zero_shot": True,
            "privileged_inputs_forbidden": sorted(FORBIDDEN_MODEL_INPUT_KEYS),
            "jepa_predictor_is_strictly_causal": True,
            "hierarchical_track_balancing": trainer_config.balanced_sampling,
        },
        output / "weights_only.pt",
    )
    summary = {
        "artifact_type": "eap_lhr_object_jepa_ttc_training_v3",
        "best_epoch": best_epoch,
        "best_validation_sequence_macro_paper_MiD_overall": best_score,
        "epochs_completed_this_invocation": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "model_config": asdict(model_config),
        "trainer_config": asdict(trainer_config),
        "label_provenance": "official_garlttc_annotations_train_parquet",
        "uses_official_garl_ttc_labels": True,
        "uses_reconstructed_public_eap_ttc": False,
        "no_privileged_model_inputs": True,
        "weights_only_checkpoint": (output / "weights_only.pt").as_posix(),
    }
    write_structured(output / "summary.json", summary)
    return summary


def evaluate_eap_lhr_zero_shot(
    *,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    splits: tuple[str, ...],
    output_path: str | Path,
    batch_size: int = 24,
    num_workers: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate the same TTC estimator without any target-dataset updates."""

    manifest = read_structured(manifest_path)
    validate_cache_manifest_input_schema(manifest)
    device = _device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("uses_official_garl_ttc_labels") is not True:
        raise ValueError("Zero-shot checkpoint was not trained with official GarlTTC labels.")
    model_config = EAPLHRJEPATTCConfig(**checkpoint["model_config"])
    model = EAPLHRJEPATTC(model_config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {
        name for name in missing if name.startswith(("target_roi_encoder.", "jepa_predictor."))
    }
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(
            f"Incompatible zero-shot checkpoint: missing={missing}, unexpected={unexpected}"
        )
    model.target_roi_encoder.load_state_dict(model.roi_encoder.state_dict())
    target_manifest = read_structured(manifest_path)
    if target_manifest.get("uses_official_garl_ttc_labels") is True:
        dataset = GarlTTCLHRCacheDataset(manifest_path, splits=splits)
    else:
        dataset = EAPObjectCacheDataset(manifest_path, splits=splits)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=cuda_is_usable(),
    )
    metrics, arrays = _evaluate(model, loader, device)
    predictions = []
    for index in range(arrays["ttc_true"].shape[0]):
        truth = float(arrays["ttc_true"][index])
        predicted = float(arrays["ttc_pred"][index])
        predictions.append(
            {
                "sequence_id": str(arrays["sequence_id"][index]),
                "sample_token": str(arrays["sample_token"][index]),
                "track_id": str(arrays["track_id"][index]),
                "timestamp_us": int(arrays["timestamp_us"][index]),
                "category": str(arrays["category"][index]),
                "sampling_group": str(arrays["sampling_group"][index]),
                "target_ttc_s": truth,
                "predicted_ttc_s": predicted,
                "absolute_error_s": abs(predicted - truth),
            }
        )
    result = {
        "artifact_type": "eap_lhr_object_jepa_ttc_zero_shot_v3",
        "training_updates_on_target_dataset": 0,
        "same_ttc_head_as_eap_training": True,
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "splits": list(splits),
        "metrics": metrics,
        "sample_count": int(arrays["ttc_true"].shape[0]),
        "no_privileged_model_inputs": True,
        "predictions": predictions,
    }
    write_structured(output_path, result)
    return result
