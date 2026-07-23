"""Training and evaluation for Object-centric Event-JEPA and TTC heads."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset, Subset

from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset, ShardLocalSampler
from e_jepa_ttc.evaluation.bootstrap import sequence_bootstrap_interval
from e_jepa_ttc.evaluation.calibration import (
    ConformalIntervalCalibrator,
    TemperatureScaler,
    fit_conformal_interval,
    fit_temperature_scaler,
    interval_metrics,
)
from e_jepa_ttc.evaluation.object_ttc import object_ttc_metrics
from e_jepa_ttc.models.object_jepa import (
    ObjectCentricEventJEPA,
    ObjectJEPAConfig,
    ObjectTTCOutput,
    geometric_dynamics_targets,
    inverse_ttc_distribution_to_seconds,
    object_event_jepa_loss,
)
from e_jepa_ttc.utils.io import write_structured


def _hash_file(filepath: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    if not Path(filepath).exists():
        return ""
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



def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _tensor_batch(
    batch: dict[str, torch.Tensor | list[str]],
    key: str,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        msg = f"Expected tensor batch field {key!r}."
        raise TypeError(msg)
    return value.to(device=device, dtype=dtype, non_blocking=device.type == "cuda")


def _horizons_from_batch(
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> torch.Tensor:
    horizons = _tensor_batch(batch, "prediction_horizons_s", device, dtype=torch.float32)
    if horizons.ndim == 2:
        if not torch.allclose(horizons, horizons[:1].expand_as(horizons)):
            msg = "Every sample in an object-cache batch must use the same horizons."
            raise ValueError(msg)
        horizons = horizons[0]
    if horizons.ndim != 1:
        msg = "prediction_horizons_s must collate to [B,H] or [H]."
        raise ValueError(msg)
    return horizons


def _jepa_forward(
    model: ObjectCentricEventJEPA,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    *,
    use_ego_actions: bool = True,
) -> tuple[dict[str, torch.Tensor], int]:
    horizons = _horizons_from_batch(batch, device)
    context_boxes = _tensor_batch(batch, "context_boxes", device, dtype=torch.float32)
    future_boxes = _tensor_batch(batch, "future_boxes", device, dtype=torch.float32)
    context_depth = _tensor_batch(batch, "context_depth_m", device, dtype=torch.float32)
    future_depth = _tensor_batch(batch, "future_depth_m", device, dtype=torch.float32)
    output = model(
        _tensor_batch(batch, "context_events", device, dtype=torch.float32),
        context_boxes,
        _tensor_batch(batch, "context_object_mask", device).bool(),
        _tensor_batch(batch, "future_events", device, dtype=torch.float32),
        future_boxes,
        _tensor_batch(batch, "future_object_mask", device).bool(),
        horizons,
        context_sampling_boxes=_tensor_batch(
            batch,
            "context_sampling_boxes",
            device,
            dtype=torch.float32,
        ),
        future_sampling_boxes=_tensor_batch(
            batch,
            "future_sampling_boxes",
            device,
            dtype=torch.float32,
        ),
        context_ego_actions=(
            _tensor_batch(batch, "context_ego_actions", device, dtype=torch.float32)
            if use_ego_actions
            else None
        ),
        context_ego_action_mask=(
            _tensor_batch(batch, "context_ego_action_mask", device).bool()
            if use_ego_actions
            else None
        ),
        future_ego_actions=(
            _tensor_batch(batch, "future_ego_actions", device, dtype=torch.float32)
            if use_ego_actions
            else None
        ),
        future_ego_action_mask=(
            _tensor_batch(batch, "future_ego_action_mask", device).bool()
            if use_ego_actions
            else None
        ),
    )
    geometry = geometric_dynamics_targets(
        context_boxes[:, -1],
        future_boxes,
        horizons,
        context_depth_m=context_depth,
        future_depth_m=future_depth,
    )
    losses = object_event_jepa_loss(
        output,
        geometry,
        ttc_target_s=None,
        geometry_weight=1.0 if model.config.use_geometry else 0.0,
    )
    return losses, int(output.future_mask.sum().item())


def _evaluate_jepa_loss(
    model: ObjectCentricEventJEPA,
    loader: DataLoader[dict[str, torch.Tensor | list[str]]],
    device: torch.device,
    *,
    use_ego_actions: bool = True,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    denominator = 0
    with torch.no_grad():
        for batch in loader:
            losses, valid_count = _jepa_forward(
                model,
                batch,
                device,
                use_ego_actions=use_ego_actions,
            )
            denominator += valid_count
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * valid_count
    if denominator == 0:
        msg = "JEPA validation contains no valid future object targets."
        raise ValueError(msg)
    return {name: value / denominator for name, value in totals.items()}


def pretrain_object_event_jepa(
    *,
    cache_manifest_path: str | Path,
    output_dir: str | Path,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.05,
    seed: int = 42,
    device_name: str = "auto",
    embedding_dim: int = 192,
    feature_dim: int = 128,
    predictor_depth: int = 3,
    predictor_heads: int = 6,
    ema_start: float = 0.99,
    ema_end: float = 0.9999,
    use_ego_actions: bool = True,
    use_recurrence: bool = True,
    use_geometry: bool = True,
) -> dict[str, Any]:
    """Pretrain the object world model without consuming TTC labels."""

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        msg = "epochs, batch_size and learning_rate must be positive."
        raise ValueError(msg)
    _set_seed(seed)
    device = _device(device_name)
    train_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("train",))
    validation_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("validation",))
    first = train_dataset[0]
    cache_metadata = json.loads(Path(cache_manifest_path).read_text(encoding="utf-8"))
    context_events = first["context_events"]
    context_actions = first["context_ego_actions"]
    valid_event_tensor = isinstance(context_events, torch.Tensor)
    valid_action_tensor = isinstance(context_actions, torch.Tensor)
    if not valid_event_tensor or not valid_action_tensor:
        msg = "Object cache is missing tensor event/action fields."
        raise TypeError(msg)
    config = ObjectJEPAConfig(
        in_channels=int(context_events.shape[1]),
        action_dim=int(context_actions.shape[-1]),
        embedding_dim=embedding_dim,
        feature_dim=feature_dim,
        predictor_depth=predictor_depth,
        predictor_heads=predictor_heads,
        pre_cropped_events=bool(cache_metadata.get("pre_cropped_events", True)),
        use_recurrence=use_recurrence,
        use_geometry=use_geometry,
    )
    model = ObjectCentricEventJEPA(config).to(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=ShardLocalSampler(train_dataset, seed=seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "object_jepa_best.pt"
    last_path = output / "object_jepa_last.pt"
    history_path = output / "history.jsonl"
    best_validation = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    import hashlib

    run_fingerprint_payload = {
        "git_commit": _get_git_commit(),
        "protocol_version": "2.0",
        "cache_sha256": _hash_file(cache_manifest_path),
        "split_manifest_sha256": _hash_file(cache_manifest_path),
        "subset_manifest_sha256": "",
        "model_name": "object_centric_event_jepa",
        "resolved_model_config": asdict(config),
        "navigation_mode": "enabled" if use_ego_actions else "disabled",
        "label_fraction": 1.0,
        "seed": seed,
        "pretraining_checkpoint_sha256": "",
        "optimizer_config": {"learning_rate": learning_rate, "weight_decay": weight_decay},
        "training_steps": epochs,
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(run_fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    start_time = time.perf_counter()
    total_steps = max(1, epochs * len(train_loader))
    global_step = 0
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, epochs + 1):
            model.train()
            train_sum = 0.0
            train_count = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    losses, valid_count = _jepa_forward(
                        model,
                        batch,
                        device,
                        use_ego_actions=use_ego_actions,
                    )
                if scaler is None:
                    losses["total"].backward()
                    nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                else:
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                progress = global_step / max(total_steps - 1, 1)
                momentum = ema_end - (ema_end - ema_start) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
                model.update_target_encoder(momentum)
                global_step += 1
                train_sum += float(losses["total"].detach()) * valid_count
                train_count += valid_count
            validation = _evaluate_jepa_loss(
                model,
                validation_loader,
                device,
                use_ego_actions=use_ego_actions,
            )
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_total": train_sum / max(train_count, 1),
                "validation": validation,
                "ema_momentum": momentum,
            }
            history.append(row)
            history_file.write(json.dumps(row, sort_keys=True) + "\n")
            history_file.flush()
            if validation["total"] < best_validation:
                best_validation = validation["total"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(config),
                        "epoch": epoch,
                        "seed": seed,
                        "checkpoint_role": "best",
                        "selected_by": "validation_object_jepa_total",
                        "uses_ttc_labels": False,
                        "uses_ego_actions": use_ego_actions,
                        "cache_manifest": str(cache_manifest_path),
                        "manifest_sha256": _hash_file(cache_manifest_path),
                        "git_commit": _get_git_commit(),
                        "protocol_version": "2.0",
                        "run_fingerprint": run_fingerprint,
                        "run_fingerprint_payload": run_fingerprint_payload,
                    },
                    best_path,
                )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(config),
            "epoch": epochs,
            "seed": seed,
            "checkpoint_role": "last",
            "selected_by": "final_epoch",
            "uses_ttc_labels": False,
            "uses_ego_actions": use_ego_actions,
            "cache_manifest": str(cache_manifest_path),
            "manifest_sha256": _hash_file(cache_manifest_path),
            "git_commit": _get_git_commit(),
            "protocol_version": "2.0",
            "run_fingerprint": run_fingerprint,
            "run_fingerprint_payload": run_fingerprint_payload,
        },
        last_path,
    )
    summary: dict[str, Any] = {
        "method": "object_centric_recurrent_event_jepa",
        "cache_manifest": str(cache_manifest_path),
        "model_config": asdict(config),
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "uses_ttc_labels": False,
        "uses_future_ego_actions_in_teacher": False,
        "uses_ego_actions_in_student_predictor": use_ego_actions,
        "best_epoch": best_epoch,
        "best_validation_total": best_validation,
        "best_checkpoint": best_path.as_posix(),
        "last_checkpoint": last_path.as_posix(),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "elapsed_seconds": time.perf_counter() - start_time,
        "history": history,
        "manifest_sha256": _hash_file(cache_manifest_path),
        "git_commit": _get_git_commit(),
        "protocol_version": "2.0",
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_payload": run_fingerprint_payload,
        "final_test_opened": False,
    }
    write_structured(output / "summary.json", summary)
    train_dataset.close()
    validation_dataset.close()
    return summary


def _predict_current(
    model: ObjectCentricEventJEPA,
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
    *,
    use_ego_actions: bool = True,
) -> ObjectTTCOutput:
    return model.predict_ttc(
        _tensor_batch(batch, "context_events", device, dtype=torch.float32),
        _tensor_batch(batch, "context_boxes", device, dtype=torch.float32),
        _tensor_batch(batch, "context_object_mask", device).bool(),
        context_sampling_boxes=_tensor_batch(
            batch,
            "context_sampling_boxes",
            device,
            dtype=torch.float32,
        ),
        context_ego_actions=(
            _tensor_batch(batch, "context_ego_actions", device, dtype=torch.float32)
            if use_ego_actions
            else None
        ),
        context_ego_action_mask=(
            _tensor_batch(batch, "context_ego_action_mask", device).bool()
            if use_ego_actions
            else None
        ),
    )


def _supervised_object_loss(
    prediction: ObjectTTCOutput,
    ttc_s: torch.Tensor,
    *,
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> dict[str, torch.Tensor]:
    valid = prediction.object_mask & torch.isfinite(ttc_s) & (ttc_s.abs() >= 0.1)
    if not torch.any(valid):
        msg = "Supervised TTC batch contains no valid object labels."
        raise ValueError(msg)
    inverse_target = torch.reciprocal(ttc_s[valid])
    residual = prediction.inverse_ttc_mean[valid] - inverse_target
    log_variance = prediction.inverse_ttc_log_variance[valid]
    inverse_nll = (0.5 * torch.exp(-log_variance) * residual.square() + 0.5 * log_variance).mean()
    thresholds = prediction.risk_logits.new_tensor(risk_thresholds_s)
    if prediction.risk_logits.shape[-1] != thresholds.numel():
        msg = "Risk thresholds do not match the model risk-head width."
        raise ValueError(msg)
    labels = ((ttc_s[..., None] > 0.0) & (ttc_s[..., None] <= thresholds[None, None, :])).to(
        prediction.risk_logits.dtype
    )
    risk_bce = functional.binary_cross_entropy_with_logits(
        prediction.risk_logits[valid],
        labels[valid],
    )
    return {
        "total": inverse_nll + 0.25 * risk_bce,
        "inverse_nll": inverse_nll,
        "risk_bce": risk_bce,
    }


def _label_subset_indices(
    dataset: Dataset[dict[str, torch.Tensor | str]],
    *,
    fraction: float,
    seed: int,
) -> list[int]:
    if not 0.0 < fraction <= 1.0:
        msg = "Label fraction must lie in (0, 1]."
        raise ValueError(msg)
    if fraction == 1.0:
        return list(range(len(dataset)))
    strata: dict[tuple[str, int], list[int]] = {}
    edges = np.asarray([-10.0, 0.0, 3.0, 6.0, 10.0])
    for index in range(len(dataset)):
        sample = dataset[index]
        sequence_id = str(sample["sequence_id"])
        ttc = sample["ttc_s"]
        if not isinstance(ttc, torch.Tensor):
            msg = "Object cache TTC label must be a tensor."
            raise TypeError(msg)
        value = float(ttc.reshape(-1)[0])
        bin_index = int(np.clip(np.digitize(value, edges), 0, len(edges)))
        strata.setdefault((sequence_id, bin_index), []).append(index)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for indices in strata.values():
        count = max(1, int(round(len(indices) * fraction)))
        selected.extend(rng.choice(indices, size=min(count, len(indices)), replace=False).tolist())
    return sorted(set(int(index) for index in selected))


def _collect_predictions(
    model: ObjectCentricEventJEPA,
    dataset: Dataset[dict[str, torch.Tensor | str]],
    *,
    device: torch.device,
    batch_size: int,
    use_ego_actions: bool = True,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "ttc_true": [],
        "ttc_pred": [],
        "ttc_std": [],
        "inverse_mean": [],
        "inverse_log_variance": [],
        "risk_logits": [],
    }
    sequences: list[str] = []
    tokens: list[str] = []
    with torch.no_grad():
        for batch in loader:
            prediction = _predict_current(
                model,
                batch,
                device,
                use_ego_actions=use_ego_actions,
            )
            ttc_mean, ttc_std = inverse_ttc_distribution_to_seconds(
                prediction.inverse_ttc_mean,
                prediction.inverse_ttc_log_variance,
            )
            mask = prediction.object_mask
            truth = _tensor_batch(batch, "ttc_s", device, dtype=torch.float32)
            for name, tensor in (
                ("ttc_true", truth[mask]),
                ("ttc_pred", ttc_mean[mask]),
                ("ttc_std", ttc_std[mask]),
                ("inverse_mean", prediction.inverse_ttc_mean[mask]),
                ("inverse_log_variance", prediction.inverse_ttc_log_variance[mask]),
                ("risk_logits", prediction.risk_logits[mask]),
            ):
                collected[name].append(tensor.detach().cpu().numpy())
            batch_sequences = batch["sequence_id"]
            batch_tokens = batch["sample_token"]
            if not isinstance(batch_sequences, list) or not isinstance(batch_tokens, list):
                msg = "String cache fields must collate to lists."
                raise TypeError(msg)
            sequences.extend(str(value) for value in batch_sequences)
            tokens.extend(str(value) for value in batch_tokens)
    output = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    output["sequence_id"] = np.asarray(sequences)
    output["sample_token"] = np.asarray(tokens)
    return output


def _prediction_metrics(predictions: dict[str, np.ndarray]) -> dict[str, object]:
    risk_probability = 1.0 / (1.0 + np.exp(-np.clip(predictions["risk_logits"], -30, 30)))
    metrics = object_ttc_metrics(
        predictions["ttc_true"],
        predictions["ttc_pred"],
        risk_probability,
    )
    intervals: dict[str, dict[str, float]] = {}
    for coverage in (0.5, 0.8, 0.95):
        z_value = NormalDist().inv_cdf((1.0 + coverage) * 0.5)
        radius = z_value * predictions["ttc_std"]
        intervals[str(coverage)] = interval_metrics(
            predictions["ttc_true"],
            predictions["ttc_pred"] - radius,
            predictions["ttc_pred"] + radius,
        )
    metrics["raw_intervals"] = intervals
    inverse_target = np.reciprocal(predictions["ttc_true"])
    inverse_variance = np.exp(predictions["inverse_log_variance"])
    metrics["inverse_ttc_nll"] = float(
        np.mean(
            0.5 * (predictions["inverse_mean"] - inverse_target) ** 2 / inverse_variance
            + 0.5 * np.log(inverse_variance)
        )
    )
    return metrics


def _fit_calibrators(
    predictions: dict[str, np.ndarray],
    *,
    risk_thresholds_s: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> tuple[ConformalIntervalCalibrator, list[TemperatureScaler]]:
    conformal = fit_conformal_interval(
        predictions["ttc_true"],
        predictions["ttc_pred"],
        predictions["ttc_std"],
        coverage=0.9,
    )
    temperatures = [
        fit_temperature_scaler(
            predictions["risk_logits"][:, index],
            ((predictions["ttc_true"] > 0.0) & (predictions["ttc_true"] <= threshold)).astype(
                np.int64
            ),
        )
        for index, threshold in enumerate(risk_thresholds_s)
    ]
    return conformal, temperatures


def fine_tune_object_ttc(
    *,
    cache_manifest_path: str | Path,
    output_dir: str | Path,
    pretrained_checkpoint_path: str | Path | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    label_fraction: float = 1.0,
    seed: int = 42,
    device_name: str = "auto",
    scratch_config: ObjectJEPAConfig | None = None,
    use_ego_actions: bool = True,
    report_splits: tuple[str, ...] = ("validation",),
    allow_final_test_evaluation: bool = False,
) -> dict[str, Any]:
    """Fine-tune a matched JEPA/scratch TTC model and calibrate on a held-out split."""

    _set_seed(seed)
    device = _device(device_name)
    checkpoint: dict[str, Any] | None = None
    if pretrained_checkpoint_path is not None:
        checkpoint = torch.load(
            pretrained_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        config = ObjectJEPAConfig(**checkpoint["model_config"])
    elif scratch_config is not None:
        config = scratch_config
    else:
        train_probe = EAPObjectCacheDataset(cache_manifest_path, splits=("train",))
        cache_metadata = json.loads(Path(cache_manifest_path).read_text(encoding="utf-8"))
        sample = train_probe[0]
        events = sample["context_events"]
        actions = sample["context_ego_actions"]
        if not isinstance(events, torch.Tensor) or not isinstance(actions, torch.Tensor):
            msg = "Cannot infer scratch Object-JEPA configuration from cache."
            raise TypeError(msg)
        config = ObjectJEPAConfig(
            in_channels=int(events.shape[1]),
            action_dim=int(actions.shape[-1]),
            pre_cropped_events=bool(cache_metadata.get("pre_cropped_events", True)),
        )
        train_probe.close()
    model = ObjectCentricEventJEPA(config).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for parameter in model.target_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.predictor.parameters():
        parameter.requires_grad_(False)

    train_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("train",))
    validation_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("validation",))

    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
    }

    if "calibration" in report_splits:
        datasets["calibration"] = EAPObjectCacheDataset(
            cache_manifest_path, splits=("calibration",)
        )

    if "test" in report_splits:
        if not allow_final_test_evaluation:
            raise ValueError(
                "Evaluation of the final test split requires allow_final_test_evaluation=True"
            )
        datasets["test"] = EAPObjectCacheDataset(cache_manifest_path, splits=("test",))
    selected_indices = _label_subset_indices(
        train_dataset,
        fraction=label_fraction,
        seed=seed,
    )
    selected_train = Subset(train_dataset, selected_indices)
    train_loader = DataLoader(
        selected_train,
        batch_size=batch_size,
        sampler=ShardLocalSampler(
            train_dataset,
            source_indices=selected_indices,
            seed=seed,
        ),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "object_ttc_best.pt"
    history_path = output / "history.jsonl"
    best_validation = float("inf")
    import hashlib

    run_fingerprint_payload = {
        "git_commit": _get_git_commit(),
        "protocol_version": "2.0",
        "cache_sha256": _hash_file(cache_manifest_path),
        "split_manifest_sha256": _hash_file(cache_manifest_path),
        "subset_manifest_sha256": "",
        "model_name": "object_ttc",
        "resolved_model_config": asdict(config),
        "navigation_mode": "enabled" if use_ego_actions else "disabled",
        "label_fraction": label_fraction,
        "seed": seed,
        "pretraining_checkpoint_sha256": _hash_file(pretrained_checkpoint_path)
        if pretrained_checkpoint_path
        else "",
        "optimizer_config": {"learning_rate": learning_rate, "weight_decay": weight_decay},
        "training_steps": epochs,
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(run_fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    best_epoch = -1
    start_time = time.perf_counter()
    with history_path.open("w", encoding="utf-8") as history_file:
        for epoch in range(1, epochs + 1):
            model.train()
            train_total = 0.0
            train_count = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = _predict_current(
                    model,
                    batch,
                    device,
                    use_ego_actions=use_ego_actions,
                )
                truth = _tensor_batch(batch, "ttc_s", device, dtype=torch.float32)
                losses = _supervised_object_loss(
                    prediction,
                    truth,
                    risk_thresholds_s=config.risk_thresholds_s,
                )
                losses["total"].backward()
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    1.0,
                )
                optimizer.step()
                count = int(prediction.object_mask.sum().item())
                train_total += float(losses["total"].detach()) * count
                train_count += count
            validation_predictions = _collect_predictions(
                model,
                validation_dataset,
                device=device,
                batch_size=batch_size,
                use_ego_actions=use_ego_actions,
            )
            inverse_target = np.reciprocal(validation_predictions["ttc_true"])
            validation_inverse_mae = float(
                np.mean(np.abs(validation_predictions["inverse_mean"] - inverse_target))
            )
            row = {
                "epoch": epoch,
                "train_total": train_total / max(train_count, 1),
                "validation_inverse_ttc_mae": validation_inverse_mae,
            }
            history_file.write(json.dumps(row, sort_keys=True) + "\n")
            history_file.flush()
            if validation_inverse_mae < best_validation:
                best_validation = validation_inverse_mae
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": asdict(config),
                        "epoch": epoch,
                        "seed": seed,
                        "label_fraction": label_fraction,
                        "initialization": "jepa" if checkpoint is not None else "scratch",
                        "pretrained_checkpoint": (
                            str(pretrained_checkpoint_path)
                            if pretrained_checkpoint_path is not None
                            else None
                        ),
                        "uses_ego_actions": use_ego_actions,
                        "checkpoint_role": "best",
                        "selected_by": "validation_inverse_ttc_mae",
                        "manifest_sha256": _hash_file(cache_manifest_path),
                        "git_commit": _get_git_commit(),
                        "protocol_version": "2.0",
                        "run_fingerprint": run_fingerprint,
                        "run_fingerprint_payload": run_fingerprint_payload,
                    },
                    best_path,
                )
    selected_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected_checkpoint["model_state_dict"])
    validation_predictions = _collect_predictions(
        model,
        validation_dataset,
        device=device,
        batch_size=batch_size,
        use_ego_actions=use_ego_actions,
    )
    # Fit calibrators if calibration is provided, else use dummy/defaults (or fail if requested)
    conformal = None
    temperatures = []
    if "calibration" in report_splits:
        calibration_predictions = _collect_predictions(
            model,
            datasets["calibration"],
            device=device,
            batch_size=batch_size,
            use_ego_actions=use_ego_actions,
        )
        conformal, temperatures = _fit_calibrators(
            calibration_predictions,
            risk_thresholds_s=config.risk_thresholds_s,
        )

    summary: dict[str, Any] = {
        "method": "object_centric_event_jepa_ttc",
        "initialization": "jepa" if checkpoint is not None else "scratch",
        "pretrained_checkpoint": (
            str(pretrained_checkpoint_path) if pretrained_checkpoint_path is not None else None
        ),
        "cache_manifest": str(cache_manifest_path),
        "model_config": asdict(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "seed": seed,
        "label_fraction": label_fraction,
        "uses_ego_actions": use_ego_actions,
        "effective_label_count": len(selected_indices),
        "full_train_count": len(train_dataset),
        "best_epoch": best_epoch,
        "best_validation_inverse_ttc_mae": best_validation,
        "validation": _prediction_metrics(validation_predictions),
        "evaluation_split": report_splits[-1] if report_splits else "validation",
        "final_test_opened": "test" in report_splits,
        "elapsed_seconds": time.perf_counter() - start_time,
        "best_checkpoint": best_path.as_posix(),
        "manifest_sha256": _hash_file(cache_manifest_path),
        "git_commit": _get_git_commit(),
        "protocol_version": "2.0",
        "run_fingerprint": run_fingerprint,
        "run_fingerprint_payload": run_fingerprint_payload,
    }

    if "calibration" in report_splits and conformal is not None:
        summary["calibration"] = {
            "split": "calibration",
            "count": int(calibration_predictions["ttc_true"].shape[0]),
            "conformal_coverage": conformal.coverage,
            "conformal_scale": conformal.scale,
            "temperatures": [temperature.temperature for temperature in temperatures],
        }

    if "test" in report_splits:
        test_predictions = _collect_predictions(
            model,
            datasets["test"],
            device=device,
            batch_size=batch_size,
            use_ego_actions=use_ego_actions,
        )
        calibrated_risk = (
            np.column_stack(
                [
                    temperature.probabilities(test_predictions["risk_logits"][:, index])
                    for index, temperature in enumerate(temperatures)
                ]
            )
            if temperatures
            else test_predictions["risk_logits"]
        )
        test_metrics = object_ttc_metrics(
            test_predictions["ttc_true"],
            test_predictions["ttc_pred"],
            calibrated_risk,
            risk_thresholds_s=config.risk_thresholds_s,
        )
        if conformal is not None:
            lower, upper = conformal.interval(
                test_predictions["ttc_pred"],
                test_predictions["ttc_std"],
            )
            test_metrics["conformal_90"] = interval_metrics(
                test_predictions["ttc_true"],
                lower,
                upper,
            )
        test_metrics["mae_sequence_bootstrap_95"] = sequence_bootstrap_interval(
            test_predictions["ttc_true"],
            test_predictions["ttc_pred"],
            test_predictions["sequence_id"],
            iterations=2000,
            confidence=0.95,
            seed=seed,
        )
        summary["test"] = test_metrics
        summary["test_evaluated_after_model_selection_and_calibration"] = True
        np.savez_compressed(output / "test_predictions.npz", **test_predictions)

    write_structured(output / "summary.json", summary)
    for dataset in datasets.values():
        dataset.close()
    return summary


def evaluate_object_ttc_checkpoint(
    *,
    cache_manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    splits: tuple[str, ...] = ("test",),
    calibration_summary_path: str | Path | None = None,
    batch_size: int = 32,
    device_name: str = "auto",
    bootstrap_iterations: int = 2000,
    seed: int = 42,
    use_ego_actions: bool = True,
) -> dict[str, Any]:
    """Evaluate a fixed checkpoint without fitting anything on evaluation data.

    When a fine-tuning summary is supplied, its calibration-split conformal
    scale and risk temperatures are applied verbatim. This is the entry point
    used for clean and corrupted-cache evaluations.
    """

    if batch_size <= 0 or bootstrap_iterations <= 0:
        msg = "Batch size and bootstrap iterations must be positive."
        raise ValueError(msg)
    device = _device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ObjectJEPAConfig(**checkpoint["model_config"])
    model = ObjectCentricEventJEPA(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    dataset = EAPObjectCacheDataset(cache_manifest_path, splits=splits)
    predictions = _collect_predictions(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        use_ego_actions=use_ego_actions,
    )
    raw_risk = 1.0 / (1.0 + np.exp(-np.clip(predictions["risk_logits"], -30.0, 30.0)))
    risk_probability = raw_risk
    calibration_payload: dict[str, Any] | None = None
    conformal: ConformalIntervalCalibrator | None = None
    if calibration_summary_path is not None:
        summary_source = Path(calibration_summary_path)
        calibration_source = json.loads(summary_source.read_text(encoding="utf-8"))["calibration"]
        temperature_values = calibration_source["temperatures"]
        if len(temperature_values) != len(config.risk_thresholds_s):
            msg = "Saved calibration temperatures do not match model risk thresholds."
            raise ValueError(msg)
        temperatures = [TemperatureScaler(float(value)) for value in temperature_values]
        risk_probability = np.column_stack(
            [
                scaler.probabilities(predictions["risk_logits"][:, index])
                for index, scaler in enumerate(temperatures)
            ]
        )
        conformal = ConformalIntervalCalibrator(
            coverage=float(calibration_source["conformal_coverage"]),
            scale=float(calibration_source["conformal_scale"]),
            calibration_count=int(calibration_source["count"]),
        )
        calibration_payload = {
            "source": summary_source.as_posix(),
            "risk_temperatures": [scaler.temperature for scaler in temperatures],
            "conformal_coverage": conformal.coverage,
            "conformal_scale": conformal.scale,
            "calibration_count": conformal.calibration_count,
        }

    metrics = object_ttc_metrics(
        predictions["ttc_true"],
        predictions["ttc_pred"],
        risk_probability,
        risk_thresholds_s=config.risk_thresholds_s,
    )
    if conformal is not None:
        lower, upper = conformal.interval(
            predictions["ttc_pred"],
            predictions["ttc_std"],
        )
        metrics[f"conformal_{conformal.coverage:g}"] = interval_metrics(
            predictions["ttc_true"],
            lower,
            upper,
        )
    metrics["mae_sequence_bootstrap_95"] = sequence_bootstrap_interval(
        predictions["ttc_true"],
        predictions["ttc_pred"],
        predictions["sequence_id"],
        iterations=bootstrap_iterations,
        confidence=0.95,
        seed=seed,
    )
    payload: dict[str, Any] = {
        "method": "fixed_object_ttc_checkpoint_evaluation",
        "checkpoint": Path(checkpoint_path).as_posix(),
        "cache_manifest": Path(cache_manifest_path).as_posix(),
        "splits": list(splits),
        "sample_count": int(predictions["ttc_true"].shape[0]),
        "sequence_count": int(np.unique(predictions["sequence_id"]).shape[0]),
        "calibration": calibration_payload,
        "fit_on_evaluation_data": False,
        "uses_ego_actions": use_ego_actions,
        "metrics": metrics,
        "final_test_opened": "test" in splits or "CPLA-high" in splits,
    }
    destination = Path(output_path)
    write_structured(destination, payload)
    np.savez_compressed(destination.with_suffix(".predictions.npz"), **predictions)
    dataset.close()
    return payload


__all__ = [
    "evaluate_object_ttc_checkpoint",
    "fine_tune_object_ttc",
    "pretrain_object_event_jepa",
]
