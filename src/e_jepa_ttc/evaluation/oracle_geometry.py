"""Training-free evaluation of causal GT-box TTC geometry."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset
from e_jepa_ttc.evaluation.object_ttc import (
    grouped_ttc_selection_components,
    object_ttc_metrics,
)
from e_jepa_ttc.geometry import (
    affine_expansion_inverse_ttc,
    area_rate_inverse_ttc,
    event_contrast_inverse_ttc,
    geometry_track_confidence,
    height_ratio_inverse_ttc,
    weighted_inverse_ttc,
)
from e_jepa_ttc.geometry.ego_motion_compensation import (
    CameraEgoMotionCompensator,
    CameraYawDerotator,
)
from e_jepa_ttc.utils.io import write_structured


@dataclass(frozen=True)
class GTGeometryOracleConfig:
    """Bounded physical experts evaluated without fitting TTC labels."""

    evaluate_yaw_derotation: bool = True
    evaluate_translation_compensation: bool = True
    fit_train_only_log_calibration: bool = True
    action_dim: int = 8
    maximum_ttc_s: float = 12.0


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def _indices(length: int, maximum: int | None) -> list[int]:
    if maximum is None or maximum >= length:
        return list(range(length))
    if maximum <= 0:
        raise ValueError("max_validation_samples must be positive.")
    return np.linspace(0, length - 1, maximum, dtype=np.int64).tolist()


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


def _experts(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    events: torch.Tensor,
    times_s: torch.Tensor,
    *,
    use_event_contrast: bool = True,
) -> dict[str, torch.Tensor]:
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    widths = (x1 - x0).clamp_min(1e-6)
    heights = (y1 - y0).clamp_min(1e-6)
    height, height_confidence = height_ratio_inverse_ttc(
        heights,
        times_s,
        valid_mask=valid,
    )
    area, area_confidence = area_rate_inverse_ttc(
        widths * heights,
        times_s,
        valid_mask=valid,
    )
    affine, affine_confidence = affine_expansion_inverse_ttc(
        boxes,
        times_s,
        valid_mask=valid,
    )
    if use_event_contrast:
        contrast, contrast_confidence = event_contrast_inverse_ttc(
            events,
            times_s,
            soft_masks=_box_masks(boxes, valid, events.shape[-2:]),
        )
        if contrast.ndim == 1:
            contrast = contrast[:, None].expand_as(height)
            contrast_confidence = contrast_confidence[:, None].expand_as(height)
    else:
        contrast = torch.zeros_like(height)
        contrast_confidence = torch.zeros_like(height)
    track_confidence = geometry_track_confidence(boxes, valid)
    confidence = (
        torch.stack(
            (
                height_confidence,
                area_confidence,
                affine_confidence,
                contrast_confidence,
            ),
            dim=-1,
        )
        * track_confidence[..., None]
    )
    estimates = torch.stack((height, area, affine, contrast), dim=-1)
    mixture, _ = weighted_inverse_ttc(estimates, confidence)
    return {
        "height": height[:, 0],
        "area": area[:, 0],
        "affine": affine[:, 0],
        "event_contrast": contrast[:, 0],
        "deterministic_mixture": mixture[:, 0],
    }


def _box_masks(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    spatial_shape: tuple[int, int],
) -> torch.Tensor:
    """Rasterize normalized causal GT boxes for object-centric event contrast."""

    height, width = spatial_shape
    y = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5) / height
    x = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    inside = (
        (grid_x >= x0[..., None, None])
        & (grid_x <= x1[..., None, None])
        & (grid_y >= y0[..., None, None])
        & (grid_y <= y1[..., None, None])
        & valid[..., None, None]
    )
    return inside.any(dim=2).to(boxes.dtype)


def _metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    sequences: np.ndarray,
) -> dict[str, Any]:
    metrics = object_ttc_metrics(truth, prediction)
    metrics.update(grouped_ttc_selection_components(truth, prediction, sequences))
    per_sequence_mae = [
        float(np.mean(np.abs(truth[sequences == sequence] - prediction[sequences == sequence])))
        for sequence in np.unique(sequences)
    ]
    metrics["sequence_macro_mae_s"] = float(np.mean(per_sequence_mae))
    metrics["worst_sequence_mae_s"] = float(np.max(per_sequence_mae))
    return metrics


def _fit_log_calibration(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float]:
    """Fit log(TTC_gt) = beta0 + beta1*log(TTC_geometry) on train only."""

    valid = (
        np.isfinite(truth)
        & np.isfinite(prediction)
        & (truth > 0.0)
        & (prediction > 0.0)
    )
    if int(valid.sum()) < 3:
        raise ValueError("At least three finite positive train rows are required.")
    x = np.log(prediction[valid].astype(np.float64))
    y = np.log(truth[valid].astype(np.float64))
    design = np.column_stack((np.ones_like(x), x))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _apply_log_calibration(
    prediction: np.ndarray,
    beta: tuple[float, float],
    *,
    maximum_ttc_s: float,
) -> np.ndarray:
    calibrated = np.exp(
        beta[0] + beta[1] * np.log(np.clip(prediction.astype(np.float64), 1e-4, None))
    )
    return np.clip(calibrated, 1e-4, maximum_ttc_s)


def _standard_predictions(
    loader: DataLoader[Any],
    *,
    device: torch.device,
    maximum_ttc_s: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Collect uncalibrated causal geometry predictions for one split."""

    predictions: dict[str, list[np.ndarray]] = {}
    truths: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            events = _tensor(batch, "context_events", device, dtype=torch.float32)
            boxes = _tensor(batch, "context_boxes", device, dtype=torch.float32)
            valid = _tensor(batch, "context_object_mask", device).bool()
            end_us = _tensor(
                batch,
                "context_window_end_us",
                device,
                dtype=torch.float32,
            )
            times_s = (end_us - end_us[:, :1]) * 1e-6
            current = _experts(boxes, valid, events, times_s)
            for name, inverse_ttc in current.items():
                ttc = inverse_ttc.clamp_min(1e-4).reciprocal().clamp_max(
                    maximum_ttc_s
                )
                predictions.setdefault(name, []).append(ttc.cpu().numpy())
            truths.append(_tensor(batch, "ttc_s", device).reshape(-1).cpu().numpy())
    return (
        {name: np.concatenate(values) for name, values in predictions.items()},
        np.concatenate(truths),
    )


def evaluate_gt_geometry_oracle(
    *,
    cache_manifest_path: str | Path,
    output_dir: str | Path,
    config: GTGeometryOracleConfig | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    device_name: str = "auto",
    dry_run_fingerprint: bool = False,
) -> dict[str, Any] | str:
    """Evaluate every physical expert once; no optimizer or checkpoint is created."""

    resolved = config or GTGeometryOracleConfig()
    assert_no_sealed_benchmark_paths((cache_manifest_path, output_dir))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative.")
    fingerprint_payload = {
        "cache_manifest_sha256": _hash_file(cache_manifest_path),
        "config": asdict(resolved),
        "max_train_samples": max_train_samples,
        "max_validation_samples": max_validation_samples,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if dry_run_fingerprint:
        return fingerprint

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("validation",))
    indices = _indices(len(dataset), max_validation_samples)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(Subset(dataset, indices), **loader_kwargs)
    calibration: dict[str, dict[str, float | int]] = {}
    calibration_sample_count = 0
    calibration_seconds = 0.0
    if resolved.fit_train_only_log_calibration:
        calibration_started = time.perf_counter()
        train_dataset = EAPObjectCacheDataset(cache_manifest_path, splits=("train",))
        train_indices = _indices(len(train_dataset), max_train_samples)
        train_loader = DataLoader(
            Subset(train_dataset, train_indices),
            **loader_kwargs,
        )
        train_predictions, train_truth = _standard_predictions(
            train_loader,
            device=device,
            maximum_ttc_s=resolved.maximum_ttc_s,
        )
        calibration_sample_count = int(train_truth.shape[0])
        for name, prediction in train_predictions.items():
            beta = _fit_log_calibration(train_truth, prediction)
            calibration[name] = {
                "beta0": beta[0],
                "beta1": beta[1],
                "train_rows": calibration_sample_count,
            }
        del train_loader
        train_dataset.close()
        calibration_seconds = time.perf_counter() - calibration_started
    yaw_derotator = CameraYawDerotator(resolved.action_dim).to(device)
    ego_compensator = CameraEgoMotionCompensator(resolved.action_dim).to(device)
    predictions: dict[str, list[np.ndarray]] = {}
    truths: list[np.ndarray] = []
    sequence_ids: list[str] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            events = _tensor(batch, "context_events", device, dtype=torch.float32)
            boxes = _tensor(batch, "context_boxes", device, dtype=torch.float32)
            valid = _tensor(batch, "context_object_mask", device).bool()
            end_us = _tensor(
                batch,
                "context_window_end_us",
                device,
                dtype=torch.float32,
            )
            times_s = (end_us - end_us[:, :1]) * 1e-6
            current = _experts(boxes, valid, events, times_s)
            if resolved.evaluate_yaw_derotation:
                aligned, _ = yaw_derotator(
                    boxes,
                    _tensor(
                        batch,
                        "context_ego_actions",
                        device,
                        dtype=torch.float32,
                    ),
                    _tensor(batch, "context_ego_action_mask", device).bool(),
                    times_s,
                    intrinsics_normalized=_tensor(
                        batch,
                        "context_intrinsics_normalized",
                        device,
                        dtype=torch.float32,
                    ),
                )
                current["yaw_derotated_deterministic_mixture"] = _experts(
                    aligned,
                    valid,
                    events,
                    times_s,
                )["deterministic_mixture"]
            if resolved.evaluate_translation_compensation:
                depth_history = _tensor(
                    batch,
                    "context_depth_history_m",
                    device,
                    dtype=torch.float32,
                )
                aligned, _, _, ego_inverse_ttc = ego_compensator(
                    boxes,
                    depth_history,
                    _tensor(
                        batch,
                        "context_ego_actions",
                        device,
                        dtype=torch.float32,
                    ),
                    _tensor(batch, "context_ego_action_mask", device).bool(),
                    times_s,
                    intrinsics_normalized=_tensor(
                        batch,
                        "context_intrinsics_normalized",
                        device,
                        dtype=torch.float32,
                    ),
                )
                residual_inverse_ttc = _experts(
                    aligned,
                    valid,
                    events,
                    times_s,
                    use_event_contrast=False,
                )["deterministic_mixture"]
                current["translation_compensated_box_mixture_oracle"] = (
                    residual_inverse_ttc + ego_inverse_ttc[:, 0]
                ).clamp_min(1e-4)
            for name, inverse_ttc in current.items():
                ttc = inverse_ttc.clamp_min(1e-4).reciprocal().clamp_max(
                    resolved.maximum_ttc_s
                )
                predictions.setdefault(name, []).append(ttc.cpu().numpy())
            truths.append(_tensor(batch, "ttc_s", device).reshape(-1).cpu().numpy())
            values = batch["sequence_id"]
            if not isinstance(values, list):
                raise TypeError("sequence_id must collate to a list.")
            sequence_ids.extend(str(value) for value in values)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    truth = np.concatenate(truths)
    sequences = np.asarray(sequence_ids)
    prediction_arrays = {
        name: np.concatenate(values) for name, values in predictions.items()
    }
    for name, parameters in calibration.items():
        prediction_arrays[f"{name}_train_calibrated"] = _apply_log_calibration(
            prediction_arrays[name],
            (float(parameters["beta0"]), float(parameters["beta1"])),
            maximum_ttc_s=resolved.maximum_ttc_s,
        )
    variants = {
        name: _metrics(truth, prediction, sequences)
        for name, prediction in prediction_arrays.items()
    }
    primary = "deterministic_mixture"
    primary_metrics = variants[primary]
    primary_metrics["evaluation_seconds"] = elapsed
    primary_metrics["milliseconds_per_window"] = (
        1000.0 * elapsed / max(truth.shape[0], 1)
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "validation_predictions.npz",
        ttc_true=truth,
        sequence_id=sequences,
        **{f"ttc_pred_{name}": value for name, value in prediction_arrays.items()},
    )
    summary: dict[str, Any] = {
        "architecture": asdict(resolved),
        "evaluation_kind": (
            "gt_geometry_oracle_with_train_only_log_calibration"
            if resolved.fit_train_only_log_calibration
            else "training_free_gt_geometry_oracle"
        ),
        "run_fingerprint": fingerprint,
        "git_commit": _git_commit(),
        "cache_manifest": Path(cache_manifest_path).as_posix(),
        "validation_samples": len(indices),
        "train_calibration_samples": calibration_sample_count,
        "train_calibration_seconds": calibration_seconds,
        "train_only_calibration": calibration,
        "epochs_completed": 0,
        "best_epoch": 0,
        "stopped_early": False,
        "best_checkpoint": None,
        "last_checkpoint": None,
        "weights_only_checkpoint": None,
        "resume_checkpoint": None,
        "checkpoint_policy": [],
        "validation": primary_metrics,
        "oracle_variants": variants,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "benchmark10_opened": False,
        "translation_compensation_depth_source": (
            "official_evttc_distance_oracle_not_available_to_final_inference"
            if resolved.evaluate_translation_compensation
            else "disabled"
        ),
    }
    write_structured(output / "summary.json", summary)
    dataset.close()
    return summary


__all__ = [
    "GTGeometryOracleConfig",
    "evaluate_gt_geometry_oracle",
]
