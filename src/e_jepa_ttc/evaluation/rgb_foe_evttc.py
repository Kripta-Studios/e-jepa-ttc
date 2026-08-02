"""Evaluate the source-traceable RGB/FoE affine-divergence baseline on EvTTC."""

from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.eap_cache import EAPObjectCacheDataset
from e_jepa_ttc.evaluation.object_ttc import (
    grouped_ttc_selection_components,
    object_ttc_metrics,
)
from e_jepa_ttc.geometry.rgb_foe import farneback_affine_ttc
from e_jepa_ttc.utils.io import write_structured


@dataclass(frozen=True)
class RGBFOEEvTTCConfig:
    """Evaluation settings for object-ROI RGB affine divergence."""

    grid_step: int = 2
    minimum_flow_px: float = 0.05
    maximum_flow_px: float = 64.0
    maximum_ttc_s: float = 12.0
    use_foreground_mask: bool = True
    max_train_samples: int | None = None
    max_validation_samples: int | None = None


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices(length: int, maximum: int | None) -> np.ndarray:
    if maximum is None or maximum >= length:
        return np.arange(length, dtype=np.int64)
    if maximum <= 0:
        raise ValueError("Sample limits must be positive.")
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


def _as_numpy(value: torch.Tensor | str, key: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor cache field {key!r}.")
    return value.detach().cpu().numpy()


def _foreground_mask(sample: dict[str, torch.Tensor | str], shape: tuple[int, int]) -> np.ndarray:
    import cv2

    masks = _as_numpy(sample["garl_foreground_mask"], "garl_foreground_mask")
    if masks.shape[0] != 2:
        raise ValueError("Garl foreground masks must contain exactly two endpoints.")
    union = np.any(masks > 0, axis=0).astype(np.uint8)
    return cv2.resize(union, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def _evaluate_dataset(
    dataset: EAPObjectCacheDataset,
    *,
    config: RGBFOEEvTTCConfig,
    maximum_samples: int | None,
) -> dict[str, np.ndarray]:
    truth: list[float] = []
    prediction: list[float] = []
    valid: list[bool] = []
    reason: list[str] = []
    divergence: list[float] = []
    residual_rmse_px: list[float] = []
    inlier_fraction: list[float] = []
    condition_number: list[float] = []
    flow_sample_count: list[int] = []
    runtime_s: list[float] = []
    sequences: list[str] = []
    tokens: list[str] = []
    for index in _indices(len(dataset), maximum_samples):
        sample = dataset[int(index)]
        rgb_pair = _as_numpy(sample["garl_rgb_pair"], "garl_rgb_pair")
        if rgb_pair.shape[0] != 2:
            raise ValueError("garl_rgb_pair must contain two RGB endpoints.")
        delta_t_s = float(_as_numpy(sample["garl_delta_t_s"], "garl_delta_t_s").reshape(-1)[0])
        mask = (
            _foreground_mask(sample, tuple(rgb_pair.shape[-2:]))
            if config.use_foreground_mask
            else None
        )
        started = time.perf_counter()
        result = farneback_affine_ttc(
            rgb_pair[0],
            rgb_pair[1],
            delta_t_s=delta_t_s,
            mask=mask,
            grid_step=config.grid_step,
            minimum_flow_px=config.minimum_flow_px,
            maximum_flow_px=config.maximum_flow_px,
            maximum_ttc_s=config.maximum_ttc_s,
        )
        runtime_s.append(time.perf_counter() - started)
        truth.append(float(_as_numpy(sample["ttc_s"], "ttc_s").reshape(-1)[0]))
        prediction.append(float(result.ttc_seconds))
        valid.append(bool(result.valid))
        reason.append(result.reason)
        divergence.append(float(result.divergence_per_second))
        residual_rmse_px.append(
            float(result.fit.residual_rmse_px) if result.fit is not None else float("nan")
        )
        inlier_fraction.append(float(result.fit.inlier_fraction) if result.fit is not None else 0.0)
        condition_number.append(
            float(result.fit.condition_number) if result.fit is not None else float("inf")
        )
        flow_sample_count.append(int(result.fit.sample_count) if result.fit is not None else 0)
        sequence = sample["sequence_id"]
        token = sample["sample_token"]
        if not isinstance(sequence, str) or not isinstance(token, str):
            raise TypeError("sequence_id and sample_token must be strings.")
        sequences.append(sequence)
        tokens.append(token)
    return {
        "truth_ttc_s": np.asarray(truth, dtype=np.float64),
        "raw_ttc_s": np.asarray(prediction, dtype=np.float64),
        "valid": np.asarray(valid, dtype=np.bool_),
        "reason": np.asarray(reason, dtype=np.str_),
        "divergence_per_second": np.asarray(divergence, dtype=np.float64),
        "residual_rmse_px": np.asarray(residual_rmse_px, dtype=np.float64),
        "inlier_fraction": np.asarray(inlier_fraction, dtype=np.float64),
        "condition_number": np.asarray(condition_number, dtype=np.float64),
        "flow_sample_count": np.asarray(flow_sample_count, dtype=np.int64),
        "runtime_s": np.asarray(runtime_s, dtype=np.float64),
        "sequence_id": np.asarray(sequences, dtype=np.str_),
        "sample_token": np.asarray(tokens, dtype=np.str_),
    }


def _fit_log_calibration(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(truth) & np.isfinite(prediction) & (truth > 0.0) & (prediction > 0.0)
    if int(np.count_nonzero(valid)) < 3:
        raise ValueError("RGB/FoE calibration requires at least three valid train estimates.")
    x = np.log(prediction[valid])
    y = np.log(truth[valid])
    design = np.column_stack((np.ones(x.shape[0], dtype=np.float64), x))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _apply_log_calibration(
    prediction: np.ndarray,
    beta: tuple[float, float],
    *,
    maximum_ttc_s: float,
) -> np.ndarray:
    calibrated = np.exp(beta[0] + beta[1] * np.log(np.clip(prediction, 1e-6, None)))
    return np.clip(calibrated, 1e-4, maximum_ttc_s)


def _metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    result = object_ttc_metrics(truth, prediction)
    result.update(grouped_ttc_selection_components(truth, prediction, groups))
    sequence_mae = [
        float(np.mean(np.abs(truth[groups == group] - prediction[groups == group])))
        for group in np.unique(groups)
    ]
    result["sequence_macro_mae_s"] = float(np.mean(sequence_mae))
    result["worst_sequence_mae_s"] = float(np.max(sequence_mae))
    return result


def evaluate_rgb_foe_evttc(
    *,
    cache_manifest: str | Path,
    output_dir: str | Path,
    config: RGBFOEEvTTCConfig | None = None,
) -> dict[str, Any]:
    """Evaluate raw, train-calibrated and honest-fallback RGB/FoE predictions."""

    source = Path(cache_manifest)
    output = Path(output_dir)
    assert_no_sealed_benchmark_paths((source, output))
    settings = config or RGBFOEEvTTCConfig()
    train_dataset = EAPObjectCacheDataset(source, splits=("train",))
    validation_dataset = EAPObjectCacheDataset(source, splits=("validation",))
    train = _evaluate_dataset(
        train_dataset,
        config=settings,
        maximum_samples=settings.max_train_samples,
    )
    validation = _evaluate_dataset(
        validation_dataset,
        config=settings,
        maximum_samples=settings.max_validation_samples,
    )
    calibration = _fit_log_calibration(train["truth_ttc_s"], train["raw_ttc_s"])
    valid = validation["valid"] & np.isfinite(validation["raw_ttc_s"])
    train_valid = train["valid"] & np.isfinite(train["raw_ttc_s"])
    train_fallback = float(np.median(train["truth_ttc_s"]))
    calibrated_valid = _apply_log_calibration(
        validation["raw_ttc_s"][valid],
        calibration,
        maximum_ttc_s=settings.maximum_ttc_s,
    )
    honest = np.full(validation["truth_ttc_s"].shape, train_fallback, dtype=np.float64)
    honest[valid] = calibrated_valid
    reason_counts = {
        str(value): int(np.count_nonzero(validation["reason"] == value))
        for value in np.unique(validation["reason"])
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz",
        **validation,
        calibrated_ttc_s=np.where(
            valid,
            _apply_log_calibration(
                np.where(valid, validation["raw_ttc_s"], 1.0),
                calibration,
                maximum_ttc_s=settings.maximum_ttc_s,
            ),
            np.nan,
        ),
        honest_fallback_ttc_s=honest,
    )
    payload: dict[str, Any] = {
        "method": "RGB_FOE_FARNEBACK_AFFINE_DIVERGENCE",
        "scientific_scope": (
            "Source-traceable local port on shared object ROIs; not claimed as an exact "
            "reproduction of the official EvTTC leaderboard implementation."
        ),
        "source_reference": "Stabinger et al., WACV 2016",
        "source_equation": "affine flow divergence=c2+c6; TTC=2*delta_t/divergence",
        "cache_manifest": str(source),
        "cache_manifest_sha256": _sha256(source),
        "git_commit": _git_commit(),
        "config": asdict(settings),
        "train": {
            "samples": int(train["truth_ttc_s"].size),
            "valid": int(np.count_nonzero(train_valid)),
            "coverage": float(np.mean(train_valid)),
            "fallback_median_ttc_s": train_fallback,
        },
        "calibration": {
            "fit_split": "train",
            "log_intercept": calibration[0],
            "log_slope": calibration[1],
        },
        "validation": {
            "samples": int(validation["truth_ttc_s"].size),
            "valid": int(np.count_nonzero(valid)),
            "coverage": float(np.mean(valid)),
            "failure_reasons": reason_counts,
            "runtime_median_ms": float(np.median(validation["runtime_s"]) * 1000.0),
            "runtime_p95_ms": float(np.quantile(validation["runtime_s"], 0.95) * 1000.0),
            "valid_only_raw": (
                _metrics(
                    validation["truth_ttc_s"][valid],
                    validation["raw_ttc_s"][valid],
                    validation["sequence_id"][valid],
                )
                if np.any(valid)
                else None
            ),
            "valid_only_train_calibrated": (
                _metrics(
                    validation["truth_ttc_s"][valid],
                    calibrated_valid,
                    validation["sequence_id"][valid],
                )
                if np.any(valid)
                else None
            ),
            "honest_train_median_fallback": _metrics(
                validation["truth_ttc_s"],
                honest,
                validation["sequence_id"],
            ),
        },
        "benchmark10_opened": False,
    }
    write_structured(output / "summary.json", payload)
    train_dataset.close()
    validation_dataset.close()
    return payload


__all__ = ["RGBFOEEvTTCConfig", "evaluate_rgb_foe_evttc"]
