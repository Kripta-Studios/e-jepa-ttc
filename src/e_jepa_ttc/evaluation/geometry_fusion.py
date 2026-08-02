"""Deterministic reliability-gated fusion of neural and geometric TTC."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.evaluation.object_ttc import (
    grouped_ttc_selection_components,
    object_ttc_metrics,
)
from e_jepa_ttc.utils.io import write_structured


@dataclass(frozen=True)
class GeometryFusionConfig:
    """Fixed physical confidence mapping; it is not fitted on validation."""

    residual_scale_px: float = 1.0
    maximum_geometry_weight: float = 1.0
    disagreement_log_scale: float = 1.0
    minimum_ttc_s: float = 0.1
    maximum_ttc_s: float = 12.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_reliability(
    *,
    valid: np.ndarray,
    inlier_fraction: np.ndarray,
    residual_rmse_px: np.ndarray,
    condition_number: np.ndarray,
    config: GeometryFusionConfig,
) -> np.ndarray:
    """Map solver health to ``[0,1]`` without looking at TTC labels."""

    if config.residual_scale_px <= 0.0 or config.disagreement_log_scale <= 0.0:
        raise ValueError("Fusion confidence scales must be positive.")
    if not 0.0 <= config.maximum_geometry_weight <= 1.0:
        raise ValueError("maximum_geometry_weight must be in [0,1].")
    valid_array = np.asarray(valid, dtype=np.bool_)
    inliers = np.clip(np.nan_to_num(inlier_fraction, nan=0.0), 0.0, 1.0)
    residual = np.nan_to_num(residual_rmse_px, nan=np.inf, posinf=np.inf)
    condition = np.nan_to_num(condition_number, nan=np.inf, posinf=np.inf)
    residual_score = np.exp(-np.maximum(residual, 0.0) / config.residual_scale_px)
    condition_score = np.reciprocal(np.sqrt(np.maximum(condition, 1.0)))
    return (
        valid_array.astype(np.float64)
        * config.maximum_geometry_weight
        * inliers
        * residual_score
        * condition_score
    )


def fuse_inverse_ttc(
    neural_ttc_s: np.ndarray,
    geometry_ttc_s: np.ndarray,
    reliability: np.ndarray,
    *,
    config: GeometryFusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse in inverse-TTC with label-free disagreement downweighting."""

    neural = np.asarray(neural_ttc_s, dtype=np.float64)
    geometry = np.asarray(geometry_ttc_s, dtype=np.float64)
    confidence = np.asarray(reliability, dtype=np.float64)
    if neural.shape != geometry.shape or neural.shape != confidence.shape:
        raise ValueError("Neural, geometry and reliability arrays must be shape matched.")
    if not np.all(np.isfinite(neural)) or np.any(neural <= 0.0):
        raise ValueError("Neural fallback TTC must be finite and positive.")
    geometry_valid = np.isfinite(geometry) & (geometry > 0.0)
    safe_geometry = np.where(geometry_valid, geometry, neural)
    log_disagreement = np.abs(
        np.log(np.clip(safe_geometry, 1e-6, None) / np.clip(neural, 1e-6, None))
    )
    agreement = np.exp(-log_disagreement / config.disagreement_log_scale)
    weight = np.clip(confidence * agreement, 0.0, 1.0)
    weight = np.where(geometry_valid, weight, 0.0)
    inverse = (1.0 - weight) / neural + weight / safe_geometry
    fused = np.clip(
        np.reciprocal(np.maximum(inverse, 1e-8)),
        config.minimum_ttc_s,
        config.maximum_ttc_s,
    )
    return fused, weight


def _metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    result = object_ttc_metrics(truth, prediction)
    result.update(grouped_ttc_selection_components(truth, prediction, groups))
    per_sequence_mae = [
        float(np.mean(np.abs(truth[groups == group] - prediction[groups == group])))
        for group in np.unique(groups)
    ]
    result["sequence_macro_mae_s"] = float(np.mean(per_sequence_mae))
    result["worst_sequence_mae_s"] = float(np.max(per_sequence_mae))
    return result


def evaluate_geometry_fusion(
    *,
    neural_predictions_path: str | Path,
    geometry_predictions_path: str | Path,
    output_dir: str | Path,
    config: GeometryFusionConfig | None = None,
) -> dict[str, Any]:
    """Evaluate a fixed fusion and an explicitly non-deployable oracle ceiling."""

    neural_path = Path(neural_predictions_path)
    geometry_path = Path(geometry_predictions_path)
    output = Path(output_dir)
    assert_no_sealed_benchmark_paths((neural_path, geometry_path, output))
    settings = config or GeometryFusionConfig()
    with np.load(neural_path, allow_pickle=False) as source:
        neural = {key: source[key] for key in source.files}
    with np.load(geometry_path, allow_pickle=False) as source:
        geometry = {key: source[key] for key in source.files}
    truth = np.asarray(neural["ttc_true"], dtype=np.float64)
    neural_ttc = np.asarray(neural["ttc_pred"], dtype=np.float64)
    groups = np.asarray(neural["sequence_id"]).astype(str)
    if not np.allclose(truth, geometry["truth_ttc_s"], rtol=1e-6, atol=1e-6):
        raise ValueError("Neural and geometry targets/order do not match.")
    if not np.array_equal(groups, np.asarray(geometry["sequence_id"]).astype(str)):
        raise ValueError("Neural and geometry sequence order does not match.")
    geometry_ttc = np.asarray(geometry["calibrated_ttc_s"], dtype=np.float64)
    valid = np.asarray(geometry["valid"], dtype=np.bool_) & np.isfinite(geometry_ttc)
    reliability = geometry_reliability(
        valid=valid,
        inlier_fraction=geometry["inlier_fraction"],
        residual_rmse_px=geometry["residual_rmse_px"],
        condition_number=geometry["condition_number"],
        config=settings,
    )
    fused, weights = fuse_inverse_ttc(
        neural_ttc,
        geometry_ttc,
        reliability,
        config=settings,
    )
    safe_geometry = np.where(valid, geometry_ttc, neural_ttc)
    oracle_uses_geometry = valid & (np.abs(safe_geometry - truth) < np.abs(neural_ttc - truth))
    oracle = np.where(oracle_uses_geometry, safe_geometry, neural_ttc)
    payload: dict[str, Any] = {
        "method": "DETERMINISTIC_RELIABILITY_GATED_INVERSE_TTC_FUSION",
        "scientific_scope": (
            "Fixed label-free solver-health and disagreement gate. The oracle complementarity "
            "row uses validation labels and is diagnostic only, never a candidate."
        ),
        "config": asdict(settings),
        "neural_predictions": str(neural_path),
        "neural_predictions_sha256": _sha256(neural_path),
        "geometry_predictions": str(geometry_path),
        "geometry_predictions_sha256": _sha256(geometry_path),
        "sample_count": int(truth.size),
        "geometry_valid_count": int(np.count_nonzero(valid)),
        "geometry_coverage": float(np.mean(valid)),
        "geometry_weight": {
            "mean": float(np.mean(weights)),
            "median": float(np.median(weights)),
            "p95": float(np.quantile(weights, 0.95)),
            "maximum": float(np.max(weights)),
        },
        "neural": _metrics(truth, neural_ttc, groups),
        "fixed_fusion": _metrics(truth, fused, groups),
        "oracle_complementarity_ceiling": {
            "uses_geometry_count": int(np.count_nonzero(oracle_uses_geometry)),
            "metrics": _metrics(truth, oracle, groups),
            "deployable": False,
        },
        "benchmark10_opened": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz",
        ttc_true=truth,
        neural_ttc_s=neural_ttc,
        geometry_ttc_s=geometry_ttc,
        geometry_valid=valid,
        geometry_reliability=reliability,
        geometry_weight=weights,
        fused_ttc_s=fused,
        sequence_id=groups,
    )
    write_structured(output / "summary.json", payload)
    return payload


__all__ = [
    "GeometryFusionConfig",
    "evaluate_geometry_fusion",
    "fuse_inverse_ttc",
    "geometry_reliability",
]
