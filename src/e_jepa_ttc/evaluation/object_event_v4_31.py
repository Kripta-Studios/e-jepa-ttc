"""Pure batched controls, aggregation and fail-closed causal decisions for v4.31."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch.nn import functional

SEEDS = (7, 13, 23)
THRESHOLDS: dict[str, float] = {
    "js_median_max": 0.02,
    "js_p95_max": 0.08,
    "displacement_p95_max": 0.5,
    "nonempty_min": 0.95,
    "valid_energy_fraction_min": 1.0,
    "high_band_cv_p95_max": 0.20,
    "effective_rank_min": 0.25,
    "analytic_pearson_min": 0.95,
    "slope_min": 0.8,
    "slope_max": 1.2,
    "sign_min": 0.95,
    "oddness_median_max": 0.20,
    "oddness_p95_max": 0.50,
    "identity_p95_max": 1e-4,
    "leakage_p95_max": 0.20,
    "swap_corr_max": -0.80,
    "swap_flip_min": 0.90,
    "swap_coverage_min": 0.25,
    "sequence_swap_corr_max": -0.50,
    "sequence_swap_flip_min": 0.75,
    "sequence_swap_coverage_min": 0.15,
}


class AuditModel(Protocol):
    """Minimal label-free model contract; implementations may return tensor or mapping."""

    def __call__(self, events: torch.Tensor) -> object: ...


def _model_device(model: AuditModel, fallback: torch.device) -> torch.device:
    """Return a module parameter device without assuming every test double is a module."""
    if isinstance(model, torch.nn.Module):
        parameter = next(model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(model.buffers(), None)
        if buffer is not None:
            return buffer.device
    return fallback


def strict_json(value: object) -> str:
    """Strict JSON rejects NaN/Inf and makes failures themselves auditable."""
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def _warp(image: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    flat = image.reshape(-1, 1, *image.shape[-2:])
    grid = functional.affine_grid(
        theta.expand(flat.shape[0], -1, -1), list(flat.shape), align_corners=False
    )
    return functional.grid_sample(
        flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).reshape_as(image)


def spatial_transform(
    events: torch.Tensor, *, log_eta: float = 0.0, translation_x: float = 0.0, rotation: float = 0.0
) -> torch.Tensor:
    """Warp only voxel channels 0:10; rate/count channels 10:12 remain exact constants."""
    if events.ndim != 5 or events.shape[2] != 12:
        raise ValueError("events must be [B,3,12,H,W]")
    scale = math.exp(log_eta)
    c, s = math.cos(rotation) / scale, math.sin(rotation) / scale
    theta = torch.tensor(
        [[c, -s, translation_x], [s, c, 0.0]], dtype=events.dtype, device=events.device
    )[None]
    return torch.cat((_warp(events[:, :, :10], theta), events[:, :, 10:12]), dim=2)


def trajectory_control(events: torch.Tensor, kind: str, amount: float = 0.0) -> torch.Tensor:
    """Produce every registered batched control through the same tensor contract."""
    if kind == "identity":
        return events.clone()
    if kind == "zero_event":
        return torch.zeros_like(events)
    if kind == "swap":
        return events[:, (0, 2, 1)]
    if kind == "reverse":
        return events[:, (2, 1, 0)]
    result = events.clone()
    if kind == "zoom":
        result[:, 0] = spatial_transform(events[:, 0:1], log_eta=-amount)[:, 0]
        result[:, 2] = spatial_transform(events[:, 2:3], log_eta=amount)[:, 0]
    elif kind == "translation":
        result[:, 0] = spatial_transform(events[:, 0:1], translation_x=-amount)[:, 0]
        result[:, 2] = spatial_transform(events[:, 2:3], translation_x=amount)[:, 0]
    elif kind == "rotation":
        result[:, 0] = spatial_transform(events[:, 0:1], rotation=-amount)[:, 0]
        result[:, 2] = spatial_transform(events[:, 2:3], rotation=amount)[:, 0]
    else:
        raise ValueError(f"unknown v4.31 control {kind!r}")
    return result


def _output(value: object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(value, Mapping):
        log_eta, unknown = value["log_eta"], value.get("unknown", False)
    else:
        log_eta, unknown = getattr(value, "log_eta", value), getattr(value, "unknown", False)
    return np.asarray(torch.as_tensor(log_eta).detach().cpu(), dtype=float).reshape(-1), np.asarray(
        torch.as_tensor(unknown).detach().cpu(), dtype=bool
    ).reshape(-1)


def batched_controls(
    model: AuditModel, events: torch.Tensor, sequence_ids: Iterable[str]
) -> dict[str, Any]:
    """Run all controls in batches, never rowwise or through an alternate input path."""
    ids = list(sequence_ids)
    if len(ids) != len(events):
        raise ValueError("sequence IDs must align with batch")
    model_events = events.to(_model_device(model, events.device))
    base, _ = _output(model(model_events))
    data: dict[str, Any] = {"base": base, "sequence_id": ids}
    for kind, amounts in {
        "identity": (0.0,),
        "zoom": (-0.04, -0.02, 0.02, 0.04),
        "translation": (-0.02, 0.02),
        "rotation": (-0.02, 0.02),
        "swap": (0.0,),
        "reverse": (0.0,),
        "zero_event": (0.0,),
    }.items():
        for amount in amounts:
            prediction, unknown = _output(model(trajectory_control(model_events, kind, amount)))
            data[f"{kind}:{amount:+.2f}"] = {"prediction": prediction, "unknown": unknown}
    return data


def chunked_controls(
    model: AuditModel, events: torch.Tensor, sequence_ids: Iterable[str], batch_size: int
) -> dict[str, Any]:
    """Run controls with a bounded forward batch and concatenate row-aligned outputs."""
    ids = list(sequence_ids)
    if batch_size <= 0 or len(events) != len(ids):
        raise ValueError("positive batch size and aligned sequence IDs are required")
    chunks = [
        batched_controls(model, events[start : start + batch_size], ids[start : start + batch_size])
        for start in range(0, len(events), batch_size)
    ]
    result: dict[str, Any] = {
        "base": np.concatenate([item["base"] for item in chunks]),
        "sequence_id": [value for item in chunks for value in item["sequence_id"]],
    }
    for key in chunks[0]:
        if key in {"base", "sequence_id"}:
            continue
        result[key] = {
            "prediction": np.concatenate([item[key]["prediction"] for item in chunks]),
            "unknown": np.concatenate([item[key]["unknown"] for item in chunks]),
        }
    return result


def pearson(left: Iterable[float], right: Iterable[float]) -> float | None:
    x, y = np.asarray(list(left), float), np.asarray(list(right), float)
    if (
        len(x) < 2
        or x.shape != y.shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def radial_spectrum(map_: torch.Tensor, bins: int = 8) -> dict[str, float | bool]:
    """DC-excluded radial FFT and normalized covariance rank for one projected map."""
    if map_.ndim != 3 or min(map_.shape[-2:]) < 4:
        raise ValueError("spectrum map must be [C,H,W] with spatial extent >= 4")
    if not bool(torch.isfinite(map_).all()):
        raise FloatingPointError("spectrum map is nonfinite")
    channels, height, width = map_.shape
    centered = map_.float() - map_.float().mean(dim=(-2, -1), keepdim=True)
    power = torch.fft.rfft2(centered).abs().square().mean(0)
    yy = torch.fft.fftfreq(height, device=map_.device)[:, None]
    xx = torch.fft.rfftfreq(width, device=map_.device)[None, :]
    radius = torch.sqrt(yy.square() + xx.square())
    edges = torch.linspace(0, float(radius.max()), bins + 1, device=map_.device)
    energy = []
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1]) & (radius > 0)
        energy.append(float(power[mask].sum().cpu()))
    total = float(sum(energy))
    valid = total > 0 and math.isfinite(total)
    fractions = np.asarray(energy, dtype=float) / total if valid else np.zeros(bins)
    flat = centered.reshape(channels, -1)
    covariance = flat @ flat.T / max(flat.shape[1], 1)
    values = torch.linalg.eigvalsh(covariance).clamp_min(0)
    normalized = values / values.sum().clamp_min(1e-12)
    rank = float(torch.exp(-(normalized * normalized.clamp_min(1e-12).log()).sum()) / channels)
    correlation = torch.corrcoef(flat) if channels > 1 else torch.ones(1, 1, device=map_.device)
    correlation = torch.nan_to_num(correlation, nan=0.0)
    return {
        "valid_energy": valid,
        "low_fraction": float(fractions[:3].sum()),
        "mid_fraction": float(fractions[3:6].sum()),
        "high_fraction": float(fractions[6:].sum()),
        "spectral_centroid": float(np.dot(fractions, (np.arange(bins) + 0.5) / bins)),
        "effective_rank": rank,
        "cross_channel_correlation": float(
            (correlation - torch.eye(channels, device=map_.device)).abs().mean().cpu()
        ),
    }


def posterior_stability_by_seed(
    posteriors: Mapping[int, Mapping[int, np.ndarray]], offsets: Mapping[int, np.ndarray]
) -> tuple[dict[int, dict[str, float]], dict[str, float], dict[str, int]]:
    """Aggregate pairwise posterior disagreement into seed-local and joint summaries."""
    seeds = tuple(sorted(posteriors))
    if seeds != SEEDS:
        raise ValueError("stability requires posterior rows for exactly seeds 7, 13, 23")
    values: dict[int, dict[str, list[float]]] = {
        seed: {"js": [], "displacement": []} for seed in seeds
    }
    joint_js: list[float] = []
    joint_displacement: list[float] = []
    for index, left in enumerate(seeds):
        for right in seeds[index + 1 :]:
            for scale in (1, 2, 4):
                p, q = posteriors[left][scale], posteriors[right][scale]
                mean = 0.5 * (p + q)
                js = 0.5 * (
                    np.sum(p * np.log(np.maximum(p, 1e-12) / np.maximum(mean, 1e-12)), axis=1)
                    + np.sum(q * np.log(np.maximum(q, 1e-12) / np.maximum(mean, 1e-12)), axis=1)
                )
                displacement = np.linalg.norm(
                    np.einsum("bkhw,ki->bhwi", p, offsets[scale])
                    - np.einsum("bkhw,ki->bhwi", q, offsets[scale]),
                    axis=-1,
                )
                row_js = js.reshape(len(js), -1).mean(axis=1)
                row_displacement = displacement.reshape(len(displacement), -1).mean(axis=1)
                for seed in (left, right):
                    values[seed]["js"].extend(row_js.tolist())
                    values[seed]["displacement"].extend(row_displacement.tolist())
                joint_js.extend(row_js.tolist())
                joint_displacement.extend(row_displacement.tolist())

    def summary(js: list[float], displacement: list[float]) -> dict[str, float]:
        return {
            "js_median": float(np.median(js)),
            "js_p95": float(np.percentile(js, 95)),
            "displacement_p95": float(np.percentile(displacement, 95)),
        }

    counts = {f"seed_{seed}_row_pair_scale": len(item["js"]) for seed, item in values.items()}
    counts["joint_row_pair_scale"] = len(joint_js)
    return (
        {seed: summary(item["js"], item["displacement"]) for seed, item in values.items()},
        summary(joint_js, joint_displacement),
        counts,
    )


def control_metrics(data: Mapping[str, Any]) -> dict[str, float | None]:
    """Aggregate actual controls; unavailable correlations stay null with caller reason."""
    base = np.asarray(data["base"], float)
    zoom = [
        (x, np.asarray(data[f"zoom:{x:+.2f}"]["prediction"], float))
        for x in (-0.04, -0.02, 0.02, 0.04)
    ]
    # v4.30 convention: approaching expansion at t2 has negative log-eta.
    analytic = np.concatenate([np.full(len(base), -x) for x, _ in zoom])
    pred = np.concatenate([p for _, p in zoom])
    paired = [p for _, p in zoom]
    reciprocal = np.concatenate([paired[3], paired[2], paired[1], paired[0]])
    denom = max(float(np.dot(analytic, analytic)), 1e-12)
    # Reversal is an endpoint observation, hence support is defined by baseline
    # responses, not by the unrelated synthetic zoom samples.
    coverage = np.abs(base) >= 0.005
    swaps = np.asarray(data["swap:+0.00"]["prediction"], float)
    reverse = np.asarray(data["reverse:+0.00"]["prediction"], float)
    return {
        "analytic_pearson": pearson(analytic, pred),
        "slope": float(np.dot(analytic, pred) / denom),
        "sign_accuracy": float(np.mean(np.sign(analytic) == np.sign(pred))),
        "oddness_median": float(
            np.median(
                np.abs(pred + reciprocal) / np.maximum(np.abs(pred) + np.abs(reciprocal), 1e-12)
            )
        ),
        "oddness_p95": float(
            np.percentile(
                np.abs(pred + reciprocal) / np.maximum(np.abs(pred) + np.abs(reciprocal), 1e-12), 95
            )
        ),
        "identity_p95": float(
            np.percentile(
                np.abs(np.asarray(data["identity:+0.00"]["prediction"], float) - base), 95
            )
        ),
        "translation_leakage_p95": float(
            np.percentile(
                np.abs(np.asarray(data["translation:+0.02"]["prediction"], float) - base), 95
            )
            / max(float(np.percentile(np.abs(pred), 95)), 1e-12)
        ),
        "rotation_leakage_p95": float(
            np.percentile(
                np.abs(np.asarray(data["rotation:+0.02"]["prediction"], float) - base), 95
            )
            / max(float(np.percentile(np.abs(pred), 95)), 1e-12)
        ),
        "swap_corr": pearson(base[coverage], swaps[coverage]),
        "swap_flip": float(np.mean(np.sign(base[coverage]) != np.sign(swaps[coverage])))
        if coverage.any()
        else None,
        "swap_coverage": float(np.mean(coverage)),
        "reverse_response_p95": float(np.percentile(np.abs(reverse), 95)),
        "zero_unknown": float(np.mean(data["zero_event:+0.00"]["unknown"])),
    }


def sequence_swap_gate(
    metrics: Mapping[str, float | None], thresholds: Mapping[str, float] = THRESHOLDS
) -> bool:
    """Fail closed reversal gate for one sequence."""
    values = [metrics.get(key) for key in ("swap_corr", "swap_flip", "swap_coverage")]
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
        return False
    corr, flip, coverage = (float(value) for value in values if value is not None)
    return bool(
        corr <= thresholds["sequence_swap_corr_max"]
        and flip >= thresholds["sequence_swap_flip_min"]
        and coverage >= thresholds["sequence_swap_coverage_min"]
    )


def gate(
    metrics: Mapping[str, float | None], thresholds: Mapping[str, float] = THRESHOLDS
) -> dict[str, bool]:
    """Fail closed individual gates, including every missing/nonfinite observation."""
    required = (
        "js_median",
        "js_p95",
        "displacement_p95",
        "nonempty_fraction",
        "valid_energy_fraction",
        "high_band_cv_p95",
        "effective_rank",
        "analytic_pearson",
        "slope",
        "sign_accuracy",
        "oddness_median",
        "oddness_p95",
        "identity_p95",
        "translation_leakage_p95",
        "rotation_leakage_p95",
        "swap_corr",
        "swap_flip",
        "swap_coverage",
        "zero_unknown",
    )
    finite = all(
        isinstance(metrics.get(key), (int, float))
        and math.isfinite(float(cast(float, metrics[key])))
        for key in required
    )
    if not finite:
        return {"finite": False, "passed": False}
    finite_metrics: dict[str, float] = {
        key: float(cast(float, metrics[key])) for key in required if metrics[key] is not None
    }
    result = {
        "finite": True,
        "stability": finite_metrics["js_median"] <= thresholds["js_median_max"]
        and finite_metrics["js_p95"] <= thresholds["js_p95_max"]
        and finite_metrics["displacement_p95"] <= thresholds["displacement_p95_max"],
        "spectrum": finite_metrics["nonempty_fraction"] >= thresholds["nonempty_min"]
        and finite_metrics["valid_energy_fraction"] >= thresholds["valid_energy_fraction_min"]
        and finite_metrics["high_band_cv_p95"] <= thresholds["high_band_cv_p95_max"]
        and finite_metrics["effective_rank"] >= thresholds["effective_rank_min"],
        "equivariance": finite_metrics["analytic_pearson"] >= thresholds["analytic_pearson_min"]
        and thresholds["slope_min"] <= finite_metrics["slope"] <= thresholds["slope_max"]
        and finite_metrics["sign_accuracy"] >= thresholds["sign_min"]
        and finite_metrics["oddness_median"] <= thresholds["oddness_median_max"]
        and finite_metrics["oddness_p95"] <= thresholds["oddness_p95_max"],
        "invariance": finite_metrics["identity_p95"] <= thresholds["identity_p95_max"]
        and finite_metrics["translation_leakage_p95"] <= thresholds["leakage_p95_max"]
        and finite_metrics["rotation_leakage_p95"] <= thresholds["leakage_p95_max"],
        "reversal": finite_metrics["swap_corr"] <= thresholds["swap_corr_max"]
        and finite_metrics["swap_flip"] >= thresholds["swap_flip_min"]
        and finite_metrics["swap_coverage"] >= thresholds["swap_coverage_min"],
        "zero": finite_metrics["zero_unknown"] == 1.0,
    }
    result["passed"] = all(
        value for key, value in result.items() if key not in {"finite", "passed"}
    )
    return result


def causal_decision(
    *,
    complete: bool = True,
    stability_pass: bool | None = None,
    spectrum_pass: bool | None = None,
    operator_pass: bool | None = None,
    stage2_pass: bool | None = None,
    diagnostic: bool = False,
) -> str:
    """Fixed priority ordering; audit gate failures are findings, not process errors."""
    if diagnostic:
        return "not_issued_diagnostic"
    if not complete:
        return "invalid_incomplete"
    if stability_pass is not True or spectrum_pass is not True:
        return "representation_instability_before_operator"
    if operator_pass is not True:
        return "object_local_correspondence_operator_failure"
    if stage2_pass is not True:
        return "supervised_objective_or_readout_collapse"
    return "failure_localized_to_ttc_magnitude_mapping_or_calibration"
