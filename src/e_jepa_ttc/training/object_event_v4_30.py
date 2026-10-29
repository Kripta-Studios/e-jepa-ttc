"""Locked train-only supervision, stabilization gates, and OOF gates for v4.30."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional

from e_jepa_ttc.models.object_event_v4_30 import ObjectEventV430Output


@dataclass(frozen=True)
class ObjectEventV430LossConfig:
    arm: str = "stable_multiscale_similarity"
    g_weight: float = 1.0
    ell_weight: float = 1.0
    bucket_ratio_weight: float = 0.5
    sign_weight: float = 0.25
    track_weight: float = 0.25
    support_weight: float = 0.25
    distill_weight: float = 0.25
    cycle_weight: float = 0.1
    normal_flow_weight: float = 0.25
    smooth_l1_beta: float = 0.004
    sign_temperature: float = 0.015
    max_abs_g: float = 0.24975
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.arm not in {
            "stable_multiscale_similarity",
            "stable_multiscale_similarity_normal_flow",
        }:
            raise ValueError("v4.30 has exactly two arms")
        if (
            min(
                self.g_weight,
                self.ell_weight,
                self.bucket_ratio_weight,
                self.sign_weight,
                self.track_weight,
                self.support_weight,
                self.distill_weight,
                self.cycle_weight,
                self.normal_flow_weight,
            )
            < 0.0
        ):
            raise ValueError("v4.30 loss weights must be nonnegative")


def g_target(delta_t_s: torch.Tensor, target_ttc_s: torch.Tensor) -> torch.Tensor:
    """Annotation-conditioned train target; it is never a model input."""
    return (delta_t_s / target_ttc_s).clamp(-0.24975, 0.24975)


def ell_target(delta_t_s: torch.Tensor, target_ttc_s: torch.Tensor) -> torch.Tensor:
    return torch.log1p(-g_target(delta_t_s, target_ttc_s))


def posterior_kl(
    consensus: torch.Tensor, student: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    """KL(P_consensus || P_student), preserving every local posterior location."""
    return (
        (consensus * (consensus.clamp_min(epsilon).log() - student.clamp_min(epsilon).log()))
        .sum(dim=1)
        .mean()
    )


def sequence_track_balanced_weights(
    sequence_ids: list[str], track_ids: list[str], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Weights inverse to sequence and track count for deterministic balanced batches/losses."""
    if len(sequence_ids) != len(track_ids):
        raise ValueError("sequence and track ids must align")
    sequence_count = {key: sequence_ids.count(key) for key in set(sequence_ids)}
    track_count = {key: track_ids.count(key) for key in set(track_ids)}
    values = [
        1.0 / (sequence_count[s] * track_count[t])
        for s, t in zip(sequence_ids, track_ids, strict=True)
    ]
    result = torch.tensor(values, device=device, dtype=dtype)
    return result / result.mean().clamp_min(1e-6)


def magnitude_bucket_ratio_loss(
    predicted: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    absolute = target.abs()
    for lo, hi in ((0.01, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, float("inf"))):
        mask = (absolute >= lo) & (absolute < hi)
        if bool(mask.any()):
            ratio = predicted[mask].abs().mean() / target[mask].abs().mean().clamp_min(epsilon)
            pieces.append((ratio - 1.0).square())
    return torch.stack(pieces).mean() if pieces else predicted.sum() * 0.0


def object_event_v4_30_loss(
    output: ObjectEventV430Output,
    delta_t_s: torch.Tensor,
    target_ttc_s: torch.Tensor,
    *,
    consensus_posteriors: Mapping[int, torch.Tensor],
    sequence_ids: list[str],
    track_ids: list[str],
    config: ObjectEventV430LossConfig,
    visible_heights_px: torch.Tensor | None = None,
    boxes_xyxy: torch.Tensor | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Locked annotation-conditioned objective. No label reaches model.forward."""
    if bool(output.unknown.any()):
        raise FloatingPointError(
            "zero-event training row is UNKNOWN and cannot enter a supervised loss"
        )
    target_g = g_target(delta_t_s, target_ttc_s)
    target_ell = torch.log1p(-target_g)
    predicted_g = output.expansion
    balanced = sequence_track_balanced_weights(
        sequence_ids, track_ids, device=predicted_g.device, dtype=predicted_g.dtype
    )
    g = (
        balanced
        * functional.smooth_l1_loss(
            predicted_g, target_g, beta=config.smooth_l1_beta, reduction="none"
        )
    ).mean()
    ell = (
        balanced
        * functional.smooth_l1_loss(
            output.log_eta, target_ell, beta=config.smooth_l1_beta, reduction="none"
        )
    ).mean()
    bucket = magnitude_bucket_ratio_loss(predicted_g, target_g, config.epsilon)
    sign = (
        balanced
        * functional.softplus(
            -torch.where(target_g >= 0, 1.0, -1.0) * predicted_g / config.sign_temperature
        )
    ).mean()
    per_track = []
    point_error = functional.smooth_l1_loss(
        predicted_g, target_g, beta=config.smooth_l1_beta, reduction="none"
    )
    for name in sorted(set(track_ids)):
        index = torch.as_tensor(
            [i for i, key in enumerate(track_ids) if key == name], device=predicted_g.device
        )
        per_track.append((balanced[index] * point_error[index]).mean())
    track = torch.stack(per_track).mean()
    support = output.posterior_variance.mean() + (1.0 - output.correlation_confidence).mean()
    if visible_heights_px is not None and boxes_xyxy is not None:
        if (
            image_height is None
            or image_width is None
            or boxes_xyxy.shape[1:] != (2, 4)
            or visible_heights_px.ndim != 2
            or visible_heights_px.shape[1] != 2
        ):
            raise ValueError("v4.30 support supervision requires t1/t2 boxes and source dimensions")
        b1, b2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1]
        h_ratio = (b1[:, 3] - b1[:, 1]).clamp_min(config.epsilon).log() - (
            b2[:, 3] - b2[:, 1]
        ).clamp_min(config.epsilon).log()
        w_ratio = (b1[:, 2] - b1[:, 0]).clamp_min(config.epsilon).log() - (
            b2[:, 2] - b2[:, 0]
        ).clamp_min(config.epsilon).log()
        vh_ratio = (
            visible_heights_px[:, 0].clamp_min(config.epsilon).log()
            - visible_heights_px[:, 1].clamp_min(config.epsilon).log()
        )
        map_h, map_w = output.foreground_map_t2.shape[-2:]
        c1 = torch.stack(
            (
                (b1[:, 0] + b1[:, 2]) * 0.5 * map_w / image_width,
                (b1[:, 1] + b1[:, 3]) * 0.5 * map_h / image_height,
            ),
            dim=-1,
        )
        c2 = torch.stack(
            (
                (b2[:, 0] + b2[:, 2]) * 0.5 * map_w / image_width,
                (b2[:, 1] + b2[:, 3]) * 0.5 * map_h / image_height,
            ),
            dim=-1,
        )
        mapped = (
            (output.fit_12.matrix @ c2[..., None]).squeeze(-1)
            + output.fit_12.translation
            - (
                (output.fit_12.matrix - torch.eye(2, device=c2.device, dtype=c2.dtype))
                @ output.fit_12.center[..., None]
            ).squeeze(-1)
        )
        center = functional.smooth_l1_loss(mapped, c1, beta=0.02)
        ratio = (
            functional.smooth_l1_loss(output.log_eta, h_ratio, beta=0.02)
            + functional.smooth_l1_loss(output.log_eta, vh_ratio, beta=0.02)
            + functional.smooth_l1_loss(output.log_eta, w_ratio, beta=0.02)
        )
        yy, xx = torch.meshgrid(
            torch.arange(map_h, device=b2.device),
            torch.arange(map_w, device=b2.device),
            indexing="ij",
        )

        def interior(box: torch.Tensor) -> torch.Tensor:
            return (
                (xx[None] >= box[:, 0, None, None] * map_w / image_width)
                & (xx[None] <= box[:, 2, None, None] * map_w / image_width)
                & (yy[None] >= box[:, 1, None, None] * map_h / image_height)
                & (yy[None] <= box[:, 3, None, None] * map_h / image_height)
            ).to(predicted_g.dtype)

        box_bce = functional.binary_cross_entropy(
            output.foreground_map_t1.clamp(config.epsilon, 1 - config.epsilon), interior(b1)
        ) + functional.binary_cross_entropy(
            output.foreground_map_t2.clamp(config.epsilon, 1 - config.epsilon), interior(b2)
        )
        support = support + center + ratio + box_bce
    distill = torch.stack(
        [
            posterior_kl(
                consensus_posteriors[scale],
                output.posteriors_12[scale].probabilities,
                config.epsilon,
            )
            for scale in (1, 2, 4)
        ]
    ).mean()
    cycle = output.cycle_matrix_error.mean() + output.cycle_translation_error.mean()
    normal = output.normal_flow_residual.mean()
    pieces = {
        "g": g,
        "ell": ell,
        "bucket_ratio": bucket,
        "sign": sign,
        "track": track,
        "support": support,
        "distill": distill,
        "cycle": cycle,
        "normal_flow": normal,
        "target_g": target_g,
        "target_ell": target_ell,
    }
    total = (
        config.g_weight * g
        + config.ell_weight * ell
        + config.bucket_ratio_weight * bucket
        + config.sign_weight * sign
        + config.track_weight * track
        + config.support_weight * support
        + config.distill_weight * distill
        + config.cycle_weight * cycle
    )
    if config.arm.endswith("normal_flow"):
        total = total + config.normal_flow_weight * normal
    if not torch.isfinite(total):
        raise FloatingPointError("v4.30 loss became nonfinite; failing closed")
    return total, pieces


def stabilization_gate(
    js_median: float, js_p95: float, expected_displacement_p95: float
) -> dict[str, bool]:
    """Frozen multiseed posterior stability gate before arm fitting."""
    return {
        "js_median": np.isfinite(js_median) and js_median <= 0.02,
        "js_p95": np.isfinite(js_p95) and js_p95 <= 0.08,
        "expected_displacement_p95": np.isfinite(expected_displacement_p95)
        and expected_displacement_p95 <= 0.5,
    }


def oof_gates(metrics: Mapping[str, float]) -> dict[str, bool]:
    """Complete frozen v4.30 promotion gates; missing values fail closed."""
    required = (
        "finite_predictions",
        "finite_posterior_variances",
        "seed_prediction_p95_range",
        "seed_prediction_max_range",
        "sign_disagreement",
        "seed_pearson_range",
        "pearson",
        "log_eta_pearson",
        "minimum_sequence_pearson",
        "negative_accuracy",
        "balanced_sign_accuracy",
        "prediction_std_ratio",
        "calibration_slope",
        "high_bucket_pearson",
        "negative_track_macro_accuracy",
        "minimum_negative_track_accuracy",
        "eligible_negative_track_p10",
        "shuffle_ratio",
        "endpoint_swap_pearson",
        "bottom_support_seed_p95_range",
        "bottom_support_uncertainty_finite",
    )
    if any(key not in metrics or not np.isfinite(float(metrics[key])) for key in required):
        return {"complete_finite": False}
    checks = {
        "complete_finite": bool(metrics["finite_predictions"])
        and bool(metrics["finite_posterior_variances"]),
        "seed_range": float(metrics["seed_prediction_p95_range"]) <= 0.02
        and float(metrics["seed_prediction_max_range"]) <= 0.08
        and float(metrics["sign_disagreement"]) <= 0.02
        and float(metrics["seed_pearson_range"]) <= 0.02,
        "pearson": float(metrics["pearson"]) >= 0.777,
        "log_eta": float(metrics["log_eta_pearson"]) >= 0.758,
        "minimum_sequence": float(metrics["minimum_sequence_pearson"]) >= 0.50,
        "negative": float(metrics["negative_accuracy"]) >= 0.84,
        "balanced_sign": float(metrics["balanced_sign_accuracy"]) >= 0.89,
        "calibration": 0.90 <= float(metrics["prediction_std_ratio"]) <= 1.10
        and 0.90 <= float(metrics["calibration_slope"]) <= 1.10,
        "high_bucket": float(metrics["high_bucket_pearson"]) >= 0.45,
        "negative_tracks": float(metrics["negative_track_macro_accuracy"]) >= 0.857
        and float(metrics["minimum_negative_track_accuracy"]) >= 0.50
        and float(metrics["eligible_negative_track_p10"]) >= 0.65,
        "controls": float(metrics["shuffle_ratio"]) <= 0.50
        and float(metrics["endpoint_swap_pearson"]) <= 0.15,
        "bottom_support": float(metrics["bottom_support_seed_p95_range"]) <= 0.05
        and bool(metrics["bottom_support_uncertainty_finite"]),
    }
    for index in range(4):
        key = f"magnitude_ratio_{index}"
        checks[key] = (
            key in metrics
            and np.isfinite(float(metrics[key]))
            and 0.85 <= float(metrics[key]) <= 1.15
        )
    return checks


def choose_rank_winner(results: Mapping[str, Mapping[str, float]]) -> str:
    return sorted(results, key=lambda name: (-float(results[name].get("pearson", -np.inf)), name))[
        0
    ]


def promoted_champion(results: Mapping[str, Mapping[str, float]]) -> str | None:
    """Rank independently; normal-flow wins only under the preregistered paired gain rule."""
    passed = {name: all(oof_gates(value).values()) for name, value in results.items()}
    if not any(passed.values()):
        return None
    a, b = "stable_multiscale_similarity", "stable_multiscale_similarity_normal_flow"
    if passed.get(a, False) and passed.get(b, False):

        def margin(name: str, threshold: float) -> bool:
            value = results[b].get(name)
            return (
                isinstance(value, (float, int))
                and np.isfinite(float(value))
                and float(value) >= threshold
            )

        gain = margin("paired_sequence_pearson_gain", 0.01)
        second = (
            margin("high_bucket_pearson_gain", 0.10)
            or margin("negative_track_macro_gain", 0.02)
            or margin("shuffle_ratio_reduction", 0.10)
        )
        return b if gain and second else a
    return a if passed.get(a, False) else None


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or not (np.isfinite(left).all() and np.isfinite(right).all()):
        return None
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def compute_oof_metrics(
    *,
    target_g: np.ndarray,
    target_log_eta: np.ndarray,
    prediction: np.ndarray,
    predicted_log_eta: np.ndarray,
    posterior_variance: np.ndarray,
    support: np.ndarray,
    sequence_ids: list[str],
    track_ids: list[str],
    seed_predictions: np.ndarray,
    shuffle_prediction: np.ndarray,
    endpoint_prediction: np.ndarray,
    zero_unknown: np.ndarray,
    zero_prediction: np.ndarray,
    delta_t_s: np.ndarray | None = None,
) -> dict[str, object]:
    """Pure fail-closed OOF metrics; ``None`` means unavailable, never optimistic."""
    arrays = (target_g, target_log_eta, prediction, predicted_log_eta, posterior_variance, support)
    n = len(target_g)
    if any(len(value) != n for value in arrays) or len(sequence_ids) != n or len(track_ids) != n:
        raise ValueError("aligned OOF arrays, sequence ids, and track ids are required")
    if seed_predictions.shape != (3, n):
        raise ValueError("seed_predictions must be [3,N]")
    pearson, log_pearson = _corr(prediction, target_g), _corr(predicted_log_eta, target_log_eta)
    negative, positive = target_g < 0, target_g >= 0

    def accuracy(mask: np.ndarray) -> float | None:
        return (
            float(np.mean(np.sign(prediction[mask]) == np.sign(target_g[mask])))
            if mask.any()
            else None
        )

    per_sequence = {}
    for name in sorted(set(sequence_ids)):
        mask = np.asarray([item == name for item in sequence_ids])
        per_sequence[name] = {
            "pearson": _corr(prediction[mask], target_g[mask]),
            "negative_accuracy": accuracy(mask & negative),
        }
    buckets = {}
    for index, (lo, hi) in enumerate(((0.01, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, np.inf))):
        mask = (np.abs(target_g) >= lo) & (np.abs(target_g) < hi)
        target_mean = float(np.abs(target_g[mask]).mean()) if mask.any() else None
        prediction_mean = float(np.abs(prediction[mask]).mean()) if mask.any() else None
        buckets[str(index)] = {
            "count": int(mask.sum()),
            "pearson": _corr(prediction[mask], target_g[mask]),
            "target_abs_mean": target_mean,
            "prediction_abs_mean": prediction_mean,
            "ratio": prediction_mean / target_mean
            if target_mean and prediction_mean is not None
            else None,
        }
    tracks = {}
    eligible = []
    strata = {"1-3": [], "4-7": [], "8+": []}
    for name in sorted(set(track_ids)):
        mask = np.asarray([item == name for item in track_ids])
        neg = mask & negative
        count = int(neg.sum())
        neg_accuracy = accuracy(neg)
        tracks[name] = {
            "pearson": _corr(prediction[mask], target_g[mask]),
            "negative_count": count,
            "negative_accuracy": neg_accuracy,
        }
        if count >= 4 and neg_accuracy is not None:
            eligible.append(neg_accuracy)
        if count:
            strata["1-3" if count <= 3 else "4-7" if count <= 7 else "8+"].append(neg_accuracy)
    seed_range = np.ptp(seed_predictions, axis=0)
    bottom = support <= np.quantile(support, 0.01)
    unperturbed = pearson
    shuffled = _corr(shuffle_prediction, target_g)
    endpoint = _corr(endpoint_prediction, target_g)
    slope = None
    if (
        np.isfinite(prediction).all()
        and np.isfinite(target_g).all()
        and np.dot(target_g - target_g.mean(), target_g - target_g.mean()) > 1e-12
    ):
        slope = float(
            np.dot(target_g - target_g.mean(), prediction - prediction.mean())
            / np.dot(target_g - target_g.mean(), target_g - target_g.mean())
        )
    positive_accuracy = accuracy(positive)
    negative_accuracy = accuracy(negative)
    seed_correlations = [_corr(row, target_g) for row in seed_predictions]
    result: dict[str, object] = {
        "count": n,
        "pearson": pearson,
        "mae": float(np.mean(np.abs(prediction - target_g)))
        if np.isfinite(prediction).all()
        else None,
        "log_eta_pearson": log_pearson,
        "log_eta_mae": float(np.mean(np.abs(predicted_log_eta - target_log_eta)))
        if np.isfinite(predicted_log_eta).all()
        else None,
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "balanced_sign_accuracy": None
        if positive_accuracy is None or negative_accuracy is None
        else 0.5 * (positive_accuracy + negative_accuracy),
        "minimum_sequence_pearson": None
        if any(
            value["pearson"] is None or not np.isfinite(float(value["pearson"]))
            for value in per_sequence.values()
        )
        else min(float(value["pearson"]) for value in per_sequence.values()),
        "minimum_sequence_negative_accuracy": min(
            (
                value["negative_accuracy"]
                for value in per_sequence.values()
                if value["negative_accuracy"] is not None
            ),
            default=None,
        ),
        "prediction_std_ratio": float(np.std(prediction) / np.std(target_g))
        if np.std(target_g) > 1e-12 and np.isfinite(prediction).all()
        else None,
        "calibration_slope": slope,
        "buckets": buckets,
        "high_bucket_pearson": buckets["3"]["pearson"],
        "mid": float(np.mean(np.abs(predicted_log_eta - target_log_eta)) * 1e4)
        if np.isfinite(predicted_log_eta).all()
        else None,
        "track_macro_pearson": float(
            np.mean([x["pearson"] for x in tracks.values() if x["pearson"] is not None])
        )
        if any(x["pearson"] is not None for x in tracks.values())
        else None,
        "negative_track_macro_accuracy": float(np.mean(eligible)) if eligible else None,
        "minimum_negative_track_accuracy": float(min(eligible)) if eligible else None,
        "eligible_negative_track_p10": float(np.percentile(eligible, 10)) if eligible else None,
        "negative_track_strata": {
            key: {
                "track_count": len(value),
                "mean_accuracy": float(np.mean(value))
                if value and all(x is not None for x in value)
                else None,
            }
            for key, value in strata.items()
        },
        "posterior_variance_finite": bool(np.isfinite(posterior_variance).all()),
        "posterior_variance_quantiles": np.quantile(posterior_variance, [0.05, 0.5, 0.95]).tolist()
        if np.isfinite(posterior_variance).all()
        else None,
        "support_quantiles": np.quantile(support, [0.01, 0.5, 0.99]).tolist()
        if np.isfinite(support).all()
        else None,
        "bottom_support_coverage": float(np.isfinite(prediction[bottom]).mean())
        if bottom.any()
        else None,
        "bottom_support_uncertainty_finite": bool(np.isfinite(posterior_variance[bottom]).all())
        if bottom.any()
        else False,
        "bottom_support_seed_p95_range": float(np.percentile(seed_range[bottom], 95))
        if bottom.any()
        else None,
        "seed_prediction_p95_range": float(np.percentile(seed_range, 95)),
        "seed_prediction_max_range": float(seed_range.max()),
        "sign_disagreement": float(
            np.mean(
                np.any(
                    np.sign(seed_predictions) != np.sign(np.median(seed_predictions, axis=0))[None],
                    axis=0,
                )
            )
        ),
        "seed_pearson_range": None
        if any(value is None for value in seed_correlations)
        else float(np.ptp([float(value) for value in seed_correlations if value is not None])),
        "shuffle_pearson": shuffled,
        "endpoint_swap_pearson": endpoint,
        "endpoint_swap_pearson_abs": None if endpoint is None else float(abs(endpoint)),
        "shuffle_ratio": None
        if unperturbed is None or abs(unperturbed) < 1e-12 or shuffled is None
        else float(abs(shuffled) / abs(unperturbed)),
        "zero_event_unknown": bool(zero_unknown.all()),
        "zero_event_physical_nan": bool(np.isnan(zero_prediction).all()),
        "per_sequence": per_sequence,
        "per_track": tracks,
    }
    if delta_t_s is not None:
        result["rte"] = float(
            np.mean(np.abs(prediction - target_g) / np.maximum(np.abs(target_g), 1e-6))
        )
        result["fr"] = float(np.mean(np.sign(prediction) != np.sign(target_g)))
    return result


def gate_constituents_and_median(
    rows: list[Mapping[str, object]], median: Mapping[str, object]
) -> dict[str, bool]:
    """Every seed and the median must independently clear all applicable gates."""

    def passed(row: Mapping[str, object]) -> bool:
        return all(
            oof_gates(
                {key: value for key, value in row.items() if isinstance(value, (float, int, bool))}
            ).values()
        )

    return {
        "constituents": len(rows) == 3 and all(passed(row) for row in rows),
        "median": passed(median),
    }


__all__ = [
    "ObjectEventV430LossConfig",
    "compute_oof_metrics",
    "gate_constituents_and_median",
    "choose_rank_winner",
    "ell_target",
    "g_target",
    "magnitude_bucket_ratio_loss",
    "object_event_v4_30_loss",
    "oof_gates",
    "posterior_kl",
    "promoted_champion",
    "sequence_track_balanced_weights",
    "stabilization_gate",
]
