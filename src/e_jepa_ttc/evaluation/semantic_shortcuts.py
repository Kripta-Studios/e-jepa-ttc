"""Controlled audit for semantic shortcut allocation in predictive embeddings.

The benchmark plants a sequence-constant watermark and a partially predictable
TTC-like expansion state. It distinguishes statistical non-collapse from useful
dynamic content without reading a real dataset. It is a falsifier for objective
choices, not benchmark or SOTA evidence.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional

BENCHMARK_ARMS: Final[tuple[str, ...]] = (
    "repo_variance",
    "repo_visreg",
    "temporal_residual",
    "r2_rate_dependence",
    "residual_r2",
)


@dataclass(frozen=True)
class SemanticShortcutConfig:
    """Fixed-budget configuration for the controlled shortcut audit."""

    train_sequences: int = 48
    test_sequences: int = 24
    steps_per_sequence: int = 14
    input_dim: int = 48
    hidden_dim: int = 96
    latent_dim: int = 16
    block_size: int = 4
    shortcut_bits: int = 12
    shortcut_mode: str = "sequence"
    shortcut_strength: float = 2.5
    dynamic_strength: float = 1.0
    observation_noise_std: float = 0.08
    process_noise_std: float = 0.004
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ema_momentum: float = 0.99
    variance_floor: float = 0.35
    regularizer_weight: float = 0.25
    rate_capacity_nats: float = 3.0
    rate_weight: float = 0.02
    dependence_weight: float = 0.05
    rff_features: int = 24
    ridge_alpha: float = 1e-2

    def validate(self) -> None:
        """Fail before allocation when the experiment contract is inconsistent."""

        positive_ints = {
            "train_sequences": self.train_sequences,
            "test_sequences": self.test_sequences,
            "steps_per_sequence": self.steps_per_sequence,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "block_size": self.block_size,
            "shortcut_bits": self.shortcut_bits,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "rff_features": self.rff_features,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"Positive integer fields required: {invalid}.")
        if self.latent_dim % self.block_size:
            raise ValueError("latent_dim must be divisible by block_size.")
        if self.shortcut_mode not in {"sequence", "frame"}:
            raise ValueError("shortcut_mode must be 'sequence' or 'frame'.")
        if not 0.0 <= self.ema_momentum <= 1.0:
            raise ValueError("ema_momentum must lie in [0,1].")
        nonnegative = (
            self.observation_noise_std,
            self.process_noise_std,
            self.weight_decay,
            self.regularizer_weight,
            self.rate_weight,
            self.dependence_weight,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("Noise and loss weights must be non-negative.")
        if self.learning_rate <= 0.0 or self.rate_capacity_nats <= 0.0:
            raise ValueError("learning_rate and rate_capacity_nats must be positive.")


@dataclass(frozen=True)
class _SyntheticPairs:
    current: torch.Tensor
    future: torch.Tensor
    dynamic: np.ndarray
    log_ttc: np.ndarray
    shortcut: np.ndarray
    sequence_index: np.ndarray


class _ShortcutEncoder(nn.Module):
    def __init__(self, config: SemanticShortcutConfig, *, stochastic: bool) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.mean = nn.Linear(config.hidden_dim, config.latent_dim)
        self.log_variance = nn.Linear(config.hidden_dim, config.latent_dim) if stochastic else None

    def forward(
        self, values: torch.Tensor, *, sample: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(values)
        mean = self.mean(features)
        if self.log_variance is None:
            return mean, torch.zeros_like(mean), mean
        log_variance = self.log_variance(features).clamp(-6.0, 3.0)
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean) if sample else mean
        return mean, log_variance, latent


class _ShortcutModel(nn.Module):
    rff_w: torch.Tensor
    rff_b: torch.Tensor

    def __init__(self, config: SemanticShortcutConfig, *, stochastic: bool, seed: int) -> None:
        super().__init__()
        self.config = config
        self.online = _ShortcutEncoder(config, stochastic=stochastic)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.target.eval()
        self.predictor = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        block_count = config.latent_dim // config.block_size
        generator = torch.Generator().manual_seed(seed + 40_003)
        self.register_buffer(
            "rff_w",
            torch.randn(
                block_count,
                config.block_size,
                config.rff_features,
                generator=generator,
            ),
        )
        self.register_buffer(
            "rff_b",
            2.0 * math.pi * torch.rand(block_count, config.rff_features, generator=generator),
        )

    @torch.no_grad()
    def update_target(self) -> None:
        momentum = self.config.ema_momentum
        for target, online in zip(self.target.parameters(), self.online.parameters(), strict=True):
            target.mul_(momentum).add_(online.detach(), alpha=1.0 - momentum)

    def nonlinear_dependence(self, latent: torch.Tensor) -> torch.Tensor:
        """Approximate cross-block HSIC with fixed random Fourier features."""

        blocks = latent.reshape(
            latent.shape[0],
            self.config.latent_dim // self.config.block_size,
            self.config.block_size,
        )
        mapped = math.sqrt(2.0 / self.config.rff_features) * torch.cos(
            torch.einsum("nkd,kdf->nkf", blocks, self.rff_w) + self.rff_b[None]
        )
        mapped = mapped - mapped.mean(dim=0, keepdim=True)
        losses = []
        for left in range(mapped.shape[1]):
            for right in range(left + 1, mapped.shape[1]):
                cross = mapped[:, left].T @ mapped[:, right] / max(mapped.shape[0] - 1, 1)
                losses.append(cross.square().mean())
        return torch.stack(losses).mean() if losses else latent.sum() * 0.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _observation(
    *,
    bits: np.ndarray,
    phase: float,
    speed: float,
    shortcut_projection: np.ndarray,
    dynamic_projection: np.ndarray,
    rng: np.random.Generator,
    config: SemanticShortcutConfig,
) -> np.ndarray:
    speed_scaled = (speed - 0.035) / 0.012
    dynamic = np.array(
        [
            phase,
            speed_scaled,
            math.sin(math.pi * phase),
            math.cos(math.pi * phase),
            phase * speed_scaled,
            1.0 - phase,
        ],
        dtype=np.float32,
    )
    noise = rng.normal(0.0, config.observation_noise_std, size=config.input_dim)
    return (
        config.shortcut_strength * np.tanh(bits @ shortcut_projection)
        + config.dynamic_strength * np.tanh(dynamic @ dynamic_projection)
        + noise
    ).astype(np.float32)


def _make_pairs(
    *,
    sequence_count: int,
    seed: int,
    sequence_offset: int,
    config: SemanticShortcutConfig,
) -> _SyntheticPairs:
    mixing_rng = np.random.default_rng(20_260_802)
    shortcut_projection = mixing_rng.normal(
        0.0,
        1.0 / math.sqrt(config.shortcut_bits),
        size=(config.shortcut_bits, config.input_dim),
    ).astype(np.float32)
    dynamic_projection = mixing_rng.normal(
        0.0, 1.0 / math.sqrt(6), size=(6, config.input_dim)
    ).astype(np.float32)
    rng = np.random.default_rng(seed)
    current_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    dynamics: list[tuple[float, float]] = []
    log_ttc: list[float] = []
    shortcuts: list[np.ndarray] = []
    sequence_indices: list[int] = []
    for local_sequence in range(sequence_count):
        bits = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), config.shortcut_bits)
        phase = float(rng.uniform(0.05, 0.22))
        speed = float(rng.uniform(0.022, 0.048))
        for _step in range(config.steps_per_sequence):
            future_bits = (
                rng.choice(np.array([-1.0, 1.0], dtype=np.float32), config.shortcut_bits)
                if config.shortcut_mode == "frame"
                else bits
            )
            next_speed = float(
                np.clip(speed + rng.normal(0.0, config.process_noise_std), 0.015, 0.055)
            )
            next_phase = float(np.clip(phase + next_speed, 0.0, 0.96))
            current_rows.append(
                _observation(
                    bits=bits,
                    phase=phase,
                    speed=speed,
                    shortcut_projection=shortcut_projection,
                    dynamic_projection=dynamic_projection,
                    rng=rng,
                    config=config,
                )
            )
            future_rows.append(
                _observation(
                    bits=future_bits,
                    phase=next_phase,
                    speed=next_speed,
                    shortcut_projection=shortcut_projection,
                    dynamic_projection=dynamic_projection,
                    rng=rng,
                    config=config,
                )
            )
            dynamics.append((phase, (speed - 0.035) / 0.012))
            log_ttc.append(math.log(max((1.0 - phase) / max(speed, 1e-4), 1e-4)))
            shortcuts.append(bits.copy())
            sequence_indices.append(sequence_offset + local_sequence)
            phase, speed, bits = next_phase, next_speed, future_bits
    return _SyntheticPairs(
        current=torch.from_numpy(np.stack(current_rows)),
        future=torch.from_numpy(np.stack(future_rows)),
        dynamic=np.asarray(dynamics, dtype=np.float64),
        log_ttc=np.asarray(log_ttc, dtype=np.float64)[:, None],
        shortcut=np.stack(shortcuts).astype(np.float64),
        sequence_index=np.asarray(sequence_indices, dtype=np.int64),
    )


def _variance_loss(latent: torch.Tensor, floor: float) -> torch.Tensor:
    std = torch.sqrt(latent.float().var(dim=0, unbiased=False) + 1e-4)
    return functional.relu(floor - std).mean()


def _covariance_loss(latent: torch.Tensor) -> torch.Tensor:
    centered = latent.float() - latent.float().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag_embed(torch.diagonal(covariance))
    return off_diagonal.square().mean()


def _sliced_gaussian_loss(latent: torch.Tensor, projection_count: int = 24) -> torch.Tensor:
    rows = latent.float()
    centered = rows - rows.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False, keepdim=True) + 1e-4)
    normalized = centered / std.detach().clamp_min(1e-4)
    projections = functional.normalize(
        torch.randn(rows.shape[1], projection_count, device=rows.device), dim=0
    )
    projected = torch.sort(normalized @ projections, dim=0).values
    quantiles = (torch.arange(rows.shape[0], device=rows.device) + 0.5) / rows.shape[0]
    gaussian = math.sqrt(2.0) * torch.special.erfinv(2.0 * quantiles - 1.0)
    return (
        rows.mean(dim=0).square().mean()
        + (1.0 - std).square().mean()
        + (projected - gaussian[:, None]).square().mean()
    )


def _block_rate_penalty(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    *,
    config: SemanticShortcutConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    kl_dimension = 0.5 * (mean.square() + log_variance.exp() - log_variance - 1.0)
    rates = (
        kl_dimension.reshape(
            mean.shape[0],
            config.latent_dim // config.block_size,
            config.block_size,
        )
        .sum(dim=2)
        .mean(dim=0)
    )
    penalty = functional.relu(rates - config.rate_capacity_nats).square().mean()
    return penalty, rates.mean()


def _objective(
    *,
    model: _ShortcutModel,
    current: torch.Tensor,
    future: torch.Tensor,
    arm: str,
    config: SemanticShortcutConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    stochastic = arm in {"r2_rate_dependence", "residual_r2"}
    current_mean, current_logvar, current_z = model.online(current, sample=stochastic)
    future_mean, future_logvar, _ = model.online(future, sample=False)
    with torch.no_grad():
        target_current, _, _ = model.target(current, sample=False)
        target_future, _, _ = model.target(future, sample=False)
    residual = arm in {"temporal_residual", "residual_r2"}
    target = target_future - target_current if residual else target_future
    prediction = model.predictor(current_z)
    prediction_loss = functional.smooth_l1_loss(prediction, target)
    regularized = future_mean - current_mean if residual else current_mean
    if arm == "repo_visreg":
        regularization = _sliced_gaussian_loss(regularized)
    elif stochastic:
        block_count = config.latent_dim // config.block_size
        regularization = torch.stack(
            [
                _sliced_gaussian_loss(
                    regularized[:, index * config.block_size : (index + 1) * config.block_size],
                    projection_count=8,
                )
                for index in range(block_count)
            ]
        ).mean()
    else:
        regularization = _variance_loss(regularized, config.variance_floor)
        regularization = regularization + _covariance_loss(regularized)
    rate_penalty = prediction.sum() * 0.0
    mean_rate = prediction.sum() * 0.0
    dependence = prediction.sum() * 0.0
    if stochastic:
        current_rate, current_mean_rate = _block_rate_penalty(
            current_mean, current_logvar, config=config
        )
        future_rate, future_mean_rate = _block_rate_penalty(
            future_mean, future_logvar, config=config
        )
        rate_penalty = 0.5 * (current_rate + future_rate)
        mean_rate = 0.5 * (current_mean_rate + future_mean_rate)
        dependence = model.nonlinear_dependence(current_mean)
    loss = (
        prediction_loss
        + config.regularizer_weight * regularization
        + config.rate_weight * rate_penalty
        + config.dependence_weight * dependence
    )
    return loss, {
        "loss": float(loss.detach().cpu()),
        "prediction_loss": float(prediction_loss.detach().cpu()),
        "regularization_loss": float(regularization.detach().cpu()),
        "rate_penalty": float(rate_penalty.detach().cpu()),
        "mean_block_rate_nats": float(mean_rate.detach().cpu()),
        "dependence_loss": float(dependence.detach().cpu()),
    }


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    x_mean = train_x.mean(axis=0, keepdims=True)
    x_std = train_x.std(axis=0, keepdims=True).clip(1e-6)
    y_mean = train_y.mean(axis=0, keepdims=True)
    x_train = (train_x - x_mean) / x_std
    x_test = (test_x - x_mean) / x_std
    weights = np.linalg.solve(
        x_train.T @ x_train + alpha * np.eye(x_train.shape[1]),
        x_train.T @ (train_y - y_mean),
    )
    return x_test @ weights + y_mean


def _r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    numerator = float(np.square(target - prediction).sum())
    denominator = float(np.square(target - target.mean(axis=0, keepdims=True)).sum())
    return float(1.0 - numerator / max(denominator, 1e-12))


def _effective_rank(values: np.ndarray) -> float:
    singular = np.linalg.svd(values - values.mean(axis=0, keepdims=True), compute_uv=False)
    mass = float(singular.sum())
    if mass <= 1e-12:
        return 0.0
    probability = singular / mass
    return float(np.exp(-(probability * np.log(np.clip(probability, 1e-12, None))).sum()))


def _tortuosity(latent: np.ndarray, sequence_index: np.ndarray) -> float:
    values = []
    for sequence in np.unique(sequence_index):
        trajectory = latent[sequence_index == sequence]
        path = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
        values.append(path / max(chord, 1e-8))
    return float(np.mean(values))


def _probe_metrics(
    *,
    train_latent: np.ndarray,
    test_latent: np.ndarray,
    train: _SyntheticPairs,
    test: _SyntheticPairs,
    config: SemanticShortcutConfig,
) -> dict[str, float]:
    dynamic_prediction = _ridge_predict(
        train_latent, train.dynamic, test_latent, alpha=config.ridge_alpha
    )
    ttc_prediction = _ridge_predict(
        train_latent, train.log_ttc, test_latent, alpha=config.ridge_alpha
    )
    shortcut_prediction = _ridge_predict(
        train_latent, train.shortcut, test_latent, alpha=config.ridge_alpha
    )
    shortcut_accuracy = float(
        np.mean(np.where(shortcut_prediction >= 0.0, 1.0, -1.0) == test.shortcut)
    )
    joint_excess = max((shortcut_accuracy - 0.5) * 2.0, 1e-8)
    block_excess = 0.0
    block_accuracies = []
    for start in range(0, config.latent_dim, config.block_size):
        prediction = _ridge_predict(
            train_latent[:, start : start + config.block_size],
            train.shortcut,
            test_latent[:, start : start + config.block_size],
            alpha=config.ridge_alpha,
        )
        accuracy = float(np.mean(np.where(prediction >= 0.0, 1.0, -1.0) == test.shortcut))
        block_accuracies.append(accuracy)
        block_excess += max((accuracy - 0.5) * 2.0, 0.0)
    dynamic_r2 = _r2_score(test.dynamic, dynamic_prediction)
    return {
        "dynamic_r2": dynamic_r2,
        "log_ttc_mae": float(np.abs(test.log_ttc - ttc_prediction).mean()),
        "shortcut_bit_accuracy": shortcut_accuracy,
        "shortcut_duplication_ratio": float(block_excess / joint_excess),
        "mean_block_shortcut_accuracy": float(np.mean(block_accuracies)),
        "effective_rank": _effective_rank(test_latent),
        "collapsed_dimension_fraction": float(np.mean(test_latent.std(axis=0) < 0.05)),
        "latent_tortuosity": _tortuosity(test_latent, test.sequence_index),
        "shortcut_to_dynamic_ratio": float(
            max((shortcut_accuracy - 0.5) * 2.0, 0.0) / max(dynamic_r2, 1e-6)
        ),
    }


def _train_arm(
    *,
    arm: str,
    seed: int,
    train: _SyntheticPairs,
    test: _SyntheticPairs,
    config: SemanticShortcutConfig,
    device: torch.device,
) -> dict[str, Any]:
    if arm not in BENCHMARK_ARMS:
        raise ValueError(f"Unknown arm {arm!r}.")
    _set_seed(seed)
    stochastic = arm in {"r2_rate_dependence", "residual_r2"}
    model = _ShortcutModel(config, stochastic=stochastic, seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        [*model.online.parameters(), *model.predictor.parameters()],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed + 991)
    last_metrics: dict[str, float] = {}
    batch_count = math.ceil(train.current.shape[0] / config.batch_size)
    for _epoch in range(config.epochs):
        order = torch.randperm(train.current.shape[0], generator=generator)
        rows = []
        for batch_index in range(batch_count):
            indices = order[batch_index * config.batch_size : (batch_index + 1) * config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _objective(
                model=model,
                current=train.current[indices].to(device),
                future=train.future[indices].to(device),
                arm=arm,
                config=config,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss for {arm}, seed {seed}.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.online.parameters(), max_norm=5.0)
            optimizer.step()
            model.update_target()
            rows.append(metrics)
        last_metrics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    model.eval()
    with torch.no_grad():
        train_latent = model.online(train.current.to(device), sample=False)[0].cpu().numpy()
        test_latent = model.online(test.current.to(device), sample=False)[0].cpu().numpy()
    return {
        "arm": arm,
        "seed": seed,
        "epochs": config.epochs,
        "train_pair_count": int(train.current.shape[0]),
        "test_pair_count": int(test.current.shape[0]),
        "last_train": last_metrics,
        "probes": _probe_metrics(
            train_latent=train_latent,
            test_latent=test_latent,
            train=train,
            test=test,
            config=config,
        ),
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    result = {}
    for arm in BENCHMARK_ARMS:
        selected = [run for run in runs if run["arm"] == arm]
        result[arm] = {}
        for metric in selected[0]["probes"]:
            values = np.asarray([run["probes"][metric] for run in selected])
            result[arm][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
    return result


def _decision(aggregate: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    baseline = aggregate["repo_variance"]
    combined = aggregate["residual_r2"]
    residual = aggregate["temporal_residual"]
    baseline_gates = {
        "shortcut_accuracy_at_least_0_75": baseline["shortcut_bit_accuracy"]["mean"] >= 0.75,
        "dynamic_r2_at_most_0_60": baseline["dynamic_r2"]["mean"] <= 0.60,
        "collapsed_fraction_at_most_0_20": baseline["collapsed_dimension_fraction"]["mean"] <= 0.20,
    }
    promotion_gates = {
        "dynamic_r2_gain_at_least_0_15": combined["dynamic_r2"]["mean"]
        >= baseline["dynamic_r2"]["mean"] + 0.15,
        "log_ttc_mae_reduction_at_least_15pct": combined["log_ttc_mae"]["mean"]
        <= 0.85 * baseline["log_ttc_mae"]["mean"],
        "shortcut_duplication_reduction_at_least_20pct": combined["shortcut_duplication_ratio"][
            "mean"
        ]
        <= 0.80 * baseline["shortcut_duplication_ratio"]["mean"],
        "collapsed_fraction_at_most_0_20": combined["collapsed_dimension_fraction"]["mean"] <= 0.20,
    }
    residual_promotion_gates = {
        "dynamic_r2_gain_at_least_0_15": residual["dynamic_r2"]["mean"]
        >= baseline["dynamic_r2"]["mean"] + 0.15,
        "log_ttc_mae_reduction_at_least_15pct": residual["log_ttc_mae"]["mean"]
        <= 0.85 * baseline["log_ttc_mae"]["mean"],
        "shortcut_accuracy_reduction_at_least_0_10": residual["shortcut_bit_accuracy"]["mean"]
        <= baseline["shortcut_bit_accuracy"]["mean"] - 0.10,
        "effective_rank_at_least_4": residual["effective_rank"]["mean"] >= 4.0,
        "collapsed_fraction_at_most_0_20": residual["collapsed_dimension_fraction"]["mean"] <= 0.20,
    }
    simplicity_gates = {
        "residual_within_0_03_dynamic_r2_of_combined": residual["dynamic_r2"]["mean"]
        >= combined["dynamic_r2"]["mean"] - 0.03,
        "residual_ttc_mae_no_more_than_5pct_worse": residual["log_ttc_mae"]["mean"]
        <= 1.05 * combined["log_ttc_mae"]["mean"],
    }
    exposed = all(baseline_gates.values())
    r2_passes = exposed and all(promotion_gates.values())
    residual_passes = exposed and all(residual_promotion_gates.values())
    residual_preferred = residual_passes and (not r2_passes or all(simplicity_gates.values()))
    if not exposed:
        verdict = "inconclusive_synthetic_fixture_did_not_expose_predeclared_failure"
    elif residual_preferred:
        verdict = "reject_full_r2_prefer_temporal_residual_on_synthetic_gate"
    elif r2_passes:
        verdict = "r2_lite_supported_on_synthetic_only_requires_real_highres_gate"
    else:
        verdict = "r2_lite_rejected_by_predeclared_synthetic_gate"
    return {
        "verdict": verdict,
        "benchmark_exposes_semantic_shortcut": exposed,
        "r2_lite_passes_synthetic_gate": r2_passes,
        "temporal_residual_passes_synthetic_gate": residual_passes,
        "temporal_residual_preferred_for_simplicity": residual_preferred,
        "baseline_failure_gates": baseline_gates,
        "combined_promotion_gates": promotion_gates,
        "temporal_residual_promotion_gates": residual_promotion_gates,
        "residual_simplicity_gate": simplicity_gates,
        "scope": "Synthetic mechanistic evidence only; not eAP/EvTTC or SOTA evidence.",
    }


def run_semantic_shortcut_benchmark(
    *,
    config: SemanticShortcutConfig,
    seeds: tuple[int, ...] = (7, 13, 23),
    device_name: str = "auto",
) -> dict[str, Any]:
    """Run all predeclared arms and return a compact, serializable audit."""

    config.validate()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty tuple of unique integers.")
    resolved = "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    device = torch.device(resolved)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    train = _make_pairs(
        sequence_count=config.train_sequences,
        seed=91_001,
        sequence_offset=0,
        config=config,
    )
    test = _make_pairs(
        sequence_count=config.test_sequences,
        seed=92_003,
        sequence_offset=config.train_sequences,
        config=config,
    )
    started = time.perf_counter()
    runs = [
        _train_arm(
            arm=arm,
            seed=seed,
            train=train,
            test=test,
            config=config,
            device=device,
        )
        for arm in BENCHMARK_ARMS
        for seed in seeds
    ]
    aggregate = _aggregate(runs)
    return {
        "artifact_type": "jepa_semantic_shortcut_benchmark_v1",
        "status": "complete",
        "evidence_scope": "synthetic_mechanistic_only",
        "uses_real_dataset": False,
        "uses_ttc_labels_for_representation_training": False,
        "config": asdict(config),
        "seeds": list(seeds),
        "arms": list(BENCHMARK_ARMS),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "elapsed_seconds": float(time.perf_counter() - started),
        "runs": runs,
        "aggregate": aggregate,
        "decision": _decision(aggregate),
    }


def assess_eap_ssl_health(payload: dict[str, Any], *, embedding_dim: int = 192) -> dict[str, Any]:
    """Classify rank evidence without pretending that it proves semantic collapse."""

    history = payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("eAP SSL artifact must contain non-empty history.")
    last = history[-1]
    if not isinstance(last, dict) or not isinstance(last.get("validation"), dict):
        raise ValueError("eAP SSL artifact has no final validation metrics.")
    validation = last["validation"]
    required = (
        "context_effective_rank",
        "pred_effective_rank",
        "target_effective_rank",
        "context_collapsed_dimension_fraction",
    )
    missing = [key for key in required if key not in validation]
    if missing:
        raise ValueError(f"eAP SSL validation metrics missing: {missing}.")
    ranks = {
        name: float(validation[f"{name}_effective_rank"]) for name in ("context", "pred", "target")
    }
    collapsed = float(validation["context_collapsed_dimension_fraction"])
    return {
        "embedding_dim": embedding_dim,
        "effective_rank": ranks,
        "effective_rank_fraction": {name: value / embedding_dim for name, value in ranks.items()},
        "context_collapsed_dimension_fraction": collapsed,
        "statistical_collapse_guard_triggered": collapsed > 0.80,
        "rank_deficiency_warning": min(ranks["context"], ranks["pred"]) / embedding_dim < 0.10,
        "semantic_shortcut_confirmed": False,
        "semantic_diagnosis": (
            "Unavailable: the compact artifact has aggregate rank/std metrics but no "
            "embeddings paired with nuisance and dynamic labels."
        ),
    }


__all__ = [
    "BENCHMARK_ARMS",
    "SemanticShortcutConfig",
    "assess_eap_ssl_health",
    "run_semantic_shortcut_benchmark",
]
