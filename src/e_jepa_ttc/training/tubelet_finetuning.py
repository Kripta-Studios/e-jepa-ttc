"""Stable optimizer and prediction-health utilities for Tubelet TTC fine-tuning.

The supervised TTC trainer has three modules with very different adaptation
needs: a transferred visual-temporal backbone, a randomly initialized pooling
readout, and a randomly initialized TTC head.  This module keeps those parameter
families explicit, supports a readout-only warm-up without rebuilding the
optimizer, and prevents a near-constant predictor from being promoted merely
because its aggregate validation metric is finite.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn


class TubeletModelProtocol(Protocol):
    """Structural protocol required by the fine-tuning helpers."""

    patch_embed: nn.Module
    spatial: nn.Module
    merge: nn.Module | None
    temporal: nn.Module
    final_norm: nn.Module
    query_tokens: nn.Parameter | None
    query_attention: nn.Module | None
    ttc_head: nn.Module
    collision_head: nn.Module

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Any: ...  # noqa: ANN401


@dataclass(frozen=True)
class TubeletOptimizationConfig:
    """Resolved optimizer, warm-up, and collapse-gate controls."""

    backbone_learning_rate: float
    pooling_learning_rate: float
    head_learning_rate: float
    warmup_pooling_learning_rate: float
    warmup_head_learning_rate: float
    backbone_weight_decay: float
    readout_weight_decay: float
    readout_warmup_optimizer_steps: int
    min_prediction_std_ratio: float
    collapse_patience: int

    def __post_init__(self) -> None:
        learning_rates = (
            self.backbone_learning_rate,
            self.pooling_learning_rate,
            self.head_learning_rate,
            self.warmup_pooling_learning_rate,
            self.warmup_head_learning_rate,
        )
        if min(learning_rates) < 0.0:
            raise ValueError("Tubelet learning rates must be non-negative")
        if self.backbone_weight_decay < 0.0 or self.readout_weight_decay < 0.0:
            raise ValueError("Tubelet weight decay values must be non-negative")
        if self.readout_warmup_optimizer_steps < 0:
            raise ValueError("readout_warmup_optimizer_steps must be non-negative")
        if self.min_prediction_std_ratio < 0.0:
            raise ValueError("min_prediction_std_ratio must be non-negative")
        if self.collapse_patience < 0:
            raise ValueError("collapse_patience must be non-negative")


@dataclass(frozen=True)
class TrainEpochResult:
    """Training result that preserves optimizer-step semantics."""

    mean_loss: float
    optimizer_steps: int
    final_optimizer_step: int
    optimization_phase: str
    group_learning_rates: dict[str, float]


class PredictionCollapseError(RuntimeError):
    """Raised when no non-collapsed validation checkpoint can be selected."""


def resolve_optimization_config(train_config: object) -> TubeletOptimizationConfig:
    """Resolve v2 controls while preserving legacy single-LR configurations."""

    learning_rate = float(getattr(train_config, "learning_rate"))
    weight_decay = float(getattr(train_config, "weight_decay"))

    def optional_float(name: str, fallback: float) -> float:
        value = getattr(train_config, name, None)
        return fallback if value is None else float(value)

    return TubeletOptimizationConfig(
        backbone_learning_rate=optional_float("backbone_learning_rate", learning_rate),
        pooling_learning_rate=optional_float("pooling_learning_rate", learning_rate),
        head_learning_rate=optional_float("head_learning_rate", learning_rate),
        warmup_pooling_learning_rate=optional_float(
            "warmup_pooling_learning_rate",
            optional_float("pooling_learning_rate", learning_rate),
        ),
        warmup_head_learning_rate=optional_float(
            "warmup_head_learning_rate",
            optional_float("head_learning_rate", learning_rate),
        ),
        backbone_weight_decay=optional_float("backbone_weight_decay", weight_decay),
        readout_weight_decay=optional_float("readout_weight_decay", weight_decay),
        readout_warmup_optimizer_steps=int(
            getattr(train_config, "readout_warmup_optimizer_steps", 0)
        ),
        min_prediction_std_ratio=float(
            getattr(train_config, "min_prediction_std_ratio", 0.0)
        ),
        collapse_patience=int(getattr(train_config, "collapse_patience", 0)),
    )


def _parameter_names_by_prefix(
    model: TubeletModelProtocol,
    prefixes: Sequence[str],
    exact_names: Sequence[str] = (),
) -> list[tuple[str, nn.Parameter]]:
    result: list[tuple[str, nn.Parameter]] = []
    exact = set(exact_names)
    for name, parameter in model.named_parameters():
        if name in exact or any(name.startswith(prefix) for prefix in prefixes):
            result.append((name, parameter))
    return result


def split_parameter_groups(
    model: TubeletModelProtocol,
) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Return disjoint optimizer families and reject accidental omissions.

    ``collision_head`` is intentionally excluded: the current downstream loss
    contains no collision target, so optimizing it would only apply weight decay
    to an unsupervised task head.
    """

    groups = {
        "backbone": _parameter_names_by_prefix(
            model,
            ("patch_embed.", "spatial.", "merge.", "temporal.", "final_norm."),
        ),
        "pooling": _parameter_names_by_prefix(
            model,
            ("query_attention.",),
            exact_names=("query_tokens",),
        ),
        "ttc_head": _parameter_names_by_prefix(model, ("ttc_head.",)),
    }

    optimized_names = [name for values in groups.values() for name, _ in values]
    if len(optimized_names) != len(set(optimized_names)):
        raise RuntimeError("Tubelet optimizer parameter groups overlap")

    expected = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("collision_head.")
    }
    received = set(optimized_names)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise RuntimeError(
            "Tubelet optimizer groups are not exhaustive: "
            f"missing={missing!r}; extra={extra!r}"
        )
    return groups


def optimizer_group_manifest(
    groups: Mapping[str, Sequence[tuple[str, nn.Parameter]]],
    config: TubeletOptimizationConfig,
) -> dict[str, Any]:
    """Create a stable, serializable optimizer identity for resume checks."""

    learning_rates = {
        "backbone": {
            "warmup": 0.0,
            "finetune": config.backbone_learning_rate,
        },
        "pooling": {
            "warmup": config.warmup_pooling_learning_rate,
            "finetune": config.pooling_learning_rate,
        },
        "ttc_head": {
            "warmup": config.warmup_head_learning_rate,
            "finetune": config.head_learning_rate,
        },
    }
    weight_decays = {
        "backbone": config.backbone_weight_decay,
        "pooling": config.readout_weight_decay,
        "ttc_head": config.readout_weight_decay,
    }
    entries: list[dict[str, Any]] = []
    for name in ("backbone", "pooling", "ttc_head"):
        values = list(groups.get(name, ()))
        if not values:
            continue
        entries.append(
            {
                "name": name,
                "parameter_names": [parameter_name for parameter_name, _ in values],
                "parameter_count": len(values),
                "parameter_numel": int(sum(parameter.numel() for _, parameter in values)),
                "learning_rates": learning_rates[name],
                "weight_decay": weight_decays[name],
            }
        )
    return {
        "artifact_type": "tubelet_optimizer_manifest_v1",
        "groups": entries,
        "optimization_config": asdict(config),
    }


def optimization_phase(
    optimizer_step: int,
    config: TubeletOptimizationConfig,
) -> str:
    """Return the phase used by the next optimizer update."""

    return (
        "readout_warmup"
        if optimizer_step < config.readout_warmup_optimizer_steps
        else "full_finetune"
    )


def _phase_learning_rate(
    group_name: str,
    phase: str,
    config: TubeletOptimizationConfig,
) -> float:
    if phase == "readout_warmup":
        return {
            "backbone": 0.0,
            "pooling": config.warmup_pooling_learning_rate,
            "ttc_head": config.warmup_head_learning_rate,
        }[group_name]
    return {
        "backbone": config.backbone_learning_rate,
        "pooling": config.pooling_learning_rate,
        "ttc_head": config.head_learning_rate,
    }[group_name]


def apply_optimizer_phase(
    optimizer: torch.optim.Optimizer,
    optimizer_step: int,
    config: TubeletOptimizationConfig,
) -> tuple[str, dict[str, float]]:
    """Set per-group LRs for the next update without rebuilding the optimizer."""

    phase = optimization_phase(optimizer_step, config)
    learning_rates: dict[str, float] = {}
    for group in optimizer.param_groups:
        raw_name = group.get("name")
        if not isinstance(raw_name, str) or raw_name not in {
            "backbone",
            "pooling",
            "ttc_head",
        }:
            raise ValueError(f"Unexpected Tubelet optimizer group name: {raw_name!r}")
        value = _phase_learning_rate(raw_name, phase, config)
        group["lr"] = value
        learning_rates[raw_name] = value
    return phase, learning_rates


def build_tubelet_optimizer(
    model: TubeletModelProtocol,
    config: TubeletOptimizationConfig,
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """Construct AdamW with named, disjoint, resume-stable parameter groups."""

    groups = split_parameter_groups(model)
    manifest = optimizer_group_manifest(groups, config)
    parameter_groups: list[dict[str, Any]] = []
    weight_decays = {
        "backbone": config.backbone_weight_decay,
        "pooling": config.readout_weight_decay,
        "ttc_head": config.readout_weight_decay,
    }
    for name in ("backbone", "pooling", "ttc_head"):
        values = groups[name]
        if not values:
            continue
        parameter_groups.append(
            {
                "name": name,
                "params": [parameter for _, parameter in values],
                "lr": _phase_learning_rate(name, optimization_phase(0, config), config),
                "weight_decay": weight_decays[name],
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    apply_optimizer_phase(optimizer, 0, config)
    return optimizer, manifest


def validate_optimizer_manifest(
    expected: Mapping[str, Any],
    received: object,
) -> None:
    """Reject a resume checkpoint created with different group membership."""

    if not isinstance(received, Mapping):
        raise ValueError("Resume checkpoint is missing optimizer_manifest")
    if dict(received) != dict(expected):
        raise ValueError("Resume optimizer manifest does not match the current model/config")


def suppress_warmup_backbone_gradients(
    optimizer: torch.optim.Optimizer,
    phase: str,
) -> None:
    """Prevent Adam moments from accumulating for a frozen warm-up backbone."""

    if phase != "readout_warmup":
        return
    for group in optimizer.param_groups:
        if group.get("name") != "backbone":
            continue
        for parameter in group["params"]:
            parameter.grad = None


def current_group_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Return named optimizer-group learning rates for artifacts and logs."""

    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = group.get("name")
        if isinstance(name, str):
            result[name] = float(group["lr"])
    return result


def _to_numpy(values: object) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().float().cpu().numpy().reshape(-1)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def prediction_health(targets: object, predictions: object) -> dict[str, Any]:
    """Measure whether the TTC predictor uses input-dependent variation."""

    target = _to_numpy(targets)
    prediction = _to_numpy(predictions)
    if target.shape != prediction.shape:
        raise ValueError(
            f"Target/prediction shape mismatch: {target.shape!r} != {prediction.shape!r}"
        )
    finite = np.isfinite(target) & np.isfinite(prediction)
    if not bool(finite.any()):
        return {
            "sample_count": int(len(target)),
            "finite_sample_count": 0,
            "target_mean": None,
            "target_std": None,
            "prediction_mean": None,
            "prediction_std": None,
            "prediction_std_ratio": 0.0,
            "pearson": None,
            "mae": None,
            "prediction_min": None,
            "prediction_max": None,
        }
    target = target[finite]
    prediction = prediction[finite]
    target_std = float(np.std(target))
    prediction_std = float(np.std(prediction))
    pearson: float | None = None
    if len(target) >= 2 and target_std > 0.0 and prediction_std > 0.0:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
        pearson = correlation if math.isfinite(correlation) else None
    return {
        "sample_count": int(len(finite)),
        "finite_sample_count": int(finite.sum()),
        "target_mean": float(np.mean(target)),
        "target_std": target_std,
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": prediction_std,
        "prediction_std_ratio": prediction_std / max(target_std, 1e-12),
        "pearson": pearson,
        "mae": float(np.mean(np.abs(prediction - target))),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
    }


def is_prediction_collapsed(
    health: Mapping[str, Any],
    config: TubeletOptimizationConfig,
) -> bool:
    """Return whether validation predictions fail the preregistered variance gate."""

    ratio = health.get("prediction_std_ratio")
    finite_count = int(health.get("finite_sample_count", 0))
    sample_count = int(health.get("sample_count", 0))
    if finite_count != sample_count or sample_count <= 0:
        return True
    if ratio is None or not math.isfinite(float(ratio)):
        return True
    return float(ratio) < config.min_prediction_std_ratio


def checkpoint_is_eligible(
    *,
    score: float,
    health: Mapping[str, Any],
    optimizer_step: int,
    config: TubeletOptimizationConfig,
) -> bool:
    """Require completed warm-up, finite metric, and non-collapsed predictions."""

    warmup_finished = (
        config.readout_warmup_optimizer_steps == 0
        or optimizer_step > config.readout_warmup_optimizer_steps
    )
    return (
        warmup_finished
        and math.isfinite(score)
        and not is_prediction_collapsed(health, config)
    )


__all__ = [
    "PredictionCollapseError",
    "TrainEpochResult",
    "TubeletOptimizationConfig",
    "apply_optimizer_phase",
    "build_tubelet_optimizer",
    "checkpoint_is_eligible",
    "current_group_learning_rates",
    "is_prediction_collapsed",
    "optimization_phase",
    "optimizer_group_manifest",
    "prediction_health",
    "resolve_optimization_config",
    "split_parameter_groups",
    "suppress_warmup_backbone_gradients",
    "validate_optimizer_manifest",
]
