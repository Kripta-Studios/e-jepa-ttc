"""Deterministic fixed-update trainer used by the bounded E-Clock X0 pipeline."""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.losses.collision_clock import uniform_benchmark_phase_loss
from e_jepa_ttc.models.collision_clock_math import ttc_to_benchmark_phase


@dataclass(frozen=True)
class CollisionClockTrainingConfig:
    arm_id: str
    seed: int = 7
    planned_updates: int = 100
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    precision_mode: str = "float32"
    checkpoint_policy: str = "last_update_fixed_budget"

    def __post_init__(self) -> None:
        if self.arm_id not in {"X0-PAIR-U", "X0-BASE-U", "X0-DYN-U"}:
            raise ValueError("trainer accepts only authorized executable X0 arms")
        if self.seed != 7:
            raise ValueError("this X0 phase authorizes seed 7 only")
        if self.planned_updates <= 0 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid fixed-budget optimizer contract")
        if self.precision_mode != "float32":
            raise ValueError("X0 precision_mode is frozen to float32")
        if self.checkpoint_policy != "last_update_fixed_budget":
            raise ValueError("outer-dev checkpoint selection is forbidden")


@dataclass(frozen=True)
class CollisionClockBatch:
    inputs: torch.Tensor
    delta_t_s: torch.Tensor
    target_ttc_seconds: torch.Tensor
    sample_tokens: tuple[str, ...]


@dataclass(frozen=True)
class CollisionClockTrainingResult:
    completed_updates: int
    losses: tuple[float, ...]
    batch_schedule_sha256: str
    checkpoint_path: Path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _schedule_hash(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def train_collision_clock_updates(
    model: nn.Module,
    batches: Sequence[CollisionClockBatch],
    *,
    config: CollisionClockTrainingConfig,
    checkpoint_path: Path,
    stop_after_updates: int | None = None,
    resume: bool = False,
) -> CollisionClockTrainingResult:
    """Train a fixed update schedule and atomically persist complete resume state."""

    if not batches:
        raise ValueError("training requires at least one batch")
    stop = config.planned_updates if stop_after_updates is None else stop_after_updates
    if stop <= 0 or stop > config.planned_updates:
        raise ValueError("stop_after_updates lies outside the planned budget")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    losses: list[float] = []
    consumed_tokens: list[str] = []
    start = 0
    if resume:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("artifact_type") != "eclock_x0_resume_checkpoint_v1":
            raise ValueError("resume artifact type mismatch")
        if payload.get("training_config") != asdict(config):
            raise ValueError("resume training contract mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        start = int(payload["completed_updates"])
        losses = [float(value) for value in payload["losses"]]
        consumed_tokens = [str(value) for value in payload["consumed_tokens"]]
        if _schedule_hash(consumed_tokens) != payload["batch_schedule_sha256"]:
            raise ValueError("resume batch schedule hash mismatch")
    else:
        _seed_everything(config.seed)
    if start > stop:
        raise ValueError("checkpoint is beyond requested stop")

    model.train()
    for update in range(start, stop):
        batch = batches[update % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.inputs, batch.delta_t_s)
        target_phase64, valid_target = ttc_to_benchmark_phase(
            batch.target_ttc_seconds.to(torch.float64),
            metric_delta_t_s=0.1,
        )
        if not bool(valid_target.all()) or not bool(torch.isfinite(target_phase64).all()):
            raise ValueError("training target is outside the benchmark-phase domain")
        loss = uniform_benchmark_phase_loss(
            output.benchmark_phase_mean,
            target_phase64.to(output.benchmark_phase_mean.dtype),
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        consumed_tokens.extend(batch.sample_tokens)
        schedule_hash = _schedule_hash(consumed_tokens)
        _atomic_torch_save(
            {
                "artifact_type": "eclock_x0_resume_checkpoint_v1",
                "training_config": asdict(config),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "completed_updates": update + 1,
                "losses": losses,
                "consumed_tokens": consumed_tokens,
                "batch_schedule_sha256": schedule_hash,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
            },
            checkpoint_path,
        )
    return CollisionClockTrainingResult(
        completed_updates=stop,
        losses=tuple(losses),
        batch_schedule_sha256=_schedule_hash(consumed_tokens),
        checkpoint_path=checkpoint_path,
    )


__all__ = [
    "CollisionClockBatch",
    "CollisionClockTrainingConfig",
    "CollisionClockTrainingResult",
    "train_collision_clock_updates",
]
