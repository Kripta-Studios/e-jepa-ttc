"""Fixed-update E-Clock trainer with complete fail-closed resume identity."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.artifacts.hashing import (
    compute_file_hash,
    sign_artifact,
    verify_artifact_hash,
)
from e_jepa_ttc.data.collision_clock_cache import CollisionClockOuterTrainBatch
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    module_topology_sha256,
    tensor_state_sha256,
)
from e_jepa_ttc.losses.collision_clock import uniform_benchmark_phase_loss
from e_jepa_ttc.models.collision_clock_math import ttc_to_benchmark_phase

_TRAINABLE_ARMS = {"X0-PAIR-U", "X0-BASE-U", "X0-DYN-U"}


@dataclass(frozen=True)
class CollisionClockTrainingConfig:
    arm_id: str
    seed: int = 7
    update_budget: int = 100
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    precision_mode: str = "float32"
    checkpoint_policy: str = "last_update_fixed_budget"

    def __post_init__(self) -> None:
        if self.arm_id not in _TRAINABLE_ARMS:
            raise ValueError("trainer accepts only authorized trainable X0 arms")
        if self.seed != 7:
            raise ValueError("this X0 phase authorizes seed 7 only")
        if self.update_budget <= 0 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid fixed-update optimizer contract")
        if self.precision_mode != "float32":
            raise ValueError("X0 precision_mode is frozen to float32")
        if self.checkpoint_policy != "last_update_fixed_budget":
            raise ValueError("outer-dev/best-epoch checkpoint selection is forbidden")


@dataclass(frozen=True)
class CollisionClockScientificIdentity:
    git_commit_observed: str
    git_dirty_observed: bool
    arm_id: str
    scientific_role: str
    reference_family: str | None
    seed: int
    outer_fold: int
    motion_feature_mode: str
    model_class: str
    model_topology_sha256: str
    initialization_sha256: str
    config_path: str
    config_sha256: str
    protocol_path: str
    protocol_sha256: str
    reference_path: str
    reference_sha256: str
    split_manifest_path: str
    split_manifest_sha256: str
    cache_manifest_path: str
    cache_manifest_sha256: str
    ordered_token_identity_sha256: str
    target_sha256: str
    fold_assignment_sha256: str
    sample_weight_sha256: str
    train_token_subset_sha256: str
    dev_token_subset_sha256: str
    optimizer_config: Mapping[str, Any]
    scheduler_config: Mapping[str, Any]
    precision_mode: str
    update_budget: int
    checkpoint_policy: str

    def __post_init__(self) -> None:
        if self.git_dirty_observed:
            raise ValueError("scientific training is forbidden from a dirty worktree")
        if self.arm_id not in _TRAINABLE_ARMS or self.seed != 7 or self.outer_fold not in (0, 1, 2):
            raise ValueError("scientific arm/seed/fold identity is invalid")
        if self.checkpoint_policy != "last_update_fixed_budget" or self.update_budget <= 0:
            raise ValueError("scientific checkpoint/update identity is invalid")
        if self.precision_mode != "float32":
            raise ValueError("scientific precision identity must be float32")
        if self.arm_id == "X0-PAIR-U" and self.reference_family != "official_a5_oof":
            raise ValueError("X0-PAIR-U requires the official_a5_oof checkpoint family")
        if self.arm_id in {"X0-BASE-U", "X0-DYN-U"} and self.reference_family is not None:
            raise ValueError("BASE/DYN do not consume an A5 checkpoint family")
        values = asdict(self)
        skipped = {
            "git_dirty_observed",
            "reference_family",
            "optimizer_config",
            "scheduler_config",
        }
        for key, value in values.items():
            if key not in skipped and isinstance(value, str) and not value:
                raise ValueError(f"scientific identity field is empty: {key}")


@dataclass(frozen=True)
class CollisionClockTrainingResult:
    completed_updates: int
    losses: tuple[float, ...]
    batch_schedule_sha256: str
    checkpoint_path: Path
    checkpoint_manifest_path: Path
    checkpoint_frozen: bool


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


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _schedule_hash(tokens: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def _checkpoint_manifest_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.name}.manifest.json")


def _state_digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _write_checkpoint_manifest(
    checkpoint_path: Path,
    *,
    identity: CollisionClockScientificIdentity,
    completed_updates: int,
    schedule_hash: str,
) -> tuple[Path, bool]:
    frozen = completed_updates == identity.update_budget
    payload = sign_artifact(
        {
            "artifact_type": "eclock_x0_checkpoint_manifest_v2",
            "checkpoint_policy": "last_update_fixed_budget",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_file_sha256": compute_file_hash(str(checkpoint_path)),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "completed_updates": completed_updates,
            "update_budget": identity.update_budget,
            "frozen": frozen,
            "scientific_identity": asdict(identity),
            "batch_schedule_sha256": schedule_hash,
        }
    )
    manifest_path = _checkpoint_manifest_path(checkpoint_path)
    _atomic_json_save(payload, manifest_path)
    return manifest_path, frozen


def validate_resume_checkpoint(
    checkpoint_path: Path,
    *,
    expected_identity: CollisionClockScientificIdentity,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify physical bytes and every scientific identity field before loading."""

    manifest_path = _checkpoint_manifest_path(checkpoint_path)
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise ValueError("resume checkpoint or checkpoint manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not verify_artifact_hash(manifest):
        raise ValueError("checkpoint manifest signature mismatch")
    if manifest.get("artifact_type") != "eclock_x0_checkpoint_manifest_v2":
        raise ValueError("checkpoint manifest artifact type mismatch")
    if manifest.get("checkpoint_bytes") != checkpoint_path.stat().st_size:
        raise ValueError("checkpoint is truncated or has a different byte count")
    physical_sha = compute_file_hash(str(checkpoint_path))
    if manifest.get("checkpoint_file_sha256") != physical_sha:
        raise ValueError("checkpoint physical SHA mismatch")
    expected = asdict(expected_identity)
    if manifest.get("scientific_identity") != expected:
        raise ValueError("resume scientific identity mismatch")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("artifact_type") != "eclock_x0_checkpoint_v2":
        raise ValueError("resume checkpoint payload type mismatch")
    if payload.get("scientific_identity") != expected:
        raise ValueError("checkpoint embedded scientific identity mismatch")
    required_states = {
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "sampler_order_state",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "model_state_dict",
    }
    if not required_states.issubset(payload):
        raise ValueError("checkpoint resume state is incomplete")
    decision = sign_artifact(
        {
            "artifact_type": "eclock_x0_resume_decision_v2",
            "decision": "resume_accepted",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_physical_sha256": physical_sha,
            "checkpoint_manifest_sha256": manifest["artifact_sha256"],
            "scientific_identity": expected,
            "completed_updates": int(payload["completed_updates"]),
            "rng_state_sha256": {
                "python": _state_digest(payload["python_rng_state"]),
                "numpy": _state_digest(payload["numpy_rng_state"]),
                "torch_cpu": _state_digest(payload["torch_cpu_rng_state"]),
                "torch_cuda": _state_digest(payload["torch_cuda_rng_state"]),
            },
            "sampler_order_state": payload["sampler_order_state"],
        }
    )
    return payload, decision


def require_frozen_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Reject evaluation until the last fixed-budget update is frozen."""

    path = _checkpoint_manifest_path(checkpoint_path)
    if not path.is_file():
        raise ValueError("checkpoint manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not verify_artifact_hash(manifest):
        raise ValueError("checkpoint manifest signature mismatch")
    if manifest.get("checkpoint_policy") != "last_update_fixed_budget" or not manifest.get(
        "frozen"
    ):
        raise ValueError("outer-dev evaluation requires a frozen last-update checkpoint")
    if manifest.get("completed_updates") != manifest.get("update_budget"):
        raise ValueError("frozen checkpoint does not exhaust the update budget")
    checkpoint_sha = compute_file_hash(str(checkpoint_path))
    if checkpoint_sha != manifest.get("checkpoint_file_sha256"):
        raise ValueError("frozen checkpoint physical SHA mismatch")
    return manifest


def train_collision_clock_updates(
    model: nn.Module,
    batches: Sequence[CollisionClockOuterTrainBatch],
    *,
    config: CollisionClockTrainingConfig,
    scientific_identity: CollisionClockScientificIdentity,
    checkpoint_path: Path,
    stop_after_updates: int | None = None,
    resume: bool = False,
) -> CollisionClockTrainingResult:
    """Train only typed outer-train batches and persist exact resume state."""

    if not batches:
        raise TypeError("trainer accepts CollisionClockOuterTrainBatch values only")
    if config.arm_id != scientific_identity.arm_id or config.seed != scientific_identity.seed:
        raise ValueError("training config and scientific identity disagree")
    if (
        config.update_budget != scientific_identity.update_budget
        or config.precision_mode != scientific_identity.precision_mode
        or config.checkpoint_policy != scientific_identity.checkpoint_policy
    ):
        raise ValueError("training budget/precision/checkpoint identity mismatch")
    expected_optimizer = {
        "name": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if dict(scientific_identity.optimizer_config) != expected_optimizer:
        raise ValueError("optimizer config disagrees with scientific identity")
    if dict(scientific_identity.scheduler_config) != {"name": "constant"}:
        raise ValueError("scheduler config disagrees with scientific identity")
    if model.__class__.__name__ != scientific_identity.model_class:
        raise ValueError("model class disagrees with scientific identity")
    model_config = getattr(model, "config", None)
    if (
        getattr(model_config, "motion_feature_mode", None)
        != scientific_identity.motion_feature_mode
    ):
        raise ValueError("motion_feature_mode disagrees with scientific identity")
    if module_topology_sha256(model) != scientific_identity.model_topology_sha256:
        raise ValueError("model topology SHA disagrees with scientific identity")
    if tensor_state_sha256(model) != scientific_identity.initialization_sha256:
        raise ValueError("model initialization SHA disagrees with scientific identity")
    stop = config.update_budget if stop_after_updates is None else stop_after_updates
    if stop <= 0 or stop > config.update_budget:
        raise ValueError("stop_after_updates lies outside the fixed update budget")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    losses: list[float] = []
    consumed_tokens: list[str] = []
    start = 0
    if resume:
        payload, _decision = validate_resume_checkpoint(
            checkpoint_path, expected_identity=scientific_identity
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_cpu_rng_state"])
        cuda_state = payload["torch_cuda_rng_state"]
        if cuda_state is not None:
            if not torch.cuda.is_available():
                raise ValueError("checkpoint has CUDA RNG state but CUDA is unavailable")
            torch.cuda.set_rng_state_all(cuda_state)
        start = int(payload["completed_updates"])
        losses = [float(value) for value in payload["losses"]]
        consumed_tokens = [str(value) for value in payload["consumed_tokens"]]
        sampler_state = payload["sampler_order_state"]
        if sampler_state != {"next_update": start, "batch_count": len(batches)}:
            raise ValueError("resume sampler/order state mismatch")
        if _schedule_hash(consumed_tokens) != payload["batch_schedule_sha256"]:
            raise ValueError("resume batch schedule hash mismatch")
    else:
        _seed_everything(config.seed)
    if start > stop:
        raise ValueError("checkpoint is beyond requested stop")

    model.train()
    model_device = next(model.parameters()).device
    manifest_path = _checkpoint_manifest_path(checkpoint_path)
    frozen = False
    for update in range(start, stop):
        batch = batches[update % len(batches)]
        if type(batch) is not CollisionClockOuterTrainBatch:
            raise TypeError("trainer accepts CollisionClockOuterTrainBatch values only")
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.inputs.to(model_device),
            batch.delta_t_s.to(model_device),
        )
        target_phase64, valid_target = ttc_to_benchmark_phase(
            batch.target_ttc_seconds.to(model_device, dtype=torch.float64),
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
        completed = update + 1
        _atomic_torch_save(
            {
                "artifact_type": "eclock_x0_checkpoint_v2",
                "scientific_identity": asdict(scientific_identity),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "completed_updates": completed,
                "losses": losses,
                "consumed_tokens": consumed_tokens,
                "batch_schedule_sha256": schedule_hash,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "sampler_order_state": {
                    "next_update": completed,
                    "batch_count": len(batches),
                },
            },
            checkpoint_path,
        )
        manifest_path, frozen = _write_checkpoint_manifest(
            checkpoint_path,
            identity=scientific_identity,
            completed_updates=completed,
            schedule_hash=schedule_hash,
        )
    return CollisionClockTrainingResult(
        completed_updates=stop,
        losses=tuple(losses),
        batch_schedule_sha256=_schedule_hash(consumed_tokens),
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=manifest_path,
        checkpoint_frozen=frozen,
    )


__all__ = [
    "CollisionClockScientificIdentity",
    "CollisionClockTrainingConfig",
    "CollisionClockTrainingResult",
    "require_frozen_checkpoint",
    "train_collision_clock_updates",
    "validate_resume_checkpoint",
]
