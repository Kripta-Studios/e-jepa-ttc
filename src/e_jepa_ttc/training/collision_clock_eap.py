"""Fixed-update E-Clock trainer with complete fail-closed resume identity."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
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
    resume_checkpoint_path: Path
    progress_path: Path
    milestone_paths: tuple[Path, ...]


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


_EMPTY_SCHEDULE_SHA256 = hashlib.sha256(b"eclock-x0-schedule-v3").hexdigest()


def _schedule_hash(tokens: Sequence[str], previous: str = _EMPTY_SCHEDULE_SHA256) -> str:
    digest = hashlib.sha256(bytes.fromhex(previous))
    for token in tokens:
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


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


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _read_progress_losses(path: Path, completed_updates: int) -> list[float]:
    if not path.is_file():
        raise ValueError("resume progress JSONL is missing")
    losses: list[float] = []
    observed_updates: list[int] = []
    original_text = path.read_text(encoding="utf-8")
    lines = original_text.splitlines()
    retained_lines: list[str] = []
    extra_updates = False
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        update = int(record["update"])
        if record.get("event") != "update":
            if update <= completed_updates:
                retained_lines.append(line)
            else:
                extra_updates = True
            continue
        if update == 0:
            retained_lines.append(line)
            continue
        if update > completed_updates:
            extra_updates = True
            continue
        if "loss" not in record:
            raise ValueError("resume progress record lacks loss")
        observed_updates.append(update)
        losses.append(float(record["loss"]))
        retained_lines.append(line)
    if observed_updates != list(range(1, completed_updates + 1)):
        raise ValueError("resume progress JSONL is incomplete or non-contiguous")
    if extra_updates:
        orphaned = path.with_name(f"{path.stem}.orphaned-after-{completed_updates}.jsonl")
        if orphaned.exists():
            raise ValueError("orphaned progress evidence path already exists")
        orphaned.write_text(original_text, encoding="utf-8")
        temporary = path.with_name(f".{path.name}.resume-prefix.tmp")
        temporary.write_text("\n".join(retained_lines) + "\n", encoding="utf-8")
        temporary.replace(path)
    return losses


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    identity: CollisionClockScientificIdentity,
    completed: int,
    schedule_hash: str,
    consumed_token_count: int,
    losses: Sequence[float],
    batch_count: int,
) -> dict[str, Any]:
    return {
        "artifact_type": "eclock_x0_checkpoint_v2",
        "scientific_identity": asdict(identity),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "completed_updates": completed,
        "loss_count": len(losses),
        "loss_sum": float(np.sum(np.asarray(losses, dtype=np.float64), dtype=np.float64)),
        "recent_losses": list(losses[-100:]),
        "consumed_token_count": consumed_token_count,
        "batch_schedule_sha256": schedule_hash,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "sampler_order_state": {"next_update": completed, "batch_count": batch_count},
    }


def _save_checkpoint(
    path: Path,
    *,
    payload: dict[str, Any],
    identity: CollisionClockScientificIdentity,
    completed: int,
    schedule_hash: str,
    immutable: bool,
) -> tuple[Path, bool]:
    if immutable and path.exists():
        existing, _decision = validate_resume_checkpoint(path, expected_identity=identity)
        if int(existing["completed_updates"]) != completed:
            raise ValueError("immutable milestone has the wrong update identity")
        return _checkpoint_manifest_path(path), completed == identity.update_budget
    _atomic_torch_save(payload, path)
    return _write_checkpoint_manifest(
        path, identity=identity, completed_updates=completed, schedule_hash=schedule_hash
    )


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
    if int(payload.get("loss_count", -1)) != int(payload.get("completed_updates", -2)):
        raise ValueError("checkpoint compact loss state is inconsistent")
    if not isinstance(payload.get("batch_schedule_sha256"), str):
        raise ValueError("checkpoint schedule digest is missing")
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
    checkpoint_every: int = 100,
    milestone_updates: Sequence[int] = (250, 500, 1000, 2000, 4000, 6840),
    progress_path: Path | None = None,
    rich_log_every: int = 25,
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
    if checkpoint_every <= 0 or rich_log_every <= 0:
        raise ValueError("checkpoint/progress cadence must be positive")
    if any(value <= 0 for value in milestone_updates):
        raise ValueError("milestone updates must be positive")
    effective_milestones = {value for value in milestone_updates if value <= config.update_budget}
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
    schedule_hash = _EMPTY_SCHEDULE_SHA256
    consumed_token_count = 0
    start = 0
    progress = progress_path or checkpoint_path.with_name(f"{checkpoint_path.stem}.progress.jsonl")
    if resume:
        payload, resume_decision = validate_resume_checkpoint(
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
        losses = _read_progress_losses(progress, start)
        if len(losses) != int(payload["loss_count"]):
            raise ValueError("resume progress/checkpoint loss count mismatch")
        schedule_hash = str(payload["batch_schedule_sha256"])
        consumed_token_count = int(payload["consumed_token_count"])
        sampler_state = payload["sampler_order_state"]
        if sampler_state != {"next_update": start, "batch_count": len(batches)}:
            raise ValueError("resume sampler/order state mismatch")
        _append_jsonl(
            progress,
            {
                "event": "training_resume",
                "utc": datetime.now(UTC).isoformat(),
                "update": start,
                "arm_id": scientific_identity.arm_id,
                "outer_fold": scientific_identity.outer_fold,
                "seed": scientific_identity.seed,
                "git_commit": scientific_identity.git_commit_observed,
                "resume_decision_sha256": resume_decision["artifact_sha256"],
            },
        )
    else:
        _seed_everything(config.seed)
        if progress.exists():
            raise FileExistsError("new training progress path already exists")
        _append_jsonl(
            progress,
            {
                "event": "training_start",
                "utc": datetime.now(UTC).isoformat(),
                "update": 0,
                "arm_id": scientific_identity.arm_id,
                "outer_fold": scientific_identity.outer_fold,
                "seed": scientific_identity.seed,
                "git_commit": scientific_identity.git_commit_observed,
                "initialization_sha256": scientific_identity.initialization_sha256,
            },
        )
    if start > stop:
        raise ValueError("checkpoint is beyond requested stop")

    model.train()
    model_device = next(model.parameters()).device
    manifest_path = _checkpoint_manifest_path(checkpoint_path)
    frozen = False
    milestones: list[Path] = []
    loss_window_25: list[float] = []
    loss_window_100: list[float] = []
    for update in range(start, stop):
        step_started = time.perf_counter()
        batch = batches[update % len(batches)]
        load_finished = time.perf_counter()
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
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at update {update + 1}")
        loss.backward()
        grad_sq = torch.zeros((), dtype=torch.float64, device=model_device)
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"non-finite gradient at update {update + 1}")
                grad_sq += torch.sum(parameter.grad.detach().to(torch.float64) ** 2)
        grad_norm = float(torch.sqrt(grad_sq).detach().cpu())
        backward_finished = time.perf_counter()
        optimizer.step()
        scheduler.step()
        optimizer_finished = time.perf_counter()
        parameter_sq = torch.zeros((), dtype=torch.float64, device=model_device)
        for parameter in model.parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"non-finite parameter at update {update + 1}")
            parameter_sq += torch.sum(parameter.detach().to(torch.float64) ** 2)
        parameter_norm = float(torch.sqrt(parameter_sq).detach().cpu())
        losses.append(float(loss.detach()))
        loss_window_25 = (loss_window_25 + [losses[-1]])[-25:]
        loss_window_100 = (loss_window_100 + [losses[-1]])[-100:]
        schedule_hash = _schedule_hash(batch.sample_tokens, schedule_hash)
        consumed_token_count += len(batch.sample_tokens)
        completed = update + 1
        phase = output.benchmark_phase_mean.detach().to(torch.float64)
        inverse = output.inverse_ttc_mean.detach().to(torch.float64)
        raw_ttc = output.predicted_ttc_raw.detach().to(torch.float64)
        record: dict[str, Any] = {
            "event": "update",
            "utc": datetime.now(UTC).isoformat(),
            "update": completed,
            "arm_id": scientific_identity.arm_id,
            "outer_fold": scientific_identity.outer_fold,
            "seed": scientific_identity.seed,
            "git_commit": scientific_identity.git_commit_observed,
            "batch_index": update % len(batches),
            "batch_rows": len(batch.sample_tokens),
            "loss": losses[-1],
            "loss_rolling_25": float(np.mean(loss_window_25, dtype=np.float64)),
            "loss_rolling_100": float(np.mean(loss_window_100, dtype=np.float64)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": grad_norm,
            "grad_finite": True,
            "parameter_norm": parameter_norm,
            "parameter_finite": True,
            "phase_mean": float(phase.mean().cpu()),
            "phase_std": float(phase.std(unbiased=False).cpu()),
            "phase_min": float(phase.min().cpu()),
            "phase_max": float(phase.max().cpu()),
            "inverse_ttc_finite_fraction": float(torch.isfinite(inverse).to(torch.float64).mean()),
            "raw_ttc_finite_fraction": float(torch.isfinite(raw_ttc).to(torch.float64).mean()),
            "batch_load_ms": (load_finished - step_started) * 1000.0,
            "forward_backward_ms": (backward_finished - load_finished) * 1000.0,
            "optimizer_ms": (optimizer_finished - backward_finished) * 1000.0,
            "samples_per_second": len(batch.sample_tokens)
            / max(optimizer_finished - step_started, 1.0e-12),
            "cpu_rss_bytes": psutil.Process().memory_info().rss,
            "system_available_ram_bytes": psutil.virtual_memory().available,
            "gpu_allocated_bytes": (
                torch.cuda.memory_allocated(model_device) if model_device.type == "cuda" else 0
            ),
            "gpu_reserved_bytes": (
                torch.cuda.memory_reserved(model_device) if model_device.type == "cuda" else 0
            ),
            "gpu_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(model_device) if model_device.type == "cuda" else 0
            ),
            "rich_record": completed % rich_log_every == 0,
            "batch_schedule_sha256": schedule_hash,
        }
        _append_jsonl(progress, record)
        payload: dict[str, Any] | None = None
        checkpoint_due = completed % checkpoint_every == 0 or completed == stop
        milestone_due = completed in effective_milestones or completed == config.update_budget
        if checkpoint_due or milestone_due:
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                identity=scientific_identity,
                completed=completed,
                schedule_hash=schedule_hash,
                consumed_token_count=consumed_token_count,
                losses=losses,
                batch_count=len(batches),
            )
        if checkpoint_due and payload is not None:
            manifest_path, frozen = _save_checkpoint(
                checkpoint_path,
                payload=payload,
                identity=scientific_identity,
                completed=completed,
                schedule_hash=schedule_hash,
                immutable=False,
            )
        if milestone_due and payload is not None:
            milestone = checkpoint_path.parent / "milestones" / f"update-{completed:06d}.pt"
            _save_checkpoint(
                milestone,
                payload=payload,
                identity=scientific_identity,
                completed=completed,
                schedule_hash=schedule_hash,
                immutable=True,
            )
            milestones.append(milestone)
    final_path = (
        checkpoint_path.parent / "milestones" / f"update-{stop:06d}.pt"
        if stop == config.update_budget
        else checkpoint_path
    )
    if stop == config.update_budget and not final_path.is_file():
        raise RuntimeError("final scientific milestone checkpoint was not created")
    return CollisionClockTrainingResult(
        completed_updates=stop,
        losses=tuple(losses),
        batch_schedule_sha256=schedule_hash,
        checkpoint_path=final_path,
        checkpoint_manifest_path=_checkpoint_manifest_path(final_path),
        checkpoint_frozen=stop == config.update_budget,
        resume_checkpoint_path=checkpoint_path,
        progress_path=progress,
        milestone_paths=tuple(milestones),
    )


__all__ = [
    "CollisionClockScientificIdentity",
    "CollisionClockTrainingConfig",
    "CollisionClockTrainingResult",
    "require_frozen_checkpoint",
    "train_collision_clock_updates",
    "validate_resume_checkpoint",
]
