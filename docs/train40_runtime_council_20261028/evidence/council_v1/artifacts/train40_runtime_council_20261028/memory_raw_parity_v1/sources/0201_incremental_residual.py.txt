"""Deterministic fixed-budget training for the preregistered E-Clock X1 adapter."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.collision_clock_protocol import (
    module_topology_sha256,
    tensor_state_sha256,
)
from e_jepa_ttc.models.incremental_residual import FrozenA5DynamicResidualAdapter


@dataclass(frozen=True)
class X1TrainingConfig:
    """Frozen X1 optimizer and budget contract."""

    arm_id: str
    seed: int
    outer_fold: int
    batch_size: int = 256
    update_budget: int = 1000
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    precision_mode: str = "float32"
    checkpoint_policy: str = "last_update_fixed_budget"

    def __post_init__(self) -> None:
        if self.arm_id not in {"X1-A5-ZERO-U", "X1-A5-DYN-U", "X1-A5-SHUFFLE-U"}:
            raise ValueError("X1 trainer received an unauthorized arm")
        if self.seed not in {7, 13, 23} or self.outer_fold not in {0, 1, 2}:
            raise ValueError("X1 seed/fold is outside the preregistered registry")
        if (self.batch_size, self.update_budget) != (256, 1000):
            raise ValueError("X1 batch size/update budget are frozen to 256/1000")
        if (self.learning_rate, self.weight_decay) != (3.0e-4, 1.0e-4):
            raise ValueError("X1 optimizer hyperparameters drifted")
        if self.precision_mode != "float32":
            raise ValueError("X1 precision is frozen to float32")
        if self.checkpoint_policy != "last_update_fixed_budget":
            raise ValueError("X1 outer-dev checkpoint selection is forbidden")


@dataclass(frozen=True)
class X1TrainingIdentity:
    """Complete identity required to resume or evaluate one X1 fold."""

    training_commit: str
    feature_table_sha256: str
    x05_gate_sha256: str
    protocol_sha256: str
    config_sha256: str
    arm_id: str
    seed: int
    outer_fold: int
    train_token_sha256: str
    dev_token_sha256: str
    topology_sha256: str
    initialization_sha256: str
    trainable_mask_sha256: str
    normalization_sha256: str
    batch_schedule_sha256: str
    a5_frozen: bool
    transport_extractor_frozen: bool
    outer_dev_available_to_trainer: bool
    update_budget: int = 1000
    checkpoint_policy: str = "last_update_fixed_budget"

    def __post_init__(self) -> None:
        if self.outer_dev_available_to_trainer:
            raise ValueError("outer-dev must be unavailable to the X1 trainer")
        if not self.a5_frozen or not self.transport_extractor_frozen:
            raise ValueError("X1 requires frozen A5 and transport features")
        if self.update_budget != 1000 or self.checkpoint_policy != "last_update_fixed_budget":
            raise ValueError("X1 identity budget/checkpoint policy mismatch")
        for key, value in asdict(self).items():
            if isinstance(value, str) and not value:
                raise ValueError(f"empty X1 training identity field: {key}")


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


def _atomic_json(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    signed = sign_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(signed, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return signed


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def trainable_mask_sha256(module: nn.Module) -> str:
    """Hash exact names/shapes/requires-grad flags for matched-arm auditing."""

    payload = [
        {"name": name, "shape": list(parameter.shape), "requires_grad": parameter.requires_grad}
        for name, parameter in module.named_parameters()
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalization_sha256(mean: np.ndarray, std: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(mean, dtype="<f8").tobytes())
    digest.update(np.asarray(std, dtype="<f8").tobytes())
    return digest.hexdigest()


def token_sha256(tokens: list[str]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def deterministic_sequence_grouped_schedule(
    sequence_ids: list[str],
    tokens: list[str],
    *,
    seed: int,
    batch_size: int = 256,
    updates: int = 1000,
) -> tuple[list[np.ndarray], str]:
    """Create a deterministic grouped epoch schedule shared by all matched arms."""

    if len(sequence_ids) != len(tokens) or not sequence_ids:
        raise ValueError("schedule identities are missing or misaligned")
    sequences = np.asarray(sequence_ids, dtype=str)
    batches: list[np.ndarray] = []
    digest = hashlib.sha256(b"eclock-x1-sequence-grouped-v1")
    epoch = 0
    pending = np.empty(0, dtype=np.int64)
    cursor = 0
    while len(batches) < updates:
        if cursor + batch_size > pending.size:
            rng = np.random.default_rng(seed + epoch * 1_000_003)
            sequence_order = rng.permutation(sorted(np.unique(sequences).tolist())).tolist()
            pieces: list[np.ndarray] = []
            for sequence in sequence_order:
                indices = np.flatnonzero(sequences == sequence)
                pieces.append(indices[rng.permutation(indices.size)])
            epoch_values = np.concatenate(pieces)
            pending = np.concatenate((pending[cursor:], epoch_values))
            cursor = 0
            epoch += 1
        batch = pending[cursor : cursor + batch_size]
        cursor += batch_size
        if batch.size != batch_size:
            raise RuntimeError("deterministic schedule emitted a partial batch")
        batches.append(batch.copy())
        for index in batch:
            encoded = tokens[int(index)].encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    return batches, digest.hexdigest()


def _weighted_smooth_l1(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    normalized = weights / weights.mean().clamp_min(1.0e-12)
    loss = functional.smooth_l1_loss(prediction, target, reduction="none")
    return torch.mean(normalized * loss)


def _checkpoint_manifest(path: Path) -> Path:
    return path.with_name(f"{path.name}.manifest.json")


def _save_checkpoint(
    *,
    path: Path,
    model: FrozenA5DynamicResidualAdapter,
    optimizer: torch.optim.Optimizer,
    identity: X1TrainingIdentity,
    completed_updates: int,
    losses: list[float],
) -> dict[str, Any]:
    payload = {
        "artifact_type": "eclock_x1_checkpoint_v1",
        "identity": asdict(identity),
        "completed_updates": completed_updates,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "loss_count": len(losses),
        "loss_sum_float64": float(np.sum(np.asarray(losses, dtype=np.float64))),
    }
    _atomic_torch_save(payload, path)
    return _atomic_json(
        {
            "artifact_type": "eclock_x1_checkpoint_manifest_v1",
            "checkpoint_path": str(path),
            "checkpoint_file_sha256": compute_file_hash(str(path)),
            "checkpoint_bytes": path.stat().st_size,
            "completed_updates": completed_updates,
            "update_budget": identity.update_budget,
            "frozen": completed_updates == identity.update_budget,
            "checkpoint_policy": identity.checkpoint_policy,
            "identity": asdict(identity),
            "model_state_sha256": tensor_state_sha256(model),
        },
        _checkpoint_manifest(path),
    )


def _load_resume(
    path: Path,
    *,
    expected_identity: X1TrainingIdentity,
    model: FrozenA5DynamicResidualAdapter,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, list[float]]:
    manifest_path = _checkpoint_manifest(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not verify_artifact_hash(manifest):
        raise ValueError("X1 resume manifest signature mismatch")
    if (
        manifest.get("artifact_type") != "eclock_x1_checkpoint_manifest_v1"
        or manifest.get("identity") != asdict(expected_identity)
        or manifest.get("checkpoint_file_sha256") != compute_file_hash(str(path))
        or manifest.get("checkpoint_bytes") != path.stat().st_size
    ):
        raise ValueError("X1 resume checkpoint identity/hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("artifact_type") != "eclock_x1_checkpoint_v1" or payload.get(
        "identity"
    ) != asdict(expected_identity):
        raise ValueError("X1 resume embedded identity mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    if payload["torch_cuda_rng_state"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("X1 resume requires unavailable CUDA RNG state")
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state"])
    completed = int(payload["completed_updates"])
    losses: list[float] = []
    progress = path.with_name("progress.jsonl")
    for line in progress.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == "update" and int(record["update"]) <= completed:
            losses.append(float(record["loss"]))
    if len(losses) != completed or int(payload["loss_count"]) != completed:
        raise ValueError("X1 resume progress is incomplete")
    return completed, losses


def train_x1_fixed_budget(
    model: FrozenA5DynamicResidualAdapter,
    *,
    a5_phase: np.ndarray,
    slots: np.ndarray,
    target_phase: np.ndarray,
    sample_weights: np.ndarray,
    schedule: list[np.ndarray],
    config: X1TrainingConfig,
    identity: X1TrainingIdentity,
    output_root: Path,
    device: torch.device,
    resume: bool = False,
    stop_after_updates: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Train without any outer-dev object and freeze exactly update 1,000."""

    if len(schedule) != config.update_budget or identity.batch_schedule_sha256 == "":
        raise ValueError("X1 fixed schedule/budget mismatch")
    values = (
        np.asarray(a5_phase, dtype=np.float32),
        np.asarray(slots, dtype=np.float32),
        np.asarray(target_phase, dtype=np.float32),
        np.asarray(sample_weights, dtype=np.float32),
    )
    if values[1].shape != (values[0].size, 9) or any(
        not np.isfinite(value).all() for value in values
    ):
        raise ValueError("X1 training arrays are invalid")
    if module_topology_sha256(model) != identity.topology_sha256:
        raise ValueError("X1 model topology drifted")
    if trainable_mask_sha256(model) != identity.trainable_mask_sha256:
        raise ValueError("X1 trainable mask drifted")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    output_root.mkdir(parents=True, exist_ok=resume)
    checkpoint = output_root / "resume_latest.pt"
    progress = output_root / "progress.jsonl"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    start = 0
    losses: list[float] = []
    if resume:
        start, losses = _load_resume(
            checkpoint,
            expected_identity=identity,
            model=model,
            optimizer=optimizer,
        )
        _append_progress(
            progress,
            {
                "event": "resume",
                "update": start,
                "utc": datetime.now(UTC).isoformat(),
            },
        )
    elif progress.exists() or checkpoint.exists():
        raise FileExistsError("X1 output exists without explicit resume")
    else:
        with torch.no_grad():
            probe_count = min(256, values[0].size)
            observed = model(
                torch.from_numpy(values[0][:probe_count]).to(device),
                torch.from_numpy(values[1][:probe_count]).to(device),
            ).cpu()
        if not torch.equal(observed, torch.from_numpy(values[0][:probe_count])):
            raise ValueError("X1 zero initialization does not exactly replay A5 on outer-train")
        _append_progress(
            progress,
            {
                "event": "update_zero",
                "update": 0,
                "a5_replay_exact": True,
                "utc": datetime.now(UTC).isoformat(),
            },
        )
        if progress_callback is not None:
            progress_callback(0)
        _save_checkpoint(
            path=output_root / "milestones" / "update-000000.pt",
            model=model,
            optimizer=optimizer,
            identity=identity,
            completed_updates=0,
            losses=losses,
        )
    stop = config.update_budget if stop_after_updates is None else stop_after_updates
    if stop <= start or stop > config.update_budget:
        raise ValueError("X1 stop_after_updates is outside the remaining budget")
    phase_tensor = torch.from_numpy(values[0]).to(device)
    slot_tensor = torch.from_numpy(values[1]).to(device)
    target_tensor = torch.from_numpy(values[2]).to(device)
    weight_tensor = torch.from_numpy(values[3]).to(device)
    started = time.perf_counter()
    milestones = {100, 250, 500, 750, 1000}
    for update in range(start, stop):
        indices = torch.as_tensor(schedule[update], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(phase_tensor[indices], slot_tensor[indices])
        loss = _weighted_smooth_l1(prediction, target_tensor[indices], weight_tensor[indices])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("X1 loss became non-finite")
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        losses.append(value)
        completed = update + 1
        _append_progress(
            progress,
            {
                "event": "update",
                "update": completed,
                "loss": value,
                "utc": datetime.now(UTC).isoformat(),
            },
        )
        if progress_callback is not None:
            progress_callback(completed)
        if completed % 100 == 0 or completed in milestones or completed == stop:
            _save_checkpoint(
                path=checkpoint,
                model=model,
                optimizer=optimizer,
                identity=identity,
                completed_updates=completed,
                losses=losses,
            )
        if completed in milestones:
            _save_checkpoint(
                path=output_root / "milestones" / f"update-{completed:06d}.pt",
                model=model,
                optimizer=optimizer,
                identity=identity,
                completed_updates=completed,
                losses=losses,
            )
    final_manifest = json.loads(_checkpoint_manifest(checkpoint).read_text(encoding="utf-8"))
    if stop == config.update_budget and not final_manifest.get("frozen"):
        raise ValueError("X1 last fixed-budget checkpoint was not frozen")
    duration = time.perf_counter() - started
    return sign_artifact(
        {
            "artifact_type": "eclock_x1_training_summary_v1",
            "arm_id": config.arm_id,
            "seed": config.seed,
            "outer_fold": config.outer_fold,
            "completed_updates": stop,
            "update_budget": config.update_budget,
            "checkpoint_policy": config.checkpoint_policy,
            "checkpoint_path": str(checkpoint),
            "checkpoint_file_sha256": final_manifest["checkpoint_file_sha256"],
            "checkpoint_manifest_sha256": final_manifest["artifact_sha256"],
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_count": len(losses),
            "wall_seconds_this_invocation": duration,
            "updates_per_second_this_invocation": (stop - start) / max(duration, 1.0e-12),
            "train_rows": int(values[0].size),
            "effective_epochs": float(stop * config.batch_size / values[0].size),
            "outer_dev_available_to_trainer": False,
            "outer_dev_used_for_selection": False,
        }
    )


def load_frozen_x1_checkpoint(
    checkpoint: Path,
    *,
    expected_identity: X1TrainingIdentity,
    model: FrozenA5DynamicResidualAdapter,
    device: torch.device,
) -> FrozenA5DynamicResidualAdapter:
    """Load only update-1,000 X1 checkpoints after complete physical verification."""

    manifest = json.loads(_checkpoint_manifest(checkpoint).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not verify_artifact_hash(manifest):
        raise ValueError("X1 frozen checkpoint manifest signature mismatch")
    if (
        manifest.get("identity") != asdict(expected_identity)
        or manifest.get("completed_updates") != 1000
        or manifest.get("update_budget") != 1000
        or manifest.get("frozen") is not True
        or manifest.get("checkpoint_policy") != "last_update_fixed_budget"
        or manifest.get("checkpoint_file_sha256") != compute_file_hash(str(checkpoint))
    ):
        raise ValueError("X1 checkpoint is not the fixed final update")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("identity") != asdict(expected_identity):
        raise ValueError("X1 checkpoint embedded identity mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


__all__ = [
    "X1TrainingConfig",
    "X1TrainingIdentity",
    "deterministic_sequence_grouped_schedule",
    "load_frozen_x1_checkpoint",
    "normalization_sha256",
    "token_sha256",
    "train_x1_fixed_budget",
    "trainable_mask_sha256",
]
