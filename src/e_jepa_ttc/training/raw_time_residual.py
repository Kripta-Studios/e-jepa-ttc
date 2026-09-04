"""Fixed-budget Stage 64 trainer with exact resume and global macro weights."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import psutil
import torch

from e_jepa_ttc.models.raw_time_residual import (
    RawArm,
    RawModelBatch,
    RawTimeResidual,
    normalize_arm_rates,
)


@dataclass(frozen=True)
class RawTrainingConfig:
    """Prospectively frozen optimizer and schedule contract."""

    updates: int = 3000
    effective_batch: int = 64
    microbatch: int = 8
    accumulation: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    checkpoint_interval: int = 100
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.updates <= 0 or self.effective_batch != 64:
            raise ValueError("Stage 64 requires a positive budget and effective batch 64")
        if (self.microbatch, self.accumulation) not in {(8, 8), (4, 16)}:
            raise ValueError("only preregistered Stage 64 microbatch plans are allowed")
        if self.microbatch * self.accumulation != self.effective_batch:
            raise ValueError("microbatch accumulation must equal the effective batch")
        if (self.learning_rate, self.weight_decay) != (3e-4, 1e-4):
            raise ValueError("Stage 64 AdamW hyperparameters are frozen")


@dataclass(frozen=True)
class TrainSupervision:
    """Targets and weights intentionally excluded from :class:`RawModelBatch`."""

    target_phase: np.ndarray
    global_mass: np.ndarray

    def __post_init__(self) -> None:
        target = np.asarray(self.target_phase)
        mass = np.asarray(self.global_mass)
        if target.ndim != 1 or mass.shape != target.shape or len(target) == 0:
            raise ValueError("Stage 64 supervision vectors are misaligned")
        if not np.isfinite(target).all() or not np.isfinite(mass).all() or np.any(mass < 0):
            raise ValueError("Stage 64 supervision must be finite with nonnegative mass")
        if not np.isclose(mass.sum(), 1.0, atol=1e-10):
            raise ValueError("global macro mass must sum to one")


class IndexableCounts(Protocol):
    """Minimal contract accepted by the trainer for zero-copy indexed memmaps."""

    shape: tuple[int, ...]

    def __getitem__(self, index: np.ndarray) -> np.ndarray: ...


class IndexedCounts:
    """Map fold-local row indices to a shared count memmap without copying it."""

    def __init__(self, source: np.ndarray, indices: np.ndarray) -> None:
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices), *source.shape[1:])

    def __getitem__(self, index: np.ndarray) -> np.ndarray:
        return np.asarray(self.source[self.indices[np.asarray(index, dtype=np.int64)]])


def deterministic_schedule(rows: int, updates: int, batch: int, seed: int) -> np.ndarray:
    """Build uniform complete-row permutations without sequence grouping."""

    if min(rows, updates, batch) <= 0:
        raise ValueError("schedule dimensions must be positive")
    rng = np.random.default_rng(seed)
    needed = updates * batch
    chunks: list[np.ndarray] = []
    size = 0
    while size < needed:
        permutation = rng.permutation(rows).astype(np.int64)
        chunks.append(permutation)
        size += len(permutation)
    return np.concatenate(chunks)[:needed].reshape(updates, batch)


def fixed_derangements(tokens: list[str], seed: int) -> np.ndarray:
    """Return deterministic Sattolo permutations ``[N,2,16]``."""

    output = np.empty((len(tokens), 2, 16), dtype=np.int64)
    for row, token in enumerate(tokens):
        for window in range(2):
            key = hashlib.sha256(f"raw16-perm-v1|{seed}|{window}|{token}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
            value = np.arange(16, dtype=np.int64)
            for index in range(15, 0, -1):
                selected = int(rng.integers(0, index))
                value[index], value[selected] = value[selected], value[index]
            output[row, window] = value
    if np.any(output == np.arange(16)[None, None, :]):
        raise AssertionError("Sattolo permutation contains a fixed point")
    return output


def state_normalization(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the sole outer-train A5 state normalizer."""

    values = np.asarray(state, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("A5 state must be finite [N,3]")
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    if np.any(std < 1e-8):
        raise ValueError("A5 state normalization is degenerate")
    return mean.astype(np.float32), std.astype(np.float32)


def _state_dict_sha256(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available() and value["torch_cuda"]:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.is_file():
        previous = path.with_suffix(path.suffix + ".previous")
        if previous.exists():
            previous.unlink()
        os.replace(path, previous)
    os.replace(temporary, path)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _telemetry(device: torch.device, update: int, updates_per_second: float) -> dict[str, Any]:
    process = psutil.Process()
    result: dict[str, Any] = {
        "time_ns": time.time_ns(),
        "update": update,
        "updates_per_second": updates_per_second,
        "rss_bytes": process.memory_info().rss,
        "host_available_bytes": psutil.virtual_memory().available,
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
                "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            }
        )
    return result


def train_raw_time_residual(
    *,
    counts: IndexableCounts,
    durations_s: np.ndarray,
    times: np.ndarray,
    valid_patches: np.ndarray,
    a5_state: np.ndarray,
    sample_tokens: list[str],
    supervision: TrainSupervision,
    rate_mean: np.ndarray,
    rate_std: np.ndarray,
    arm: RawArm,
    seed: int,
    outer_fold: int,
    output_dir: Path,
    device: torch.device,
    identity: dict[str, Any],
    config: RawTrainingConfig | None = None,
    resume: bool = False,
    stop_after_updates: int | None = None,
) -> dict[str, Any]:
    """Train one arm/fold to the fixed endpoint; no outer data is accepted."""

    config = config or RawTrainingConfig()
    rows = len(a5_state)
    if counts.shape != (rows, 2, 16, 2, 64, 64):
        raise ValueError("training counts are not row-aligned")
    if durations_s.shape != (rows, 2, 16) or times.shape != (rows, 6):
        raise ValueError("training observation metadata is not row-aligned")
    if valid_patches.shape != (rows, 16) or len(sample_tokens) != rows:
        raise ValueError("training mask/token metadata is not row-aligned")
    if len(supervision.target_phase) != rows:
        raise ValueError("training supervision is not row-aligned")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = RawTimeResidual().to(device)
    if sum(parameter.numel() for parameter in model.parameters()) >= 250_000:
        raise ValueError("RAW16-MTR exceeds the preregistered parameter cap")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = deterministic_schedule(
        rows, config.updates, config.effective_batch, seed * 10 + outer_fold
    )
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()
    state_mean, state_std = state_normalization(a5_state)
    permutations = fixed_derangements(sample_tokens, seed)
    init_sha = _state_dict_sha256(model)
    output_dir.mkdir(parents=True, exist_ok=resume)
    checkpoint_path = output_dir / "checkpoint_last.pt"
    progress_path = output_dir / "progress.jsonl"
    start_update = 0
    loss_history: list[dict[str, float | int]] = []
    if checkpoint_path.is_file():
        if not resume:
            raise FileExistsError("checkpoint exists but resume was not requested")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected_identity = {**identity, "arm": arm, "seed": seed, "outer_fold": outer_fold}
        if checkpoint["identity"] != expected_identity or checkpoint["config"] != asdict(config):
            raise ValueError("Stage 64 checkpoint identity/config mismatch")
        if checkpoint["schedule_sha256"] != schedule_sha:
            raise ValueError("Stage 64 checkpoint schedule mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _restore_rng(checkpoint["rng"])
        start_update = int(checkpoint["completed_updates"])
        loss_history = list(checkpoint["loss_history"])
        if progress_path.is_file():
            records = [
                json.loads(line)
                for line in progress_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if any(int(record["update"]) > start_update for record in records):
                orphan = output_dir / f"orphaned_progress_{time.time_ns()}.jsonl"
                progress_path.replace(orphan)
                progress_path.write_text(
                    "".join(
                        json.dumps(record, allow_nan=False) + "\n"
                        for record in records
                        if int(record["update"]) <= start_update
                    ),
                    encoding="utf-8",
                )
    elif resume and any(output_dir.iterdir()):
        raise ValueError("resume root contains progress without a valid checkpoint")
    model.train()
    target = np.asarray(supervision.target_phase, dtype=np.float32)
    mass = np.asarray(supervision.global_mass, dtype=np.float32)
    telemetry_path = output_dir / "telemetry.jsonl"
    last_telemetry = 0.0
    began = time.perf_counter()
    endpoint_update = config.updates if stop_after_updates is None else int(stop_after_updates)
    if not start_update < endpoint_update <= config.updates:
        raise ValueError("stop_after_updates must advance within the frozen total budget")
    for update in range(start_update, endpoint_update):
        optimizer.zero_grad(set_to_none=True)
        batch_indices = schedule[update]
        update_loss = 0.0
        for start in range(0, config.effective_batch, config.microbatch):
            index = batch_indices[start : start + config.microbatch]
            raw = torch.as_tensor(np.asarray(counts[index]), device=device)
            duration = torch.as_tensor(durations_s[index], dtype=torch.float32, device=device)
            permutation = torch.as_tensor(permutations[index], dtype=torch.long, device=device)
            normalized = normalize_arm_rates(
                raw,
                duration,
                torch.as_tensor(rate_mean, dtype=torch.float32, device=device),
                torch.as_tensor(rate_std, dtype=torch.float32, device=device),
                arm=arm,
                permutations=permutation if arm == "S64-PERM-L1" else None,
            )
            state_value = torch.as_tensor(a5_state[index], dtype=torch.float32, device=device)
            model_batch = RawModelBatch(
                normalized_rates=normalized,
                a5_phase=state_value[:, 0],
                normalized_a5_state=(
                    state_value - torch.as_tensor(state_mean, dtype=torch.float32, device=device)
                )
                / torch.as_tensor(state_std, dtype=torch.float32, device=device),
                times=torch.as_tensor(times[index], dtype=torch.float32, device=device),
                valid_patches=torch.as_tensor(
                    valid_patches[index], dtype=torch.bool, device=device
                ),
            )
            output = model(model_batch)
            target_value = torch.as_tensor(target[index], device=device)
            mass_value = torch.as_tensor(mass[index], device=device)
            elements = rows * mass_value * torch.abs(output.benchmark_phase - target_value)
            loss = elements.sum() / config.effective_batch
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite Stage 64 loss at update {update}")
            loss.backward()
            update_loss += float(loss.detach().cpu())
        for parameter in model.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"non-finite Stage 64 gradient at update {update}")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError(f"non-finite Stage 64 parameter at update {update}")
        loss_history.append({"update": update + 1, "loss": update_loss})
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(loss_history[-1], allow_nan=False) + "\n")
        now = time.perf_counter()
        if now - last_telemetry >= 5.0 or update + 1 == config.updates:
            rate = (update + 1 - start_update) / max(now - began, 1e-9)
            with telemetry_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(_telemetry(device, update + 1, rate)) + "\n")
            last_telemetry = now
        if (update + 1) % config.checkpoint_interval == 0 or update + 1 == endpoint_update:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": {"type": "constant", "last_update": update + 1},
                "rng": _rng_state(),
                "completed_updates": update + 1,
                "next_schedule_index": update + 1,
                "schedule_sha256": schedule_sha,
                "identity": {**identity, "arm": arm, "seed": seed, "outer_fold": outer_fold},
                "config": asdict(config),
                "rate_mean": np.asarray(rate_mean),
                "rate_std": np.asarray(rate_std),
                "state_mean": state_mean,
                "state_std": state_std,
                "initialization_sha256": init_sha,
                "loss_history": loss_history,
            }
            _atomic_torch_save(checkpoint_path, checkpoint)
    checkpoint_sha = _checkpoint_sha256(checkpoint_path)
    frozen = endpoint_update == config.updates
    manifest = {
        "artifact_type": "scientific_recovery_v9_stage64_frozen_endpoint_v1",
        "status": "frozen" if frozen else "checkpointed",
        "arm": arm,
        "seed": seed,
        "outer_fold": outer_fold,
        "completed_updates": endpoint_update,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": checkpoint_sha,
        "initialization_sha256": init_sha,
        "schedule_sha256": schedule_sha,
        "normalization": {
            "rate_mean": np.asarray(rate_mean).tolist(),
            "rate_std": np.asarray(rate_std).tolist(),
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
        },
        "config": asdict(config),
        "outer_evaluation_opened": False,
    }
    manifest_name = "frozen_manifest.json" if frozen else "checkpoint_manifest.json"
    (output_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_frozen_raw_endpoint(
    path: Path, *, device: torch.device
) -> tuple[RawTimeResidual, dict[str, Any], dict[str, Any]]:
    """Load a fixed-update endpoint only after its manifest hash check."""

    manifest = json.loads((path / "frozen_manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(str(manifest["checkpoint_path"]))
    if manifest.get("status") != "frozen" or manifest.get("completed_updates") != 3000:
        raise ValueError("Stage 64 endpoint is not the fixed update3000 checkpoint")
    if _checkpoint_sha256(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise ValueError("Stage 64 frozen checkpoint SHA mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RawTimeResidual().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, manifest


__all__ = [
    "IndexedCounts",
    "RawTrainingConfig",
    "TrainSupervision",
    "deterministic_schedule",
    "fixed_derangements",
    "load_frozen_raw_endpoint",
    "state_normalization",
    "train_raw_time_residual",
]
