"""Matched, fixed-budget Stage 62 local-field training."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.data.stage61_pair_feature_cache import LocalTemporalFieldBatch
from e_jepa_ttc.evaluation.collision_clock_protocol import tensor_state_sha256
from e_jepa_ttc.models.local_temporal_phase_field import LocalTemporalPhaseField
from e_jepa_ttc.training.incremental_residual import deterministic_sequence_grouped_schedule


@dataclass(frozen=True)
class LocalFieldTrainingConfig:
    """Frozen optimizer contract shared by GLOBAL, LOCAL, and SHUFFLE."""

    arm_id: str
    seed: int
    batch_size: int = 256
    update_budget: int = 2000
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    checkpoint_policy: str = "last_update_fixed_budget"

    def __post_init__(self) -> None:
        if self.arm_id not in {"X2-GLOBALPOOL", "X2-LOCALFIELD", "X2-SHUFFLEFIELD"}:
            raise ValueError("unauthorized X2 training arm")
        if (self.batch_size, self.update_budget) != (256, 2000):
            raise ValueError("X2 batch size/update budget are frozen to 256/2000")
        if (self.learning_rate, self.weight_decay) != (3e-4, 1e-4):
            raise ValueError("X2 optimizer recipe drifted")
        if self.checkpoint_policy != "last_update_fixed_budget":
            raise ValueError("X2 requires the fixed last-update checkpoint")


def global_pool_field(features: np.ndarray) -> np.ndarray:
    """Broadcast global motion summaries, zero coordinates, preserve A5 state."""

    result = np.asarray(features, dtype=np.float32).copy()
    result[:, :, :29] = result[:, :, :29].mean(axis=1, keepdims=True)
    result[:, :, 29:31] = 0.0
    return result


def time_swap_field(features: np.ndarray) -> np.ndarray:
    """Exchange m01/m12 and reverse the nine temporal differences."""

    result = np.asarray(features, dtype=np.float32).copy()
    first = result[:, :, :10].copy()
    second = result[:, :, 10:20].copy()
    result[:, :, :10] = second
    result[:, :, 10:20] = first
    result[:, :, 20:29] *= -1.0
    return result


def derange_cross_track(
    features: np.ndarray,
    *,
    sequence_ids: list[str],
    track_ids: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derange only the 29 local fields within sequence and between tracks."""

    values = np.asarray(features, dtype=np.float32)
    sequences = np.asarray(sequence_ids, dtype=str)
    tracks = np.asarray(track_ids, dtype=str)
    permutation = np.arange(len(values), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for sequence in sorted(np.unique(sequences).tolist()):
        indices = np.flatnonzero(sequences == sequence)
        found: np.ndarray | None = None
        for _ in range(10_000):
            candidate = rng.permutation(indices)
            if np.all(candidate != indices) and np.all(tracks[candidate] != tracks[indices]):
                found = candidate
                break
        if found is None:
            raise ValueError(f"cannot construct cross-track derangement for {sequence}")
        permutation[indices] = found
    result = values.copy()
    result[:, :, :29] = values[permutation, :, :29]
    return result, permutation


def train_local_field(
    *,
    patch_features: np.ndarray,
    patch_valid: np.ndarray,
    a5_phase: np.ndarray,
    target_phase: np.ndarray,
    sample_weight: np.ndarray,
    sequence_ids: list[str],
    sample_tokens: list[str],
    config: LocalFieldTrainingConfig,
    initial_state: dict[str, torch.Tensor],
    output_dir: Path,
    device: torch.device,
    identity: dict[str, Any],
    resume: bool = False,
) -> tuple[LocalTemporalPhaseField, dict[str, Any]]:
    """Train a matched X2 arm without exposing any outer-dev object."""

    features = np.asarray(patch_features, dtype=np.float32)
    valid = np.asarray(patch_valid, dtype=np.bool_)
    base = np.asarray(a5_phase, dtype=np.float32).reshape(-1)
    target = np.asarray(target_phase, dtype=np.float32).reshape(-1)
    weights = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    LocalTemporalFieldBatch(
        torch.from_numpy(features), torch.from_numpy(valid), torch.from_numpy(base)
    )
    if not np.isfinite(target).all() or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("X2 supervision is invalid")
    schedule, schedule_sha = deterministic_sequence_grouped_schedule(
        sequence_ids,
        sample_tokens,
        seed=config.seed,
        batch_size=config.batch_size,
        updates=config.update_budget,
    )
    output_dir.mkdir(parents=True, exist_ok=resume)
    checkpoint = output_dir / "model_last.pt"
    progress = output_dir / "progress.jsonl"
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = LocalTemporalPhaseField().to(device)
    model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    initialization_sha = tensor_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    run_identity = {
        **identity,
        "config": asdict(config),
        "initialization_sha256": initialization_sha,
        "batch_schedule_sha256": schedule_sha,
    }
    start, losses = 0, []
    if resume:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload["identity"] != run_identity:
            raise ValueError("X2 resume identity mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start = int(payload["completed_updates"])
        losses = [float(value) for value in payload["losses"]]
    elif checkpoint.exists() or progress.exists():
        raise FileExistsError("X2 output exists; pass resume explicitly")
    tensors = (
        torch.from_numpy(features).to(device),
        torch.from_numpy(valid).to(device),
        torch.from_numpy(base).to(device),
        torch.from_numpy(target).to(device),
        torch.from_numpy(weights).to(device),
    )
    milestones = {0, 250, 500, 1000, 1500, 2000}

    def save(update: int, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "artifact_type": "stage62_local_field_checkpoint_v1",
                "identity": run_identity,
                "completed_updates": update,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
            },
            path,
        )

    if start == 0:
        save(0, output_dir / "milestones" / "update-000000.pt")
    began = time.perf_counter()
    for update in range(start, config.update_budget):
        indices = torch.as_tensor(schedule[update], dtype=torch.long, device=device)
        batch = LocalTemporalFieldBatch(
            tensors[0][indices], tensors[1][indices], tensors[2][indices]
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch).benchmark_phase
        per_row = functional.smooth_l1_loss(prediction, tensors[3][indices], reduction="none")
        normalized = tensors[4][indices] / tensors[4][indices].mean().clamp_min(1e-12)
        loss = torch.mean(normalized * per_row)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("X2 loss became non-finite")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        completed = update + 1
        with progress.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"update": completed, "loss": losses[-1]}) + "\n")
        if completed % 100 == 0 or completed in milestones:
            save(completed, checkpoint)
        if completed in milestones:
            save(completed, output_dir / "milestones" / f"update-{completed:06d}.pt")
    elapsed = time.perf_counter() - began
    manifest = sign_artifact(
        {
            "artifact_type": "stage62_local_field_training_v1",
            "status": "completed",
            "identity": run_identity,
            "completed_updates": config.update_budget,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": compute_file_hash(str(checkpoint)),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "elapsed_seconds": elapsed,
            "outer_dev_available_to_trainer": False,
            "outer_dev_used_for_selection": False,
        }
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return model.eval(), manifest


def load_local_field(path: Path, *, device: torch.device) -> LocalTemporalPhaseField:
    """Load only a completed update-2000 X2 checkpoint."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("completed_updates") != 2000:
        raise ValueError("X2 checkpoint is not the fixed-budget endpoint")
    model = LocalTemporalPhaseField().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


__all__ = [
    "LocalFieldTrainingConfig",
    "derange_cross_track",
    "global_pool_field",
    "load_local_field",
    "time_swap_field",
    "train_local_field",
]
