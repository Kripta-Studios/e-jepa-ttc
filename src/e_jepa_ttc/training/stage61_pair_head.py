"""Fixed-budget cached PAIR-head training for Stage 61."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch, PairSupervisionBatch
from e_jepa_ttc.models.collision_clock_math import (
    benchmark_phase_to_inverse_ttc,
    neutral_raw_phase,
    phase_lower_bound,
)
from e_jepa_ttc.training.incremental_residual import deterministic_sequence_grouped_schedule


@dataclass(frozen=True)
class PairHeadTrainingConfig:
    """The exact X0-PAIR-U recipe, applied to a sealed 133-D cache."""

    seed: int
    batch_size: int = 32
    update_budget: int = 6840
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    dropout: float = 0.05
    metric_delta_t_s: float = 0.1
    minimum_abs_prediction_ttc_s: float = 0.1
    checkpoint_policy: str = "last_update_fixed_budget"

    def __post_init__(self) -> None:
        if (self.batch_size, self.update_budget) != (32, 6840):
            raise ValueError("PAIR batch size/update budget are frozen to 32/6840")
        if (self.learning_rate, self.weight_decay, self.dropout) != (3e-4, 1e-4, 0.05):
            raise ValueError("PAIR optimizer/dropout recipe drifted")
        if self.checkpoint_policy != "last_update_fixed_budget":
            raise ValueError("PAIR requires the fixed last-update checkpoint")


class CachedPairDirectPhase(nn.Module):
    """Public cache-only equivalent of the X0 direct phase head."""

    def __init__(self, *, dropout: float = 0.05) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(133),
            nn.Linear(133, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(
            final.bias,
            neutral_raw_phase(metric_delta_t_s=0.1, minimum_abs_prediction_ttc_s=0.1),
        )

    def raw_phase(self, batch: PairFeatureBatch) -> torch.Tensor:
        """Return the unconstrained scalar emitted by the historical head."""

        return self.network(batch.features).squeeze(-1)

    def forward(self, batch: PairFeatureBatch) -> torch.Tensor:
        lower = phase_lower_bound(metric_delta_t_s=0.1, minimum_abs_prediction_ttc_s=0.1)
        return lower + functional.softplus(self.raw_phase(batch))

    def predict_ttc(self, batch: PairFeatureBatch) -> torch.Tensor:
        """Map the safe benchmark phase to unclipped scientific TTC."""

        return torch.reciprocal(benchmark_phase_to_inverse_ttc(self(batch), metric_delta_t_s=0.1))


def train_pair_head(
    *,
    features: np.ndarray,
    target_phase: np.ndarray,
    sample_weight: np.ndarray,
    sequence_ids: list[str],
    sample_tokens: list[str],
    config: PairHeadTrainingConfig,
    output_dir: Path,
    device: torch.device,
    identity: dict[str, Any],
    resume: bool = False,
) -> tuple[CachedPairDirectPhase, dict[str, Any]]:
    """Train only the PAIR head; no dev arrays are accepted by this API."""

    feature = np.asarray(features, dtype=np.float32)
    target = np.asarray(target_phase, dtype=np.float32).reshape(-1)
    weights = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
    PairFeatureBatch(torch.from_numpy(feature))
    PairSupervisionBatch(torch.from_numpy(target), torch.from_numpy(weights))
    if len(feature) != len(sequence_ids) or len(feature) != len(sample_tokens):
        raise ValueError("PAIR cache identities are row-misaligned")
    schedule, schedule_sha = deterministic_sequence_grouped_schedule(
        sequence_ids,
        sample_tokens,
        seed=config.seed,
        batch_size=config.batch_size,
        updates=config.update_budget,
    )
    run_identity = {**identity, "config": asdict(config), "batch_schedule_sha256": schedule_sha}
    output_dir.mkdir(parents=True, exist_ok=resume)
    checkpoint = output_dir / "model_last.pt"
    progress = output_dir / "progress.jsonl"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = CachedPairDirectPhase(dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    start, losses = 0, []
    if resume:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if payload["identity"] != run_identity:
            raise ValueError("PAIR resume identity mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        start = int(payload["completed_updates"])
        losses = [float(value) for value in payload["losses"]]
        torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        if torch.cuda.is_available() and payload["torch_cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state"])
    elif checkpoint.exists() or progress.exists():
        raise FileExistsError("PAIR output exists; pass resume explicitly")
    tensors = (
        torch.from_numpy(feature).to(device),
        torch.from_numpy(target).to(device),
        torch.from_numpy(weights).to(device),
    )
    milestones = {0, 250, 500, 1000, 2000, 4000, 6840}

    def save(update: int, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "artifact_type": "stage61_pair_head_checkpoint_v1",
                "identity": run_identity,
                "completed_updates": update,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "losses": losses,
            },
            path,
        )

    if start == 0:
        save(0, output_dir / "milestones" / "update-000000.pt")
    began = time.perf_counter()
    for update in range(start, config.update_budget):
        indices = torch.as_tensor(schedule[update], dtype=torch.long, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(PairFeatureBatch(tensors[0][indices]))
        error = functional.smooth_l1_loss(prediction, tensors[1][indices], reduction="none")
        loss = error.mean()  # Exact X0 PAIR reduction; weights are intentionally unused.
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("PAIR loss became non-finite")
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
            "artifact_type": "stage61_pair_head_training_v1",
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


def load_pair_head(path: Path, *, device: torch.device) -> CachedPairDirectPhase:
    """Load a completed Stage 61 cached PAIR checkpoint."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("completed_updates") != 6840:
        raise ValueError("PAIR checkpoint is not the fixed-budget endpoint")
    model = CachedPairDirectPhase().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


__all__ = [
    "CachedPairDirectPhase",
    "PairHeadTrainingConfig",
    "load_pair_head",
    "train_pair_head",
]
