"""Deterministic V8 training utilities for causal-scale JEPA attribution.

The trainer owns only label-free t0/t1/t2 batches.  It is intentionally not a
TTC trainer: supervised fine-tuning and the nested router are separate phases.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from e_jepa_ttc.models.causal_scale_jepa_v8 import (
    JEPA_VIEW_NAMES,
    CausalScaleJEPAV8,
    CausalScaleJEPAV8Output,
)


@dataclass(frozen=True)
class ScientificRecoveryV8JEPATrainerConfig:
    """Closed V8 optimization and deterministic shuffled-future controls."""

    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    total_updates: int = 1_000
    seed: int = 7
    gradient_clip_norm: float | None = 1.0
    shuffled_future: bool = False

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if self.total_updates <= 0:
            raise ValueError("total_updates must be positive.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when enabled.")

    def sha256(self) -> str:
        """Return a stable hash used to reject incompatible resume payloads."""

        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def deterministic_shuffled_future(
    future_t2: torch.Tensor,
    *,
    track_ids: Sequence[Hashable],
    seed: int,
    update_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a seeded complete cross-track derangement for the D4 control.

    The operation preserves shape, dtype, device, batch count and therefore the
    number of forward/backward operations. It permutes only future event input;
    labels are not accepted by this API. Each future is paired with a different
    row from a different track, or the function fails closed before training.
    """

    if future_t2.ndim != 4:
        raise ValueError("future_t2 must have shape [B,C,H,W].")
    batch = future_t2.shape[0]
    if batch < 2:
        raise ValueError("D4 shuffled-future control requires batch size >= 2.")
    normalized_track_ids = _validate_track_ids(track_ids, batch=batch)
    permutation = _cross_track_derangement(
        normalized_track_ids, seed=seed, update_index=update_index
    )
    return future_t2.index_select(0, permutation.to(future_t2.device)), permutation


def _validate_track_ids(track_ids: Sequence[Hashable], *, batch: int) -> tuple[Hashable, ...]:
    """Validate row-aligned, hashable, stable-JSON track identities."""

    if isinstance(track_ids, (str, bytes)):
        raise ValueError("D4 track_ids must be a row-aligned sequence, not a scalar string.")
    if len(track_ids) != batch:
        raise ValueError(
            "D4 track_ids must be row-aligned with future_t2: "
            f"got {len(track_ids)} ids for {batch} rows."
        )
    normalized = tuple(track_ids)
    for index, track_id in enumerate(normalized):
        try:
            hash(track_id)
        except TypeError as error:
            raise ValueError(f"D4 track_ids[{index}] must be hashable.") from error
        try:
            json.dumps(track_id, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"D4 track_ids[{index}] must have a stable JSON serialization."
            ) from error
    return normalized


def _cross_track_derangement(
    track_ids: tuple[Hashable, ...], *, seed: int, update_index: int
) -> torch.Tensor:
    """Solve the deterministic bipartite D4 assignment without retry sampling."""

    batch = len(track_ids)
    track_counts: dict[Hashable, int] = {}
    for track_id in track_ids:
        track_counts[track_id] = track_counts.get(track_id, 0) + 1
    largest_track = max(track_counts.values())
    if largest_track > batch - largest_track:
        raise ValueError(
            "D4 has no complete cross-track derangement: one track dominates the batch "
            f"({largest_track}/{batch} rows). Construct cross-track batches before training."
        )

    # This seeded ordering only resolves otherwise equivalent complete matchings.
    # The matching itself is exhaustive and deterministic; it never relies on retry.
    randomizer = random.Random((int(seed) * 1_000_003 + int(update_index)) % (2**63 - 1))
    target_order = list(range(batch))
    source_tie_order = list(range(batch))
    randomizer.shuffle(target_order)
    randomizer.shuffle(source_tie_order)
    source_rank = {source: rank for rank, source in enumerate(source_tie_order)}
    candidates = {
        source: [
            target
            for target in target_order
            if target != source and track_ids[source] != track_ids[target]
        ]
        for source in range(batch)
    }
    source_order = sorted(
        range(batch), key=lambda source: (len(candidates[source]), source_rank[source])
    )
    target_to_source = [-1] * batch

    def assign(source: int, visited: set[int]) -> bool:
        for target in candidates[source]:
            if target in visited:
                continue
            visited.add(target)
            prior_source = target_to_source[target]
            if prior_source == -1 or assign(prior_source, visited):
                target_to_source[target] = source
                return True
        return False

    for source in source_order:
        if not assign(source, set()):
            raise ValueError(
                "D4 has no complete cross-track derangement for these track identities; "
                "construct batches with enough distinct tracks."
            )
    permutation = [-1] * batch
    for target, source in enumerate(target_to_source):
        if source < 0:
            raise RuntimeError("D4 complete matching returned an unassigned future target.")
        permutation[source] = target
    if any(target < 0 for target in permutation):
        raise RuntimeError("D4 complete matching returned an unassigned source row.")
    return torch.tensor(permutation, dtype=torch.long)


def assert_all_online_encoder_gradients(model: CausalScaleJEPAV8) -> dict[str, float]:
    """Fail closed unless every online endpoint parameter has finite nonzero grad."""

    failures: list[str] = []
    magnitudes: dict[str, float] = {}
    for name, parameter in model.online_encoder.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            failures.append(f"{name}:missing")
            continue
        if not bool(torch.isfinite(gradient).all()):
            failures.append(f"{name}:nonfinite")
            continue
        magnitude = float(gradient.detach().abs().max())
        magnitudes[name] = magnitude
        if magnitude <= 0.0:
            failures.append(f"{name}:zero")
    if failures:
        raise RuntimeError(
            "V8 JEPA requires finite nonzero gradients for every online encoder parameter: "
            + ", ".join(failures)
        )
    return magnitudes


def _capture_rng_state() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"JEPA checkpoint lacks RNG state: {sorted(missing)}")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in payload:
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


class ScientificRecoveryV8JEPATrainer:
    """Minimal, serializable JEPA trainer with per-view fail-closed health gates."""

    artifact_type = "scientific_recovery_v8_causal_scale_jepa_checkpoint_v1"

    def __init__(
        self,
        model: CausalScaleJEPAV8,
        config: ScientificRecoveryV8JEPATrainerConfig | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config or ScientificRecoveryV8JEPATrainerConfig()
        self.optimizer = optimizer or torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.update_count = 0
        self.collapse_counts = {name: 0 for name in JEPA_VIEW_NAMES}
        self.last_permutation: torch.Tensor | None = None

    def compute_manifest(self) -> dict[str, Any]:
        """Record equal-compute semantics for matched and shuffled D4 runs."""

        return {
            "contract": "v8_jepa_equal_compute",
            "seed": self.config.seed,
            "total_updates": self.config.total_updates,
            "shuffled_future": self.config.shuffled_future,
            "matched_compute": {
                "outer_train_pool": "identical",
                "batches": "identical",
                "update_count": "identical",
                "model": "identical",
                "optimizer": "identical",
                "masking": "identical",
                "ema_schedule": "identical",
                "augmentations": "identical",
            },
            "only_difference_from_matched": "future_pairing",
            "d4_future_pairing": {
                "deterministic_seed_and_update": True,
                "no_fixed_points": True,
                "cross_track": True,
                "target_marginal_preserved": True,
            },
            "uses_labels": False,
        }

    def _enforce_health(self, output: CausalScaleJEPAV8Output) -> None:
        for name in JEPA_VIEW_NAMES:
            collapsed = (
                output.health[name]["collapsed_dimension_fraction"]
                > self.model.config.collapse_fraction_threshold
            )
            self.collapse_counts[name] = self.collapse_counts[name] + 1 if collapsed else 0
            if self.collapse_counts[name] >= self.model.config.collapse_patience:
                raise RuntimeError(
                    "V8 JEPA collapse persisted for "
                    f"{self.collapse_counts[name]} checks in {name!r} view "
                    f"({output.health[name]['collapsed_dimension_fraction']:.1%} dims)."
                )

    def step(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        t2: torch.Tensor,
        *,
        track_ids: Sequence[Hashable] | None = None,
    ) -> dict[str, Any]:
        """Perform exactly one label-free JEPA update and one aligned EMA update."""

        if self.update_count >= self.config.total_updates:
            raise RuntimeError(
                "JEPA update budget exhausted; refusing an unregistered extra update."
            )
        future = t2
        self.last_permutation = None
        if self.config.shuffled_future:
            if track_ids is None:
                raise ValueError(
                    "D4 shuffled-future training requires row-aligned track_ids; "
                    "track identities must not enter the model."
                )
            future, self.last_permutation = deterministic_shuffled_future(
                t2,
                track_ids=track_ids,
                seed=self.config.seed,
                update_index=self.update_count,
            )
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(t0, t1, future)
        loss = output.loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("V8 JEPA produced a non-finite label-free loss.")
        loss.backward()
        gradient_magnitudes = assert_all_online_encoder_gradients(self.model)
        if self.config.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        momentum = self.model.update_target_ema(
            self.update_count, total_updates=self.config.total_updates
        )
        self.update_count += 1
        self._enforce_health(output)
        return {
            "loss": float(loss.detach()),
            "losses": {name: float(value.detach()) for name, value in output.losses.items()},
            "health": output.health,
            "ema_momentum": momentum,
            "gradient_magnitudes": gradient_magnitudes,
            "shuffled_future": self.config.shuffled_future,
            "permutation": (
                self.last_permutation.detach().cpu().tolist()
                if self.last_permutation is not None
                else None
            ),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """Create a fully serializable state sufficient for deterministic resume."""

        compute_manifest = self.compute_manifest()
        compute_manifest_sha256 = hashlib.sha256(
            json.dumps(compute_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return {
            "artifact_type": self.artifact_type,
            "trainer_config": asdict(self.config),
            "trainer_config_sha256": self.config.sha256(),
            "model_config": asdict(self.model.config),
            "model_manifest": self.model.encoder_manifest(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "collapse_counts": dict(self.collapse_counts),
            "compute_manifest": compute_manifest,
            "compute_manifest_sha256": compute_manifest_sha256,
            "rng_state": _capture_rng_state(),
        }

    def load_checkpoint_state(self, payload: Mapping[str, Any]) -> None:
        """Restore state strictly, including RNG and target-eval/frozen invariants."""

        if payload.get("artifact_type") != self.artifact_type:
            raise ValueError("Checkpoint has the wrong V8 causal-scale JEPA artifact type.")
        if payload.get("trainer_config_sha256") != self.config.sha256():
            raise ValueError("Checkpoint trainer configuration hash differs from current trainer.")
        model_config = payload.get("model_config")
        if model_config != asdict(self.model.config):
            raise ValueError("Checkpoint model configuration differs from current JEPA model.")
        state = payload.get("model_state_dict")
        optimizer = payload.get("optimizer_state_dict")
        if not isinstance(state, Mapping) or not isinstance(optimizer, Mapping):
            raise ValueError("Checkpoint lacks model or optimizer state.")
        saved_compute_manifest = payload.get("compute_manifest")
        saved_compute_manifest_sha256 = payload.get("compute_manifest_sha256")
        expected_compute_manifest = self.compute_manifest()
        expected_compute_manifest_sha256 = hashlib.sha256(
            json.dumps(expected_compute_manifest, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if (
            saved_compute_manifest != expected_compute_manifest
            or saved_compute_manifest_sha256 != expected_compute_manifest_sha256
        ):
            raise ValueError("Checkpoint D4 equal-compute manifest differs from current trainer.")
        self.model.load_state_dict(state, strict=True)
        self.optimizer.load_state_dict(optimizer)
        manifest = payload.get("model_manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError("Checkpoint lacks the V8 JEPA model manifest.")
        saved_online = manifest.get("online_encoder_sha256")
        saved_target = manifest.get("target_encoder_sha256")
        current = self.model.encoder_manifest()
        if (
            saved_online != current["online_encoder_sha256"]
            or saved_target != current["target_encoder_sha256"]
        ):
            raise RuntimeError("Checkpoint encoder hashes do not match restored state.")
        source_hash = manifest.get("source_encoder_sha256")
        if not isinstance(source_hash, str):
            raise ValueError("Checkpoint lacks source encoder SHA provenance.")
        self.model.source_encoder_sha256 = source_hash
        update_count = int(payload.get("update_count", -1))
        if not 0 <= update_count <= self.config.total_updates:
            raise ValueError("Checkpoint update_count lies outside the configured update budget.")
        counts = payload.get("collapse_counts")
        if not isinstance(counts, Mapping) or set(counts) != set(JEPA_VIEW_NAMES):
            raise ValueError("Checkpoint has invalid per-view collapse counts.")
        self.update_count = update_count
        self.collapse_counts = {name: int(counts[name]) for name in JEPA_VIEW_NAMES}
        _restore_rng_state(payload.get("rng_state", {}))
        for parameter in self.model.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.model.target_encoder.eval()

    def save_checkpoint(self, path: str | Path) -> None:
        """Atomically persist a deterministic resume checkpoint."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(self.checkpoint_state(), temporary)
        temporary.replace(destination)

    def load_checkpoint(
        self, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> None:
        """Load a checkpoint written by :meth:`save_checkpoint`."""

        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("V8 JEPA checkpoint must contain a mapping.")
        self.load_checkpoint_state(payload)


__all__ = [
    "ScientificRecoveryV8JEPATrainer",
    "ScientificRecoveryV8JEPATrainerConfig",
    "assert_all_online_encoder_gradients",
    "deterministic_shuffled_future",
]
