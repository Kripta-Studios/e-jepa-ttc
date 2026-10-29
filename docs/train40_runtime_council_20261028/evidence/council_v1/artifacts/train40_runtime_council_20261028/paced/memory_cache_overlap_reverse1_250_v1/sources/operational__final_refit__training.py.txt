"""Durable cached-head training for the prospective all-TRAIN40 final chain."""

from __future__ import annotations

import math
import platform
import random
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.rgb_port.fusion import fusion_loss
from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner, training_loss
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, phase_to_ttc
from e_jepa_ttc.training.incremental_residual import deterministic_sequence_grouped_schedule
from e_jepa_ttc.training.stage61_pair_head import CachedPairDirectPhase
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port.train_producers import ProducerCheckpoint

from .features import validate_cache_receipt
from .recipe import (
    FinalRefitRecipe,
    StageRecipe,
    canonical_sha256,
    ordered_row_identity_sha256,
    sha256,
    verify_bound_sources,
)

CACHE_KEYS = {
    "pair_head": {
        "features",
        "target_phase",
        "group_id",
        "sample_token",
        "sequence_id",
        "role",
    },
    "temporal_head": {
        "features",
        "timing",
        "valid",
        "expert_phase",
        "target_phase",
        "group_id",
        "sample_token",
        "role",
    },
    "phase_only_head": {
        "features",
        "rgb_width_logits",
        "target_phase",
        "group_id",
        "sample_token",
        "role",
    },
}
TECHNICAL_CALL_UPDATE_CAP = 5
TECHNICAL_LIFETIME_UPDATE_CAP = 10


@dataclass(frozen=True)
class CachedTrainingScope:
    """Explicit population and role for one cache-backed training stage."""

    population: int
    groups: int
    role: str
    experiment_id: str
    split_version: str
    scope_sha256: str
    external_parent_sha256: dict[str, str]


class PhaseOnlyFinal(nn.Module):
    """Exact V16 PHASE_ONLY correction architecture over frozen 340-D states."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(20, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.output = nn.Linear(32, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def inputs(features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 340:
            raise ValueError("PHASE_ONLY expects frozen [B,340] states")
        selected = torch.cat((features[:, 160:170], features[:, 330:340]), dim=-1).clone()
        selected[:, :5] /= 0.03
        selected[:, 10:15] /= 0.03
        return selected

    def forward(self, features: Tensor, rgb_width_logits: Tensor) -> dict[str, Tensor]:
        if rgb_width_logits.shape != (len(features), 2):
            raise ValueError("PHASE_ONLY expects frozen R_CTX width logits [B,2]")
        correction = self.output(self.mlp(self.inputs(features)))
        location = features[:, 160] + 0.03 * correction[:, 0]
        widths = 0.03 * functional.softplus(rgb_width_logits + correction[:, 1:])
        return {
            "raw_location": location,
            "point_phase": emitted_phase(location),
            "ttc": phase_to_ttc(location),
            "q10": (location - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
            "q90": (location + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
            "correction": correction,
        }


def _balanced_mass(
    groups: np.ndarray[Any, Any], expected_groups: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    names = np.asarray(groups).astype(str)
    unique, inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
    if len(unique) != expected_groups or np.any(counts < 1):
        raise ValueError(f"Final training cache must contain exactly {expected_groups} groups")
    # The historical objective multiplies the batch sum by the full population.
    # Unit-sum global mass therefore gives every acquisition group equal weight.
    mass = 1.0 / (len(unique) * counts[inverse])
    return mass.astype(np.float32)


class FinalCache:
    """Strict all-TRAIN cache with optional one-time device residency."""

    def __init__(
        self,
        path: Path,
        stage: StageRecipe,
        recipe: FinalRefitRecipe,
        scope: CachedTrainingScope | None = None,
    ) -> None:
        if stage.kind not in CACHE_KEYS or stage.cache_schema is None:
            raise ValueError("Stage does not consume a final-refit cache")
        receipt_path = path.with_suffix(".json")
        expected_population = recipe.train_queries if scope is None else scope.population
        expected_groups = recipe.train_groups if scope is None else scope.groups
        expected_role = "TRAIN" if scope is None else scope.role
        receipt = validate_cache_receipt(path, stage.cache_schema, expected_role=expected_role)
        if (
            int(receipt.get("rows", -1)) != expected_population
            or int(receipt.get("groups", -1)) != expected_groups
        ):
            raise ValueError("Cache receipt does not bind the frozen final-refit population")
        with np.load(path, allow_pickle=False) as archive:
            self.values = {name: archive[name] for name in archive.files}
        if set(self.values) != CACHE_KEYS[stage.kind]:
            differences = sorted(set(self.values) ^ CACHE_KEYS[stage.kind])
            raise ValueError(f"Cache fields differ: {differences}")
        n = expected_population
        if any(len(value) != n for value in self.values.values()):
            raise ValueError("Cache arrays are row-misaligned")
        if set(np.asarray(self.values["role"]).astype(str)) != {expected_role}:
            raise ValueError(f"Training cache must use the explicit {expected_role} role")
        tokens = np.asarray(self.values["sample_token"]).astype(str)
        if len(set(tokens.tolist())) != n:
            raise ValueError("Final cache sample tokens must be unique")
        if scope is None:
            groups = np.asarray(self.values["group_id"]).astype(str)
            observed_order = ordered_row_identity_sha256(tokens.tolist(), groups.tolist())
            if (
                observed_order != recipe.ordered_row_identity_sha256
                or receipt.get("ordered_row_identity_sha256") != observed_order
                or receipt.get("native_train40_rows_sha256") != recipe.native_train40_rows_sha256
            ):
                raise ValueError(
                    "Cache token/group/native order differs from the frozen TRAIN40 recipe"
                )
        for name in ("features", "target_phase"):
            value = np.asarray(self.values[name])
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite FP32")
        if stage.kind == "pair_head" and self.values["features"].shape != (n, 133):
            raise ValueError("PAIR_R cache must be [N,133]")
        if stage.kind == "temporal_head":
            features = self.values["features"]
            valid = self.values["valid"]
            if (
                features.ndim != 3
                or features.shape[0] != n
                or features.shape[-1] != 17
                or not 1 <= features.shape[1] <= 8
                or self.values["timing"].shape != (*features.shape[:2], 4)
                or valid.shape != features.shape[:2]
                or valid.dtype != np.bool_
                or self.values["expert_phase"].shape != (n, 3)
                or not np.all(valid[:, -1])
                or np.any(valid[:, :-1] & ~valid[:, 1:])
            ):
                raise ValueError("R_CTX cache history schema differs")
        if stage.kind == "phase_only_head" and (
            self.values["features"].shape != (n, 340)
            or self.values["rgb_width_logits"].shape != (n, 2)
        ):
            raise ValueError("PHASE_ONLY cache must contain [N,340] states and [N,2] widths")
        self.mass = _balanced_mass(np.asarray(self.values["group_id"]), expected_groups)
        self.receipt_path = receipt_path
        self.receipt = receipt
        self._device_values: dict[str, Tensor] = {}

    @property
    def population(self) -> int:
        return len(self.values["target_phase"])

    def make_resident(self, device: torch.device) -> int:
        """Copy numeric arrays once; return resident bytes for provenance."""
        self._device_values = {
            name: torch.from_numpy(np.asarray(value)).to(device)
            for name, value in self.values.items()
            if np.asarray(value).dtype.kind in {"b", "f", "i", "u"}
        }
        self._device_values["mass"] = torch.from_numpy(self.mass).to(device)
        return sum(value.numel() * value.element_size() for value in self._device_values.values())

    def tensor(self, name: str, ids: Tensor, device: torch.device) -> Tensor:
        if self._device_values:
            return self._device_values[name].index_select(0, ids.to(device))
        source = self.mass if name == "mass" else self.values[name]
        return torch.as_tensor(source[ids.cpu().numpy()]).to(device)


def _model(stage: StageRecipe) -> nn.Module:
    if stage.kind == "pair_head":
        return CachedPairDirectPhase(dropout=0.05)
    if stage.kind == "temporal_head":
        return TemporalRefiner(
            TemporalConfig(feature_count=17, hidden=160, backbone="gru", output_mode="residual")
        )
    if stage.kind == "phase_only_head":
        return PhaseOnlyFinal()
    raise ValueError(f"Unsupported cached stage kind: {stage.kind}")


def learning_rate(stage: StageRecipe, update: int) -> float:
    """Scale the historical 100/2500 upper-stage schedule to full TRAIN exposure."""
    if stage.kind == "pair_head":
        return 3e-4
    warmup = math.ceil(100 * stage.updates / 2_500)
    if update <= warmup:
        return 3e-4 * update / warmup
    progress = (update - warmup) / (stage.updates - warmup)
    return 3e-5 + (3e-4 - 3e-5) * (1 + math.cos(math.pi * progress)) / 2


def _loss(
    model: nn.Module, stage: StageRecipe, cache: FinalCache, ids: Tensor, device: torch.device
) -> Tensor:
    truth = cache.tensor("target_phase", ids, device).float()
    if stage.kind == "pair_head":
        prediction = model(PairFeatureBatch(cache.tensor("features", ids, device).float()))
        return functional.smooth_l1_loss(prediction, truth)
    mass = cache.tensor("mass", ids, device).float()
    if stage.kind == "temporal_head":
        experts = cache.tensor("expert_phase", ids, device).float()
        output = model(
            cache.tensor("features", ids, device).float(),
            cache.tensor("timing", ids, device).float(),
            cache.tensor("valid", ids, device).bool(),
            experts,
        )
        return training_loss(output, truth, experts, mass, cache.population)
    output = model(
        cache.tensor("features", ids, device).float(),
        cache.tensor("rgb_width_logits", ids, device).float(),
    )
    return fusion_loss(output, truth, mass, cache.population)


def _parent_identity(
    paths: Mapping[str, Path], stage: StageRecipe
) -> tuple[dict[str, str], dict[str, str]]:
    if set(paths) != set(stage.depends_on):
        raise ValueError(
            f"{stage.stage_id} parent receipts differ: "
            f"expected {stage.depends_on}, got {tuple(paths)}"
        )
    receipt_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        receipt = read_json_shared(path.resolve(strict=True))
        admitted_reuse = (
            receipt.get("schema") == "final_refit_external_parent_admission_v1"
            and receipt.get("status") == "PASS"
            and receipt.get("reuse_authorized") is True
            and receipt.get("fit_id") == name
        )
        completed_fit = (
            receipt.get("schema") == "rgb_port_checkpoint_receipt_v1"
            and receipt.get("status") == "COMPLETE"
            and receipt.get("scientific_endpoint") is True
            and receipt.get("fit_id") == name
        )
        if not (admitted_reuse or completed_fit):
            raise ValueError(f"Parent {name} schema, fit identity, or endpoint status differs")
        checkpoint = Path(str(receipt["checkpoint_path"])).resolve(strict=True)
        if receipt.get("checkpoint_sha256") != sha256(checkpoint):
            raise ValueError(f"Parent {name} checkpoint changed")
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        embedded_identity = (
            checkpoint_payload.get("identity") if isinstance(checkpoint_payload, dict) else None
        )
        if isinstance(embedded_identity, dict) and embedded_identity.get("technical") is True:
            raise ValueError(f"Parent {name} is a technical checkpoint")
        if admitted_reuse:
            identity = dict(receipt)
            claimed_identity = identity.pop("checkpoint_identity_sha256", None)
            original_receipt = Path(str(receipt.get("original_receipt_path", ""))).resolve(
                strict=True
            )
            if claimed_identity != canonical_sha256(identity) or receipt.get(
                "original_receipt_sha256"
            ) != sha256(original_receipt):
                raise ValueError(f"External parent {name} canonical identity changed")
        else:
            claimed_identity = str(receipt.get("identity_sha256", ""))
            payload = checkpoint_payload
            checkpoint_identity = embedded_identity
            identity_payload = (
                {
                    key: value
                    for key, value in checkpoint_identity.items()
                    if key != "identity_sha256"
                }
                if isinstance(checkpoint_identity, dict)
                else None
            )
            if (
                not isinstance(checkpoint_identity, dict)
                or checkpoint_identity.get("fit_id") != name
                or checkpoint_identity.get("identity_sha256") != claimed_identity
                or payload.get("identity_sha256") != claimed_identity
                or canonical_sha256(identity_payload) != claimed_identity
            ):
                raise ValueError(f"Parent {name} checkpoint identity is not canonical")
        receipt_hashes[name] = sha256(path)
        checkpoint_hashes[name] = str(receipt["checkpoint_sha256"])
    return receipt_hashes, checkpoint_hashes


def _technical_execution_limit(
    *, stage_updates: int, completed_updates: int, requested_updates: int
) -> int:
    """Return a bounded technical stop strictly below the scientific endpoint."""
    if requested_updates < 1 or requested_updates > TECHNICAL_CALL_UPDATE_CAP:
        raise ValueError("Technical benchmark requires between one and five updates")
    if stage_updates <= 1 or completed_updates < 0:
        raise ValueError("Technical benchmark cannot be isolated from this endpoint")
    lifetime_limit = min(TECHNICAL_LIFETIME_UPDATE_CAP, stage_updates - 1)
    if completed_updates > lifetime_limit:
        raise ValueError("Technical checkpoint exceeded its isolated lifetime cap")
    return min(completed_updates + requested_updates, lifetime_limit)


def _validate_technical_receipt(receipt: Mapping[str, Any], stage_updates: int) -> None:
    """Reject a technical receipt that claims or reaches a scientific endpoint."""
    completed = int(receipt.get("completed_updates", -1))
    if (
        receipt.get("status") == "COMPLETE"
        or receipt.get("scientific_endpoint") is not False
        or completed < 0
        or completed >= stage_updates
        or completed > min(TECHNICAL_LIFETIME_UPDATE_CAP, stage_updates - 1)
    ):
        raise RuntimeError("Technical cached training attempted to create a scientific endpoint")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError("Git did not return a full commit identity")
    return commit


def fit_cached_stage(
    *,
    recipe: FinalRefitRecipe,
    recipe_path: Path,
    stage: StageRecipe,
    cache_path: Path,
    parent_receipts: Mapping[str, Path],
    run_root: Path,
    device_name: str = "cuda",
    max_updates_this_call: int | None = None,
    resident_cache: bool = True,
    technical: bool = False,
    cache_scope: CachedTrainingScope | None = None,
    execution_authority: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Train or resume one frozen cached stage without selecting a checkpoint."""
    if stage.kind not in CACHE_KEYS or device_name not in {"cpu", "cuda"}:
        raise ValueError("Unsupported cached stage or device")
    verify_bound_sources(recipe)
    if technical and max_updates_this_call is None:
        raise ValueError("Technical benchmark requires a positive bounded update count")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache = FinalCache(cache_path.resolve(strict=True), stage, recipe, cache_scope)
    parents, parent_checkpoints = _parent_identity(parent_receipts, stage)
    cache_parents = cache.receipt.get("parents")
    if not isinstance(cache_parents, dict):
        raise ValueError("Cache receipt lacks parent checkpoint ancestry")
    mismatched = {
        name: (cache_parents.get(name), checkpoint_sha)
        for name, checkpoint_sha in parent_checkpoints.items()
        if cache_parents.get(name) != checkpoint_sha
    }
    if mismatched:
        raise ValueError(f"Cache parent checkpoints differ from supplied receipts: {mismatched}")
    for name in ("E_A5", "E_C2F"):
        expected_external = (
            recipe.external_parent_sha256
            if cache_scope is None
            else cache_scope.external_parent_sha256
        )
        if name in parent_checkpoints and expected_external.get(name) != (parent_checkpoints[name]):
            raise ValueError(f"External event parent differs from frozen recipe: {name}")
    output = run_root / ("technical" if technical else "fits") / stage.stage_id
    pair_schedule: Sequence[np.ndarray[Any, Any]] | None = None
    schedule_sha: str | None = None
    if stage.kind == "pair_head":
        pair_schedule, schedule_sha = deterministic_sequence_grouped_schedule(
            list(np.asarray(cache.values["sequence_id"]).astype(str)),
            list(np.asarray(cache.values["sample_token"]).astype(str)),
            seed=recipe.seed,
            batch_size=stage.batch_size,
            updates=stage.updates,
        )
    identity: dict[str, Any] = {
        "schema": "final_refit_cached_stage_identity_v1",
        "fit_id": stage.stage_id,
        "stage": stage.stage_id,
        "kind": stage.kind,
        "recipe_sha256": recipe.recipe_sha256,
        "recipe_file_sha256": sha256(recipe_path.resolve(strict=True)),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "cache_sha256": sha256(cache_path),
        "training_cache_sha256": sha256(cache_path),
        "training_sample_token_sha256": canonical_sha256(
            np.asarray(cache.values["sample_token"]).astype(str).tolist()
        ),
        "cache_receipt_sha256": sha256(cache.receipt_path),
        "parent_receipt_sha256": parents,
        "parent_checkpoint_sha256": parent_checkpoints,
        "seed": recipe.seed,
        "device": device_name,
        "resident_cache": resident_cache,
        "technical": technical,
        "cache_scope": None if cache_scope is None else cache_scope.__dict__,
        "batch_schedule_sha256": schedule_sha,
    }
    if execution_authority is not None:
        identity["execution_authority"] = dict(execution_authority)
        from .scientific_authority import verify_execution_binding

        verify_execution_binding(identity)
    identity["identity_sha256"] = canonical_sha256(identity)
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(recipe.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    resident_bytes = cache.make_resident(device) if resident_cache else 0
    model = _model(stage).float().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=stage.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator(device="cpu").manual_seed(recipe.seed)
    endpoint = stage.updates
    state = ProducerCheckpoint(output, identity, endpoint)
    cursor: dict[str, Any] = {"losses": []}
    if state.pointer.is_file():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    execution_endpoint = endpoint
    if technical:
        if max_updates_this_call is None:  # guarded before allocating the training state
            raise AssertionError("Technical update bound disappeared")
        execution_endpoint = _technical_execution_limit(
            stage_updates=stage.updates,
            completed_updates=state.completed,
            requested_updates=max_updates_this_call,
        )
    if schedule_sha is not None:
        atomic_write_json(
            output / "BATCH_SCHEDULE.json",
            {"stage": stage.stage_id, "batch_schedule_sha256": schedule_sha},
        )
    provenance_path = output / "RUN_PROVENANCE.json"
    provenance = {
        "schema": "final_refit_run_provenance_v1",
        "experiment_id": (
            "FINAL_REFIT_TRAIN40" if cache_scope is None else cache_scope.experiment_id
        ),
        "run_name": stage.stage_id,
        "git_commit": _git_commit(),
        "config_hash": recipe.recipe_sha256 if cache_scope is None else cache_scope.scope_sha256,
        "seed": recipe.seed,
        "dataset_manifest_hash": cache.receipt.get("dataset_manifest_sha256")
        or canonical_sha256(recipe.source_sha256),
        "split_version": (
            "TRAIN40_ALL_NO_INTERNAL_EVALUATION"
            if cache_scope is None
            else cache_scope.split_version
        ),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "start_time": datetime.now(UTC).isoformat(),
        "end_time": None,
        "status": "RUNNING",
        "checkpoint_path": None,
        "metrics_path": None,
        "identity_sha256": identity["identity_sha256"],
        "resident_cache_bytes": resident_bytes,
        "technical": technical,
    }
    if provenance_path.is_file():
        prior = read_json_shared(provenance_path)
        if prior.get("identity_sha256") != identity["identity_sha256"]:
            raise ValueError("Existing run provenance belongs to another identity")
        provenance = prior
    else:
        atomic_write_json(provenance_path, provenance)
    started = time.perf_counter()
    initial = state.completed
    while state.completed < execution_endpoint:
        ids = (
            torch.as_tensor(pair_schedule[state.completed], dtype=torch.long)
            if pair_schedule is not None
            else torch.randint(cache.population, (stage.batch_size,), generator=generator)
        )
        optimizer.param_groups[0]["lr"] = learning_rate(stage, state.completed + 1)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, stage, cache, ids, device)
        if not torch.isfinite(loss):
            raise FloatingPointError("Final-refit loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        state.begin()
        optimizer.step()
        scheduler.step()
        state.commit()
        cursor["losses"].append(float(loss.detach().cpu()))
        requested_pause = (
            max_updates_this_call is not None and state.completed - initial >= max_updates_this_call
        )
        if (
            state.completed % 100 == 0
            or state.completed == endpoint
            or state.completed == execution_endpoint
            or requested_pause
        ):
            status = "COMPLETE" if state.completed == endpoint else "IN_PROGRESS"
            state.save(model, optimizer, scheduler, generator, cursor, status=status)
        if requested_pause:
            break
    receipt = read_json_shared(state.receipt)
    if technical:
        _validate_technical_receipt(receipt, stage.updates)
    elapsed = time.perf_counter() - started
    updates_this_call = state.completed - initial
    provenance.update(
        status=receipt["status"],
        end_time=datetime.now(UTC).isoformat() if receipt["status"] == "COMPLETE" else None,
        checkpoint_path=receipt["checkpoint_path"],
        elapsed_seconds_last_call=elapsed,
        updates_last_call=updates_this_call,
        updates_per_second_last_call=updates_this_call / elapsed if elapsed else None,
    )
    atomic_write_json(provenance_path, provenance)
    return receipt


def load_cached_endpoint(
    fit_directory: Path, stage: StageRecipe, device: torch.device
) -> nn.Module:
    """Load a complete receipt-bound final cached endpoint."""
    receipt = read_json_shared(fit_directory / "CHECKPOINT_RECEIPT.json")
    checkpoint = Path(str(receipt["checkpoint_path"])).resolve(strict=True)
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("completed_updates") != stage.updates
        or receipt.get("checkpoint_sha256") != sha256(checkpoint)
    ):
        raise ValueError("Final cached endpoint is incomplete or changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = _model(stage)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).float().eval().requires_grad_(False)


__all__ = [
    "CachedTrainingScope",
    "FinalCache",
    "PhaseOnlyFinal",
    "fit_cached_stage",
    "learning_rate",
    "load_cached_endpoint",
]
