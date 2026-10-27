"""Recoverable technical adapter for producer-fold source and recipe parity.

Scientific fold fitting remains unavailable until the transitive OOF lineage,
generation and deployment-refit contracts are admitted.  The bounded technical
path exercises the historical model, objective, sampler and checkpoint machinery
without creating a scientific endpoint.
"""

from __future__ import annotations

import platform
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import Tensor

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.training.causal_scale_eap import _autocast, _foreground_only_loss_config
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port.recipe import ProducerRecipe, file_sha256, resolved_recipe
from operational.rgb_port.train_producers import (
    ProducerCheckpoint,
    ProducerSource,
    _epoch_order,
    _producer_loss,
    _validate_batch,
    build_producer,
)
from operational.rgb_port_revision.loss import EffectiveBatch

from .fold_sources import load_fold_event_source, load_fold_rgb_source
from .recipe import canonical_sha256

TECHNICAL_UPDATE_CAP = 10
FOLD_FITS = ("E_A5_MATCHED", "E_C2F_MATCHED", "R_A5", "R_C2F")


class FoldProducerSource(Protocol):
    population_size: int
    frame_counts: Sequence[int]
    input_span_us: Sequence[int]
    sampling_groups: Sequence[Sequence[int]]
    identity: Mapping[str, Any]

    def batch(
        self, indices: Sequence[int], modality: str
    ) -> ObjectEventV4Batch: ...

    def close(self) -> None: ...


def build_fold_recipe(
    source: FoldProducerSource,
    fit_id: str,
    config_path: Path,
    *,
    microbatch_size: int = 32,
) -> ProducerRecipe:
    """Resolve the unchanged 18-epoch seed-7 recipe for exact fold membership."""
    if fit_id not in FOLD_FITS:
        raise ValueError("Unknown fold producer fit")
    identity = dict(source.identity)
    if (
        identity.get("schema") != "final_refit_fold_source_v1"
        or identity.get("role") != "producer_pair"
        or int(identity.get("population_size", -1)) != source.population_size
    ):
        raise ValueError("Producer source is not an admitted fold source")
    recipe = resolved_recipe(
        config_path.resolve(strict=True),
        fit_id=fit_id,
        producer_population=source.population_size,
        role_manifest_sha256=str(identity["inventory_sha256"]),
        microbatch_size=microbatch_size,
    )
    if recipe.seed != 7 or recipe.epochs != 18:
        raise ValueError("Fold producer seed or exposure differs")
    teacher = {
        **recipe.teacher_policy,
        "roles": ["producer_pair"],
        "historical_roles_preserved": ["P", "H", "V"],
        "lookup": "original_native_source_ordinal_plus_sample_token",
    }
    return replace(recipe, teacher_policy=teacher)


def _effective_batch(
    source: FoldProducerSource, indices: Sequence[int], recipe: ProducerRecipe
) -> EffectiveBatch:
    groups: dict[int, list[int]] = {}
    for index in indices:
        groups.setdefault(int(source.frame_counts[int(index)]), []).append(int(index))
    parts = [
        source.batch(group[start : start + recipe.microbatch_size], recipe.modality)
        for _, group in sorted(groups.items())
        for start in range(0, len(group), recipe.microbatch_size)
    ]
    if not parts:
        raise ValueError("Fold effective batch is empty")
    return EffectiveBatch(parts)


def _identity(
    source: FoldProducerSource,
    recipe: ProducerRecipe,
    config_path: Path,
    device: str,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    sources = {
        "operational/final_refit/fold_producers.py": Path(__file__).resolve(),
        "operational/final_refit/fold_sources.py": repository
        / "operational/final_refit/fold_sources.py",
        "operational/rgb_port/train_producers.py": repository
        / "operational/rgb_port/train_producers.py",
        "operational/rgb_port_revision/loss.py": repository
        / "operational/rgb_port_revision/loss.py",
        "configs/rgb_port/producers.json": config_path.resolve(strict=True),
    }
    payload: dict[str, Any] = {
        **recipe.identity,
        "schema": "final_refit_fold_producer_technical_identity_v1",
        "fit_id": recipe.fit_id,
        "rotation": source.identity["rotation"],
        "population_role": "producer_pair",
        "source_identity": dict(source.identity),
        "recipe_identity_sha256": recipe.identity_sha256,
        "initialization": "fresh_seed7_no_final_or_historical_checkpoint_weights",
        "technical": True,
        "scientific_endpoint_authorized": False,
        "technical_update_cap": TECHNICAL_UPDATE_CAP,
        "device": device,
        "environment": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
        "executed_source_sha256": {
            name: file_sha256(path) for name, path in sorted(sources.items())
        },
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def fit_fold_producer_technical(
    *,
    source: FoldProducerSource,
    fit_id: str,
    config_path: Path,
    run_root: Path,
    device_name: str = "cpu",
    max_updates_this_call: int,
    microbatch_size: int = 32,
    scientific: bool = False,
) -> dict[str, Any]:
    """Run/resume at most ten lifetime technical updates; never form an endpoint."""
    if scientific:
        raise RuntimeError(
            "Scientific fold producers are fail-closed until transitive OOF lineage, "
            "generation coverage and deployment-refit ancestry are admitted"
        )
    if max_updates_this_call < 1 or max_updates_this_call > 5:
        raise ValueError("One technical call must request between one and five updates")
    recipe = build_fold_recipe(source, fit_id, config_path, microbatch_size=microbatch_size)
    target = torch.device(device_name)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(recipe.seed)
    torch.use_deterministic_algorithms(True)
    model = build_producer(recipe).to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe.training_config["learning_rate"]),
        weight_decay=float(recipe.training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=recipe.epochs,
        eta_min=float(recipe.training_config["minimum_learning_rate"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(recipe.seed)
    fit_directory = (
        run_root.resolve()
        / "technical"
        / f"rotation_{int(source.identity['rotation'])}"
        / fit_id
    )
    identity = _identity(source, recipe, config_path, device_name)
    state = ProducerCheckpoint(fit_directory, identity, recipe.updates)
    cursor: dict[str, Any] = {
        "epoch": 1,
        "position": 0,
        "order": _epoch_order(cast(ProducerSource, source), generator),
        "losses": [],
    }
    if state.pointer.is_file():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    if state.completed >= TECHNICAL_UPDATE_CAP:
        return read_json_shared(state.receipt)
    allowed_updates = min(
        max_updates_this_call,
        TECHNICAL_UPDATE_CAP - state.completed,
    )
    loss_config = CausalScaleTTCLossConfig(**recipe.loss_config)
    warmup = _foreground_only_loss_config(loss_config)
    model.train()
    performed = 0
    while performed < allowed_updates:
        epoch = int(cursor["epoch"])
        start = int(cursor["position"])
        stop = min(start + recipe.effective_batch_size, source.population_size)
        indices = cast(Tensor, cursor["order"])[start:stop].tolist()
        batch = _effective_batch(source, indices, recipe).to(target)
        for part in batch.parts:
            _validate_batch(part, recipe)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(target, str(recipe.training_config["precision"])):
            loss, components = _producer_loss(
                model, batch, warmup if epoch <= 3 else loss_config
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Fold producer technical loss became nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(recipe.training_config["grad_clip_norm"]),
            error_if_nonfinite=True,
        )
        state.begin()
        optimizer.step()
        state.commit()
        performed += 1
        cursor["position"] = stop
        cursor["losses"].append(
            {
                "update": state.completed,
                "total": float(loss.detach().float().cpu()),
                "components": {
                    name: float(value.detach().float().cpu())
                    for name, value in components.items()
                },
            }
        )
        if stop == source.population_size:
            scheduler.step()
            cursor.update(
                epoch=epoch + 1,
                position=0,
                order=_epoch_order(cast(ProducerSource, source), generator),
            )
    state.save(model, optimizer, scheduler, generator, cursor, status="IN_PROGRESS")
    provenance = {
        "schema": "final_refit_fold_producer_technical_provenance_v1",
        "status": "IN_PROGRESS",
        "scientific_updates": 0,
        "technical_optimizer_updates": state.completed,
        "technical_update_cap": TECHNICAL_UPDATE_CAP,
        "scientific_endpoint": False,
        "fit_id": fit_id,
        "rotation": source.identity["rotation"],
        "source_identity_sha256": source.identity["identity_sha256"],
        "identity_sha256": identity["identity_sha256"],
        "checkpoint_receipt_sha256": file_sha256(state.receipt),
        "updated_at": datetime.now(UTC).isoformat(),
        "limitations": (
            "Technical recoverability only; no OOF generation, stitching, final refit, "
            "evaluation or deployment ancestry is admitted"
        ),
    }
    atomic_write_json(fit_directory / "TECHNICAL_PROVENANCE.json", provenance)
    return read_json_shared(state.receipt)


def run_fold_producer_technical(
    *,
    modality: str,
    fit_id: str,
    fold_contract: Path,
    rotation: int,
    run_root: Path,
    config_path: Path = Path("configs/rgb_port/producers.json"),
    source_descriptor: Path | None = None,
    teacher_manifest: Path | None = None,
    event_source_root: Path | None = None,
    device_name: str = "cpu",
    max_updates_this_call: int = 1,
) -> dict[str, Any]:
    """Load one fold source, run bounded recoverable technical work, and close storage."""
    if modality == "rgb":
        if source_descriptor is None or teacher_manifest is None:
            raise ValueError("RGB fold producer requires descriptor and teacher manifest")
        source: FoldProducerSource = load_fold_rgb_source(
            source_descriptor, teacher_manifest, fold_contract, rotation
        )
    elif modality == "event":
        if event_source_root is None:
            raise ValueError("Event fold producer requires audited TRAIN40 storage root")
        source = load_fold_event_source(fold_contract, rotation, event_source_root)
    else:
        raise ValueError("modality must be rgb or event")
    expected_modality = "event" if fit_id.startswith("E_") else "rgb"
    if modality != expected_modality:
        source.close()
        raise ValueError("Fit identity and fold source modality differ")
    try:
        return fit_fold_producer_technical(
            source=source,
            fit_id=fit_id,
            config_path=config_path,
            run_root=run_root,
            device_name=device_name,
            max_updates_this_call=max_updates_this_call,
        )
    finally:
        source.close()


__all__ = [
    "FOLD_FITS",
    "TECHNICAL_UPDATE_CAP",
    "build_fold_recipe",
    "fit_fold_producer_technical",
    "run_fold_producer_technical",
]
