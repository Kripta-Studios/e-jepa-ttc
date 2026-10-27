"""Scientific, fixed-endpoint OOF producers with exact group roles and durable resume.

This is a separate runner: bounded technical checkpoints are never promoted.
Generation and downstream consumers must verify the resulting endpoint lineage.
"""

from __future__ import annotations

import argparse
import os
import platform
import random
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from e_jepa_ttc.training.causal_scale_eap import _autocast
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port.recipe import ProducerRecipe, file_sha256
from operational.rgb_port.train_producers import (
    ProducerCheckpoint,
    ProducerSource,
    _epoch_order,
    _validate_batch,
    build_producer,
)

from .fold_producers import FoldProducerSource, _effective_batch, build_fold_recipe
from .fold_sources import load_fold_event_source, load_fold_rgb_source
from .oof_lineage import build_endpoint_lineage, load_rotation_identity, write_endpoint_lineage
from .producers import _final_memory_allowed, configure_rgb_runtime, producer_objective
from .recipe import canonical_sha256
from .scientific_authority import execution_binding, load_config
from .shared_resources import SharedResourceMonitor

FIT_NAMES = {"E_A5": "E_A5_MATCHED", "E_C2F": "E_C2F_MATCHED", "R_A5": "R_A5", "R_C2F": "R_C2F"}


class ResourceGuard(Protocol):
    def guard(
        self, run_root: Path, *, checkpoint_reservation: int, completed_updates: int
    ) -> tuple[bool, dict[str, Any]]: ...


def scientific_recipe(source: FoldProducerSource, fit: str, config: Path) -> ProducerRecipe:
    """Preserve seed, architecture, objective and exposure; use deterministic transport."""
    recipe = build_fold_recipe(source, FIT_NAMES[fit], config)
    return replace(
        recipe,
        model_config={**recipe.model_config, "deterministic_transport_sampling": True},
        training_config={
            **recipe.training_config,
            "deterministic_transport_sampling": "ENABLED_OOF_PROSPECTIVE",
        },
    )


def source_bindings(repository: Path) -> dict[str, str]:
    """Bind the training dependency closure, not mutable downstream queue code."""
    names = [
        "operational/final_refit/scientific_fold_producers.py",
        "operational/final_refit/fold_producers.py",
        "operational/final_refit/fold_sources.py",
        "operational/final_refit/oof_lineage.py",
    ]
    current = read_json_shared(
        repository / "artifacts/final_refit_20261018_runtime_v6/PRODUCER_PLAN.json"
    )
    names.extend(current["executed_source_sha256"])
    for name, digest in current["executed_source_sha256"].items():
        if file_sha256(repository / name) != digest:
            raise ValueError(f"Admitted producer source changed: {name}")
    return {name: file_sha256(repository / name) for name in sorted(set(names))}


def identity_for(
    source: FoldProducerSource,
    recipe: ProducerRecipe,
    *,
    fit: str,
    contract: Path,
    bindings: dict[str, str],
    device: str,
) -> dict[str, Any]:
    """Embed exact training membership and execution identity in every checkpoint."""
    rotation = load_rotation_identity(contract, int(source.identity["rotation"]))
    role = rotation.roles["producer"]
    if (
        source.identity.get("schema") != "final_refit_fold_source_v1"
        or source.identity.get("role") != "producer_pair"
        or source.identity.get("modality") != ("event" if fit.startswith("E_") else "rgb")
        or source.identity.get("ordered_native_ordinal_sha256")
        != role.ordered_native_ordinal_sha256
        or sorted(source.identity["group_ids"]) != list(role.group_ids)
        or source.population_size != role.population
        or source.identity["ordered_sample_token_sha256"] != role.ordered_sample_token_sha256
    ):
        raise ValueError("Producer source escaped the bound fold training role")
    payload: dict[str, Any] = {
        "schema": "final_refit_fold_producer_identity_v1",
        "fit_id": fit,
        "rotation": rotation.rotation,
        "technical": False,
        "source_identity": dict(source.identity),
        "producer_recipe": recipe.identity,
        "recipe_identity_sha256": recipe.identity_sha256,
        "fold_contract_sha256": rotation.fold_contract_sha256,
        "rotation_manifest_sha256": role.manifest_sha256,
        "training_cache_sha256": role.rows_sha256,
        "training_sample_token_sha256": role.ordered_sample_token_sha256,
        "train_group_ids": list(role.group_ids),
        "generation_group_ids": list(rotation.roles["generation"].group_ids),
        "initialization": "fresh_seed7_no_checkpoint_import",
        "endpoint_updates": recipe.updates,
        "checkpoint_interval_updates": 100,
        "executed_source_sha256": bindings,
        "device": device,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "host": platform.node(),
        },
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def verify_endpoint(directory: Path, identity: dict[str, Any], updates: int) -> dict[str, Any]:
    """Reject partial, technical, corrupted or wrong-role producer endpoints."""
    receipt = read_json_shared(directory / "CHECKPOINT_RECEIPT.json")
    journal = read_json_shared(directory / "UPDATE_JOURNAL.json")
    checkpoint = Path(receipt["checkpoint_path"]).resolve(strict=True)
    if not (
        receipt.get("status") == "COMPLETE"
        and receipt.get("complete_state") is True
        and receipt.get("scientific_endpoint") is True
        and receipt.get("completed_updates")
        == journal.get("completed_updates")
        == journal.get("durable_updates")
        == updates
        and journal.get("pending_update_upper") == 0
        and receipt.get("identity_sha256") == identity["identity_sha256"]
        and checkpoint.is_relative_to(directory.resolve())
        and file_sha256(checkpoint) == receipt.get("checkpoint_sha256")
    ):
        raise ValueError("Scientific OOF producer endpoint is incomplete or inconsistent")
    return receipt


def write_admission(
    directory: Path, identity: dict[str, Any], contract: Path, updates: int
) -> dict[str, Any]:
    """Seal structural ancestry plus a separately verified scientific endpoint admission."""
    receipt = verify_endpoint(directory, identity, updates)
    rotation = load_rotation_identity(contract, int(identity["rotation"]))
    role = rotation.roles["producer"]
    generation = rotation.roles["generation"]
    receipt_path = directory / "CHECKPOINT_RECEIPT.json"
    lineage = build_endpoint_lineage(
        fold_contract_path=contract,
        rotation=rotation.rotation,
        stage_id=identity["fit_id"],
        checkpoint_sha256=receipt["checkpoint_sha256"],
        checkpoint_receipt_sha256=file_sha256(receipt_path),
        training_cache_sha256=role.rows_sha256,
        checkpoint_path=Path(receipt["checkpoint_path"]),
        checkpoint_receipt_path=receipt_path,
        training_cache_path=Path(role.rows_path),
        parent_lineage_paths={},
    )
    write_endpoint_lineage(directory / "STRUCTURAL_LINEAGE.json", lineage)
    value = {
        "schema": "final_refit_fold_endpoint_admission_v1",
        "status": "PASS",
        "technical": False,
        "scientific_endpoint": True,
        "rotation": rotation.rotation,
        "trained_role": "producer_pair",
        "fit_id": identity["fit_id"],
        "fold_contract_sha256": rotation.fold_contract_sha256,
        "rotation_manifest_sha256": role.manifest_sha256,
        "train_rows_sha256": role.rows_sha256,
        "training_cache_sha256": role.rows_sha256,
        "ordered_sample_token_sha256": role.ordered_sample_token_sha256,
        "sample_token_set_sha256": role.sample_token_set_sha256,
        "ordered_native_ordinal_sha256": role.ordered_native_ordinal_sha256,
        "checkpoint_path": receipt["checkpoint_path"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "checkpoint_identity_sha256": receipt["identity_sha256"],
        "checkpoint_receipt_path": str(receipt_path.resolve()),
        "checkpoint_receipt_sha256": file_sha256(receipt_path),
        "train_group_ids": list(role.group_ids),
        "training_cache_role": "producer_pair",
        "generation_group_ids": list(generation.group_ids),
        "generation_token_set_sha256": generation.sample_token_set_sha256,
        "generation_rows_excluded_from_all_ancestors": True,
        "parent_checkpoint_sha256": {},
        "parent_receipt_sha256": {},
        "parent_admission_sha256": {},
        "updates": updates,
        "structural_lineage_sha256": file_sha256(directory / "STRUCTURAL_LINEAGE.json"),
    }
    value["admission_payload_sha256"] = canonical_sha256(value)
    atomic_write_json(directory / "ENDPOINT_ADMISSION.json", value)
    return value


def train(
    *,
    source: FoldProducerSource,
    fit: str,
    config: Path,
    contract: Path,
    run: Path,
    monitor: ResourceGuard,
    device: str = "cuda",
    max_updates: int | None = None,
    additional_stops: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Train or resume one scientific producer; pause only at optimizer boundaries."""
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    authority = execution_binding(run)
    registered = load_config(run)
    if (
        config.resolve() != Path(registered["config"]).resolve()
        or contract.resolve() != Path(registered["fold_contract"]).resolve()
    ):
        raise ValueError("Producer launch differs from its fixed scientific configuration")
    recipe = scientific_recipe(source, fit, config)
    repository = Path(__file__).resolve().parents[2]
    identity = identity_for(
        source,
        recipe,
        fit=fit,
        contract=contract,
        bindings=source_bindings(repository),
        device=device,
    )
    identity = {k: v for k, v in identity.items() if k != "identity_sha256"}
    identity["execution_authority"] = authority
    identity["identity_sha256"] = canonical_sha256(identity)
    directory = run / "rotations" / str(identity["rotation"]) / "fits" / fit
    configure_rgb_runtime()
    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.manual_seed_all(recipe.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
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
    generator = torch.Generator().manual_seed(recipe.seed)
    state = ProducerCheckpoint(directory, identity, recipe.updates)
    cursor: dict[str, Any] = {
        "epoch": 1,
        "position": 0,
        "order": _epoch_order(cast(ProducerSource, source), generator),
        "losses": [],
    }
    if state.pointer.exists():
        cursor = state.restore(model, optimizer, scheduler, generator)
    else:
        state.save(model, optimizer, scheduler, generator, cursor, status="READY")
    initial = state.completed
    started = time.monotonic()
    stop_files = (run / "STOP_REQUEST", directory / "STOP_REQUEST", *additional_stops)
    model.train()
    while state.completed < recipe.updates:
        allowed, resources = monitor.guard(
            run, checkpoint_reservation=1024**3, completed_updates=state.completed
        )
        manual = any(path.exists() for path in stop_files)
        if manual or not _final_memory_allowed(allowed, resources):
            state.save(model, optimizer, scheduler, generator, cursor, status="PAUSED_RESOURCE")
            atomic_write_json(
                directory / "RESOURCE_PAUSE.json",
                {**resources, "completed_updates": state.completed, "manual": manual},
            )
            break
        epoch, start = int(cursor["epoch"]), int(cursor["position"])
        stop = min(start + recipe.effective_batch_size, source.population_size)
        indices = cursor["order"][start:stop].tolist()
        if not indices or epoch > recipe.epochs:
            raise ValueError("OOF optimizer cursor escaped its fixed exposure")
        host_batch = _effective_batch(source, indices, recipe)
        for part in host_batch.parts:
            _validate_batch(part, recipe)
        batch = host_batch.to(target)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(target, str(recipe.training_config["precision"])):
            total, _ = producer_objective(model, batch, recipe, epoch)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Nonfinite OOF producer loss")
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(recipe.training_config["grad_clip_norm"]),
            error_if_nonfinite=True,
        )
        state.begin()
        optimizer.step()
        state.commit()
        cursor["position"] = stop
        cursor["losses"].append(float(total.detach().float().cpu()))
        if stop == source.population_size:
            scheduler.step()
            cursor.update(
                epoch=epoch + 1,
                position=0,
                order=_epoch_order(cast(ProducerSource, source), generator),
            )
        bounded_stop = max_updates is not None and state.completed - initial >= max_updates
        if state.completed % 100 == 0 or state.completed == recipe.updates or bounded_stop:
            state.save(
                model,
                optimizer,
                scheduler,
                generator,
                cursor,
                status="COMPLETE" if state.completed == recipe.updates else "IN_PROGRESS",
            )
        atomic_write_json(
            directory / "PROGRESS.json",
            {
                "updated_utc": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "rotation": identity["rotation"],
                "fit": fit,
                "completed_updates": state.completed,
                "endpoint_updates": recipe.updates,
                "updates_this_call": state.completed - initial,
                "seconds_this_call": time.monotonic() - started,
                "epoch": epoch,
            },
        )
        if bounded_stop:
            break
    receipt = read_json_shared(state.receipt)
    if receipt["status"] == "COMPLETE":
        write_admission(directory, identity, contract, recipe.updates)
    return receipt


def main() -> int:
    """Run one role-bound scientific producer in its own recoverable subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "fold-contract", "config", "rgb-source", "teacher", "event-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=range(5), required=True)
    parser.add_argument("--fit", choices=FIT_NAMES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-updates", type=int)
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    source = (
        load_fold_event_source(args.fold_contract, args.rotation, args.event_root)
        if args.fit.startswith("E_")
        else load_fold_rgb_source(args.rgb_source, args.teacher, args.fold_contract, args.rotation)
    )
    repo = Path(__file__).resolve().parents[2]
    monitor = SharedResourceMonitor(
        repo,
        session_path=args.run / "RESOURCE_SESSION.json",
        snapshot_path=args.run / "RESOURCE_SNAPSHOT.json",
        run_root=args.run,
    )
    try:
        receipt = train(
            source=source,
            fit=args.fit,
            config=args.config,
            contract=args.fold_contract,
            run=args.run,
            monitor=monitor,
            device=args.device,
            max_updates=args.max_updates,
        )
    finally:
        source.close()
    return 0 if receipt["status"] == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
