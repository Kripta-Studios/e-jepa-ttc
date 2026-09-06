"""Synchronous registered phase queue; no scoring and no raw expert replay.

The CLI must supply actual freeze/role/producer/replay validation and a validated
cached-source loader. This module deliberately has no permissive default for
that prerequisite check. It is not a replacement for production data validation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json
from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .arms import resolve_arm
from .endpoint import load_endpoint
from .lifecycle import ExclusiveLease
from .phase_manifest import EndpointReceipt, fit_key, seal_phase
from .registry import FitSpec, registered_graph
from .training import QuerySource, fit, load_checkpoint
from .work_budget import EngineWorkJournal, WorkBudget


def run_phase(
    output: Path,
    *,
    stage: str,
    availability: dict[str, bool],
    freeze_sha256: str,
    train_source_hashes: dict[str, str],
    technical_reserved: int,
    validate_prerequisites: Callable[[], None],
    source_loader: Callable[[FitSpec], QuerySource],
    resource_ok: Callable[[], bool],
    resume: bool,
) -> dict[str, Any]:
    """Run every fit in one authorized phase or persist a synchronous resource pause.

    output is the campaign execution root, shared across phases. D/density pools
    and technical reservation must be stable across resumes.
    Scientific gate validation and stage order belong to validate_prerequisites;
    callbacks must raise on unresolved evidence, not silently return defaults.
    """
    if any(type(value) is not bool for value in availability.values()):
        raise ValueError("availability must be resolved")
    graph = registered_graph(**availability)
    selected = [spec for spec in graph if spec.stage == stage]
    if not selected or set(train_source_hashes) != {fit_key(spec) for spec in selected}:
        raise ValueError("phase source identities do not match the registered graph")
    for digest in [freeze_sha256, *train_source_hashes.values()]:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid frozen identity")
    validate_prerequisites()  # Before creating output files, loading inputs or allocating heads.
    state_path, endpoint_path = output / f"{stage}_STATE.json", output / f"{stage}_ENDPOINTS.json"
    contract = {
        "freeze": freeze_sha256,
        "stage": stage,
        "availability": availability,
        "train_source_hashes": train_source_hashes,
        "technical_reserved": technical_reserved,
    }
    with ExclusiveLease(output / "HEAD_WRITER.lock"):
        if endpoint_path.exists():
            raise FileExistsError("phase is already sealed; evaluate its pinned endpoints")
        if state_path.exists():
            if not resume:
                raise FileExistsError("existing phase requires explicit resume")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["contract"] != contract:
                raise ValueError("phase resume contract changed")
        else:
            state = {"contract": contract, "fits": {}, "status": "PREPARED"}
        # Reserve possible registered branches, without authorizing them. Their
        # practical gates may enable execution later without a new budget graph.
        possible = registered_graph(
            d1=availability["d1"],
            density=availability["density"],
            t3=True,
            latent=True,
            replicate_scalar=True,
            replicate_latent=True,
        )
        budget = WorkBudget(
            output / "PHYSICAL_WORK.json",
            {fit_key(s): s.updates for s in possible},
            technical_reserved,
        )
        receipts = {}
        for spec in selected:
            key = fit_key(spec)
            if not resource_ok():
                state["status"] = "PAUSED_RESOURCE"
                atomic_json(state_path, state)
                return state
            binding = resolve_arm(spec, graph)
            source = source_loader(spec)
            if source.identity_sha256 != train_source_hashes[key]:
                raise ValueError("loader differs from frozen TRAIN source")
            folder = output / "fits" / key
            checkpoint = folder / "checkpoint_last.pt"
            if checkpoint.exists() and not resume:
                raise FileExistsError("existing fit requires explicit resume")
            prior = load_checkpoint(checkpoint) if checkpoint.exists() else None
            if prior is not None and prior["status"] == "COMPLETED":
                model = load_endpoint(
                    checkpoint,
                    binding.model,
                    seed=spec.seed,
                    freeze_sha256=freeze_sha256,
                    train_source_sha256=source.identity_sha256,
                    endpoint_sha256=sha256(checkpoint),
                )
                del model
                EngineWorkJournal(budget, key).start(2500, checkpoint)
                result = {
                    "status": "COMPLETED",
                    "completed_updates": 2500,
                    "scientific_endpoint": True,
                }
            else:
                result = fit(
                    source,
                    binding.model,
                    folder,
                    seed=spec.seed,
                    freeze_sha256=freeze_sha256,
                    resource_ok=resource_ok,
                    resume=prior is not None,
                    journal=EngineWorkJournal(budget, key),
                )
            del source
            state["fits"][key] = result
            state["status"] = result["status"] if result["status"] != "COMPLETED" else "RUNNING"
            atomic_json(state_path, state)
            if result["status"] != "COMPLETED":
                if result["status"] != "PAUSED_RESOURCE":
                    raise ValueError("scientific queue cannot accept a technical partial")
                return state
            receipts[key] = EndpointReceipt(
                checkpoint, sha256(checkpoint), train_source_hashes[key]
            )
        state["endpoints"] = seal_phase(
            endpoint_path,
            checkpoint_root=output / "fits",
            stage=stage,
            availability=availability,
            freeze_sha256=freeze_sha256,
            receipts=receipts,
        )
        state["status"] = "ENDPOINTS_SEALED_NOT_EVALUATED"
        atomic_json(state_path, state)
        return state
