"""Resumable source/normalizer identity preparation; never an optimizer or freeze authority."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json

from .arms import resolve_arm
from .cache import CachedQueries
from .campaign_sources import CampaignSources
from .lifecycle import ExclusiveLease
from .phase_manifest import fit_key
from .registry import registered_graph


def prepare_source_identities(
    output: Path,
    *,
    sources: CampaignSources,
    availability: dict[str, bool],
    configuration_sha256: str,
    authority_sha256: str,
    validate_prerequisites: Callable[[], None],
    resource_ok: Callable[[], bool],
    resume: bool,
    validate_loaded_source: Callable[[CachedQueries], dict] | None = None,
    verify_only: bool = False,
) -> dict:
    """Persist complete TRAIN/OLD_DEV identity pairs at fit boundaries.

    The mandatory caller validator checks actual role/time/producer authority and
    configuration pins before loading data. Preparation does not enable stage
    gates. Resumption revalidates recorded identities rather than trusting stale
    normalizers. No checkpoints or optimizer updates are created.
    """
    for digest in (configuration_sha256, authority_sha256):
        if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ValueError("source preparation requires pinned configuration and authority")
    if set(availability) != {
        "d1",
        "density",
        "t3",
        "latent",
        "replicate_scalar",
        "replicate_latent",
    } or any(type(value) is not bool for value in availability.values()):
        raise ValueError("complete registered preparation graph flags required")
    graph = registered_graph(**availability)
    if sources.graph != graph:
        raise ValueError("source adapter differs from the complete registered preparation graph")
    contract = {
        "configuration_sha256": configuration_sha256,
        "authority_sha256": authority_sha256,
        "availability": availability,
        "fits": [
            {"fit": asdict(spec), "model": asdict(resolve_arm(spec, graph).model)} for spec in graph
        ],
    }
    if validate_loaded_source is not None:
        contract["loaded_source_qa"] = "simplex_t_full_population_gather_qa_v1"
    validate_prerequisites()
    state_path = output / "SOURCE_PREPARATION.json"
    if verify_only:
        try:
            if not state_path.exists():
                return {"status": "SOURCE_IDENTITIES_INCOMPLETE", "optimizer_updates": 0}

            def read() -> bytes:
                if not resource_ok():
                    raise InterruptedError("PAUSED_RESOURCE: source identity verification")
                with state_path.open("rb") as stream:
                    data = stream.read(8_388_609)
                if len(data) > 8_388_608:
                    raise ValueError("source preparation exceeds freeze metadata bound")
                return data

            payload = read()
            state = json.loads(payload)
            if (
                state.get("schema") != "simplex_t_source_preparation_v1"
                or state.get("contract") != contract
                or type(state.get("optimizer_updates")) is not int
                or state["optimizer_updates"] != 0
                or state.get("scientific_freeze") is not False
                or state.get("gates_enabled") is not False
            ):
                raise ValueError("source preparation verification contract differs")
            if state.get("status") in {"PREPARING", "PAUSED_RESOURCE"}:
                return {"status": "SOURCE_IDENTITIES_INCOMPLETE", "optimizer_updates": 0}
            expected = {fit_key(spec) for spec in graph}
            if (
                state.get("status") != "SOURCE_IDENTITIES_COMPLETE_NOT_SCIENTIFIC_FREEZE"
                or set(state.get("identities", {})) != expected
                or (
                    validate_loaded_source is not None
                    and set(state.get("source_qa", {})) != expected
                )
            ):
                raise ValueError("source preparation completion inventory differs")
            for spec in sorted(
                graph,
                key=lambda item: (
                    item.fold,
                    resolve_arm(item, graph).model.feature_count,
                    resolve_arm(item, graph).pool,
                    fit_key(item),
                ),
            ):
                validate_prerequisites()
                pair, checked_pair = {}, {}
                for role in ("inner_oof", "outer_dev"):
                    if not resource_ok():
                        raise InterruptedError("PAUSED_RESOURCE: source identity verification")
                    source = sources.source(spec, role)
                    pair[role] = source.identity_sha256
                    if validate_loaded_source is not None:
                        checked_pair[role] = validate_loaded_source(source)
                key = fit_key(spec)
                if state["identities"][key] != pair:
                    raise ValueError("verified source or TRAIN normalizer differs")
                if validate_loaded_source is not None and state["source_qa"][key] != checked_pair:
                    raise ValueError("verified loaded source QA differs")
            validate_prerequisites()
            if read() != payload:
                raise ValueError("source preparation changed during verification")
            return state
        finally:
            sources.release()
    with ExclusiveLease(output / "SOURCE_PREPARATION.lock"):
        if state_path.exists():
            if not resume:
                raise FileExistsError("existing source preparation requires explicit resume")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                state.get("schema") != "simplex_t_source_preparation_v1"
                or type(state.get("optimizer_updates")) is not int
                or state["optimizer_updates"] != 0
                or state.get("scientific_freeze") is not False
                or state.get("gates_enabled") is not False
                or not isinstance(state.get("identities"), dict)
            ):
                raise ValueError("invalid non-scientific source preparation state")
            if state["contract"] != contract:
                raise ValueError("source preparation resume contract changed")
        else:
            if resume:
                raise FileNotFoundError("no source preparation state to resume")
            state = {
                "schema": "simplex_t_source_preparation_v1",
                "contract": contract,
                "identities": {},
                "status": "PREPARING",
                "optimizer_updates": 0,
                "scientific_freeze": False,
                "gates_enabled": False,
            }
        if validate_loaded_source is not None:
            state.setdefault("source_qa", {})
        if set(state["identities"]) - {fit_key(spec) for spec in graph}:
            raise ValueError("unregistered identity in source preparation state")
        try:
            ordered = sorted(
                graph,
                key=lambda spec: (
                    spec.fold,
                    resolve_arm(spec, graph).model.feature_count,
                    resolve_arm(spec, graph).pool,
                    fit_key(spec),
                ),
            )
            for spec in ordered:
                validate_prerequisites()
                pair = {}
                checked_pair = {}
                for role in ("inner_oof", "outer_dev"):
                    if not resource_ok():
                        state["status"] = "PAUSED_RESOURCE"
                        atomic_json(state_path, state)
                        return state
                    source = sources.source(spec, role)
                    digest = source.identity_sha256
                    if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
                        raise ValueError("source loader returned an invalid identity")
                    pair[role] = digest
                    if validate_loaded_source is not None:
                        checked_pair[role] = validate_loaded_source(source)
                key = fit_key(spec)
                if key in state["identities"] and state["identities"][key] != pair:
                    raise ValueError("recorded source or TRAIN normalizer changed on resume")
                state["identities"][key] = pair
                if validate_loaded_source is not None:
                    if key in state["source_qa"] and state["source_qa"][key] != checked_pair:
                        raise ValueError("loaded source QA changed on resume")
                    state["source_qa"][key] = checked_pair
                state["status"] = "PREPARING"
                atomic_json(state_path, state)
            validate_prerequisites()
            state["status"] = "SOURCE_IDENTITIES_COMPLETE_NOT_SCIENTIFIC_FREEZE"
            atomic_json(state_path, state)
            return state
        except InterruptedError:
            state["status"] = "PAUSED_RESOURCE"
            atomic_json(state_path, state)
            return state
        finally:
            sources.release()
