"""Connect pinned source preparation to the synchronous scientific phase queue."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .arms import resolve_arm
from .campaign_sources import CampaignSources
from .freeze_integrity import verify_stage_sources
from .phase_manifest import fit_key
from .queue import run_phase
from .registry import FitSpec, registered_graph
from .training import QuerySource


def run_prepared_phase(
    output: Path,
    *,
    sources: CampaignSources,
    preparation: Path,
    preparation_sha256: str,
    configuration_sha256: str,
    authority_sha256: str,
    freeze_sha256: str,
    stage: str,
    availability: dict[str, bool],
    technical_reserved: int,
    validate_prerequisites: Callable[[], None],
    resource_ok: Callable[[], bool],
    resume: bool,
) -> dict[str, Any]:
    """Verify both source roles before each fit; release cached arrays on exit.

    The mandatory validator must verify the scientific freeze, stage gates and
    actual authority. Source preparation is not itself a freeze or permission.
    Its graph can reserve later stages; the execution graph only enables stages
    whose gates have independently passed. Neither this adapter nor the queue
    evaluates development predictions before the phase endpoints are sealed.
    """
    try:
        validate_prerequisites()
        if preparation.stat().st_size > 8_388_608 or sha256(preparation) != preparation_sha256:
            raise ValueError("source preparation changed or exceeds metadata bound")
        state = json.loads(preparation.read_text(encoding="utf-8"))
        if (
            state.get("schema") != "simplex_t_source_preparation_v1"
            or state.get("status") != "SOURCE_IDENTITIES_COMPLETE_NOT_SCIENTIFIC_FREEZE"
            or type(state.get("optimizer_updates")) is not int
            or state["optimizer_updates"] != 0
            or state.get("scientific_freeze") is not False
            or state.get("gates_enabled") is not False
        ):
            raise ValueError("complete non-scientific source preparation required")
        contract = state["contract"]
        if (
            contract["configuration_sha256"] != configuration_sha256
            or contract["authority_sha256"] != authority_sha256
        ):
            raise ValueError("source preparation configuration or authority differs")
        prepared_graph = registered_graph(**contract["availability"])
        if sources.graph != prepared_graph or contract["fits"] != [
            {"fit": asdict(spec), "model": asdict(resolve_arm(spec, prepared_graph).model)}
            for spec in prepared_graph
        ]:
            raise ValueError("prepared source graph or model configuration differs")
        if set(state["identities"]) != {fit_key(spec) for spec in prepared_graph}:
            raise ValueError("source preparation does not contain every registered fit")
        execution_graph = registered_graph(**availability)
        if any(spec not in prepared_graph for spec in execution_graph) or any(
            availability[key] != contract["availability"][key] for key in ("d1", "density")
        ):
            raise ValueError("execution changes prepared data pools or introduces fits")
        selected = [spec for spec in execution_graph if spec.stage == stage]
        frozen = {fit_key(spec): state["identities"][fit_key(spec)] for spec in selected}
        # Validate complete canonical stage membership and both digest formats.
        verify_stage_sources(stage=stage, availability=availability, frozen=frozen, observed=frozen)

        def validate() -> None:
            validate_prerequisites()
            if sha256(preparation) != preparation_sha256:
                raise ValueError("source preparation changed during phase")

        def load(spec: FitSpec) -> QuerySource:
            key = fit_key(spec)
            train = sources.source(spec, "inner_oof")
            dev = sources.source(spec, "outer_dev")
            if frozen[key] != {
                "inner_oof": train.identity_sha256,
                "outer_dev": dev.identity_sha256,
            }:
                raise ValueError("TRAIN normalizer or OLD_DEV source differs from preparation")
            return train

        return run_phase(
            output,
            stage=stage,
            availability=availability,
            freeze_sha256=freeze_sha256,
            train_source_hashes={key: pair["inner_oof"] for key, pair in frozen.items()},
            technical_reserved=technical_reserved,
            validate_prerequisites=validate,
            source_loader=load,
            resource_ok=resource_ok,
            resume=resume,
        )
    finally:
        sources.release()
