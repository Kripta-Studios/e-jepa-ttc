"""Synchronous scientific entry point with mandatory real admission and stage gates."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .configuration_preflight import open_acknowledged_source_configuration
from .phase_execution import run_historical_cohort_phase
from .scientific_admission import validate_scientific_admission
from .stage_gate import CanonicalPublication


def execute_configured_phase(
    *,
    local_paths: Path,
    source_configuration: Path,
    source_configuration_sha256: str,
    evidence_profile: Path,
    evidence_profile_sha256: str,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    execution: Path,
    publication: Path,
    stage: str,
    availability: dict[str, bool],
    publications: dict[str, CanonicalPublication],
    resource_ok: Callable[[], bool],
    resume: bool,
) -> dict:
    """Run fit → complete endpoint seal → OLD publication, or exact resource pause.

    There is no caller-supplied authority/gate bypass. All real QA is mandatory;
    the lower-level reader independently reconstructs and verifies the canonical
    freeze and code inventory. Publication is not final T6 analysis. Reusing this
    entry point for a later stage requires its separately verified practical gate.
    """
    if stage not in {"T2", "T3", "T4", "T5"}:
        raise ValueError("only registered scientific stages can execute")
    work = Path(json.loads(local_paths.read_text(encoding="utf-8"))["worktree"]).resolve(
        strict=True
    )
    for output in (execution, publication):
        target = output.resolve()
        if target == work or not target.is_relative_to(work):
            raise ValueError("scientific outputs must be inside the companion worktree")
    if execution.resolve() == publication.resolve():
        raise ValueError("execution and prediction publication require distinct directories")
    if freeze.stat().st_size > 8_388_608 or sha256(freeze) != freeze_sha256:
        raise ValueError("scientific freeze bytes changed")
    record = json.loads(freeze.read_text(encoding="utf-8"))

    def validate() -> None:
        if sha256(freeze) != freeze_sha256:
            raise ValueError("scientific freeze changed during execution")
        validate_scientific_admission(
            record,
            roots=roots,
            local_paths=local_paths,
            source_configuration=source_configuration,
            source_configuration_sha256=source_configuration_sha256,
            evidence_profile=evidence_profile,
            evidence_profile_sha256=evidence_profile_sha256,
            resource_ok=resource_ok,
        )

    # Fail before constructing fit sources, output directories or optimizer state.
    validate()
    paths = json.loads(local_paths.read_text(encoding="utf-8"))
    sources, _ = open_acknowledged_source_configuration(
        local_paths, source_configuration, source_configuration_sha256
    )
    # Ownership transfers to run_historical_cohort_phase, including failures.
    return run_historical_cohort_phase(
        execution,
        publication,
        sources=sources,
        freeze=freeze,
        freeze_sha256=freeze_sha256,
        roots=roots,
        stage=stage,
        availability=availability,
        validate_authority_and_qa=validate,
        validate_stage_gate=None,
        publications=publications,
        risk17_ack=Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json",
        resource_ok=resource_ok,
        resume=resume,
    )
