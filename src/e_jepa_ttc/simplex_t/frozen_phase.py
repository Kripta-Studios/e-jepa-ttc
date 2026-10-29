"""Execute a registered phase using identities read from the pinned freeze."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .campaign_sources import CampaignSources
from .prepared_phase import run_prepared_phase
from .scientific_freeze import read_scientific_freeze


def run_frozen_phase(
    output: Path,
    *,
    sources: CampaignSources,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    stage: str,
    availability: dict[str, bool],
    validate_authority_and_qa: Callable[[], None],
    validate_stage_gate: Callable[[str, dict[str, bool]], None],
    resource_ok: Callable[[], bool],
    resume: bool,
) -> dict:
    """Derive preparation and budget from verified bytes, never duplicate CLI values.

    Campaign-specific authority/QA and practical gate validators remain required.
    A preparation graph reserves later stages; this does not pass their gates.
    Every queue boundary rechecks the freeze and its complete pinned file set.
    """
    handed_to_queue = False
    try:
        record = read_scientific_freeze(
            freeze,
            expected_sha256=freeze_sha256,
            roots=roots,
            validate_prerequisites=validate_authority_and_qa,
        )

        def validate() -> None:
            current = read_scientific_freeze(
                freeze,
                expected_sha256=freeze_sha256,
                roots=roots,
                validate_prerequisites=validate_authority_and_qa,
            )
            if current != record:
                raise ValueError("scientific freeze changed during phase")
            validate_stage_gate(stage, dict(availability))

        validate()
        pin = record["preparation"]
        preparation = roots[pin["root"]] / pin["relative_path"]
        contract = record["source_contract"]
        handed_to_queue = True
        return run_prepared_phase(
            output,
            sources=sources,
            preparation=preparation,
            preparation_sha256=pin["sha256"],
            configuration_sha256=contract["configuration_sha256"],
            authority_sha256=contract["authority_sha256"],
            freeze_sha256=freeze_sha256,
            stage=stage,
            availability=availability,
            technical_reserved=record["technical_reserved_not_execution_claim"],
            validate_prerequisites=validate,
            resource_ok=resource_ok,
            resume=resume,
        )
    finally:
        # run_prepared_phase owns release once entered, including exceptional exits.
        if not handed_to_queue:
            sources.release()
