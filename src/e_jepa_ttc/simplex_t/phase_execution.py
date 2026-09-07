"""Synchronous frozen fitting followed by complete sealed-phase publication."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .campaign_sources import CampaignSources
from .frozen_phase import run_frozen_phase
from .phase_export import export_phase
from .phase_inference import validated_phase
from .phase_manifest import fit_key
from .registry import registered_graph
from .scientific_freeze import read_scientific_freeze


def run_and_publish_frozen_phase(
    execution: Path,
    publication: Path,
    *,
    sources: CampaignSources,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    stage: str,
    availability: dict[str, bool],
    expected_queries: pd.DataFrame,
    validate_authority_qa_and_cohort: Callable[[], None],
    validate_stage_gate: Callable[[str, dict[str, bool]], None],
    resource_ok: Callable[[], bool],
    resume: bool,
) -> dict:
    """Finish one registered phase, then publish OLD predictions, without selection.

    The required callbacks must verify actual campaign authority/QA, the supplied
    OLD cohort and practical gates. They have no permissive default. This joins
    the fit and publication engines, not the still-separate authority admission.
    A resource pause during fitting never opens evaluation inputs. On resumption
    an existing complete seal is verified and exported without repeating fits.
    All T2--T5 stages use the same frozen 2500-update endpoint machinery.
    """
    flags = dict(availability)
    # Detach mutable caller data used by long-running publication.
    cohort = expected_queries.copy(deep=True)
    sources_released = False
    try:
        record = read_scientific_freeze(
            freeze,
            expected_sha256=freeze_sha256,
            roots=roots,
            validate_prerequisites=validate_authority_qa_and_cohort,
        )

        def validate() -> None:
            current = read_scientific_freeze(
                freeze,
                expected_sha256=freeze_sha256,
                roots=roots,
                validate_prerequisites=validate_authority_qa_and_cohort,
            )
            if current != record or not expected_queries.equals(cohort):
                raise ValueError("phase freeze or independently verified OLD cohort changed")
            validate_stage_gate(stage, dict(flags))

        validate()
        identity = ["sample_token", "sequence_id", "track_id", "outer_fold"]
        if (
            not set(identity) <= set(cohort)
            or len(cohort) != 8192
            or cohort.sample_token.duplicated().any()
            or cohort.loc[:, identity].isna().to_numpy().any()
            or cohort.sequence_id.nunique() != 9
            or set(cohort.outer_fold) != {0, 1, 2}
        ):
            raise ValueError("complete acknowledged OLD8192 cohort required before fitting")
        selected = [spec for spec in registered_graph(**flags) if spec.stage == stage]
        if not selected or any(spec not in sources.graph for spec in selected):
            raise ValueError("phase lacks complete registered sources")
        frozen_flags = record["source_contract"]["availability"]
        if any(flags[key] != frozen_flags[key] for key in ("d1", "density")) or any(
            enabled and not frozen_flags[key] for key, enabled in flags.items()
        ):
            raise ValueError("phase changes frozen pools or enables an unavailable branch")
        endpoints = execution / f"{stage}_ENDPOINTS.json"
        if endpoints.exists():
            if not resume:
                raise FileExistsError("sealed phase requires explicit publication resume")
            training_status = "EXISTING_SEAL_REQUIRES_VALIDATION"
        else:
            # run_frozen_phase owns release on every exit, but the adapter can
            # subsequently reload the same pinned sources for export.
            sources_released = True
            training = run_frozen_phase(
                execution,
                sources=sources,
                freeze=freeze,
                freeze_sha256=freeze_sha256,
                roots=roots,
                stage=stage,
                availability=flags,
                validate_authority_and_qa=validate_authority_qa_and_cohort,
                validate_stage_gate=validate_stage_gate,
                resource_ok=resource_ok,
                resume=resume,
            )
            training_status = training["status"]
            if training_status == "PAUSED_RESOURCE":
                return {
                    "status": "PAUSED_RESOURCE",
                    "phase": stage,
                    "training": training,
                    "publication_started": False,
                }
            if training_status != "ENDPOINTS_SEALED_NOT_EVALUATED":
                raise ValueError("phase cannot publish before complete endpoint sealing")
        validate()
        endpoint_hash = sha256(endpoints)
        # Verify every endpoint state before export can load any OLD population.
        validated_phase(
            endpoints,
            execution / "fits",
            manifest_sha256=endpoint_hash,
            freeze_sha256=freeze_sha256,
            stage=stage,
            availability=flags,
            resource_ok=resource_ok,
        )
        validate()
        dev_hashes = {
            fit_key(spec): record["source_identities"][fit_key(spec)]["outer_dev"]
            for spec in selected
        }
        sources_released = True  # export_phase owns release, including exceptions.
        exported = export_phase(
            publication,
            sources=sources,
            manifest=endpoints,
            checkpoint_root=execution / "fits",
            manifest_sha256=endpoint_hash,
            freeze_sha256=freeze_sha256,
            stage=stage,
            availability=flags,
            dev_source_hashes=dev_hashes,
            expected_queries=cohort,
            validate_prerequisites=validate,
            resource_ok=resource_ok,
            resume=resume,
        )
        return {
            "status": exported["status"],
            "phase": stage,
            "training_status": training_status,
            "endpoint_sha256": endpoint_hash,
            "publication": exported,
            "next_stage_authorized": False,
            "holdout_opened": False,
        }
    finally:
        if not sources_released:
            sources.release()
