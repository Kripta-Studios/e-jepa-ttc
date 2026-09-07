"""Synchronous assembly of all required sealed analyses, compact heads and interface."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json

from .campaign_completion import verify_completed_scientific_graph
from .campaign_sources import CampaignSources
from .candidate_interface import publish_candidate_interface
from .compact_phase import export_compact_phase
from .followup_analysis import analyze_followup_phase, analyze_three_seed_family
from .sealed_analysis import analyze_sealed_t2
from .stage_gate import CanonicalPublication


class _AnalysisInputs(TypedDict):
    freeze: Path
    freeze_sha256: str
    roots: dict[str, Path]
    expected_queries: pd.DataFrame
    validate_authority_and_qa: Callable[[], None]
    resource_ok: Callable[[], bool]


def postprocess_completed_campaign(
    output: Path,
    *,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    sources: CampaignSources,
    phases: dict[str, CanonicalPublication],
    expected_queries: pd.DataFrame,
    load_verified_risk17: Callable[[], pd.DataFrame],
    validate_authority_and_qa: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict:
    """Run every required post-fit component, never treating a missing phase as negative.

    No fitting occurs. Inputs must be admitted, frozen, complete publications;
    the graph verifier derives practical gates from canonical predictions. A
    resource interruption preserves partial output and requires a new output
    directory for analysis retry, not refitting. Final transport packaging and
    full technical/resource accounting remain distinct required delivery steps.
    """
    work = roots["work"].resolve(strict=True)
    if output.exists() or output.resolve() == work or not output.resolve().is_relative_to(work):
        raise ValueError("new postprocessing output inside companion required")

    def verify() -> dict:
        return verify_completed_scientific_graph(
            freeze=freeze,
            freeze_sha256=freeze_sha256,
            roots=roots,
            sources=sources,
            phases=phases,
            expected_queries=expected_queries,
            load_verified_risk17=load_verified_risk17,
            validate_authority_and_qa=validate_authority_and_qa,
            resource_ok=resource_ok,
        )

    coverage = verify()
    common = _AnalysisInputs(
        freeze=freeze,
        freeze_sha256=freeze_sha256,
        roots=roots,
        expected_queries=expected_queries,
        validate_authority_and_qa=validate_authority_and_qa,
        resource_ok=resource_ok,
    )
    output.mkdir(parents=True)
    write_new_json(output / "SCIENTIFIC_GRAPH_COVERAGE.json", coverage)
    analyses, weights = {}, {}
    for stage in sorted(phases):
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: campaign postprocessing")
        binding = phases[stage]
        destination = output / "analyses" / stage
        if stage == "T2":
            analyze_sealed_t2(binding, output=destination, **common)
            name = "T2_ANALYSIS.json"
        else:
            analyze_followup_phase(
                binding,
                stage=stage,
                output=destination,
                scalar_seed7=phases["T2"] if stage == "T3" else None,
                **common,
            )
            name = "FOLLOWUP_ANALYSIS.json"
        path = destination / name
        analyses[stage] = {"path": path.relative_to(output).as_posix(), "sha256": sha256(path)}
        target = output / "compact_weights" / stage
        export_compact_phase(
            target,
            endpoints=binding.endpoints,
            endpoints_sha256=binding.endpoints_sha256,
            checkpoint_root=binding.checkpoint_root,
            freeze=freeze,
            freeze_sha256=freeze_sha256,
            roots=roots,
            stage=stage,
            availability=binding.availability,
            validate_authority_and_qa=validate_authority_and_qa,
            resource_ok=resource_ok,
            resume=False,
        )
        path = target / "COMPACT_PHASE.json"
        weights[stage] = {"path": path.relative_to(output).as_posix(), "sha256": sha256(path)}
    three_seed = {}
    flags = coverage["resolved_availability"]
    for family, flag, stage in (
        ("TPR", "replicate_scalar", "T2"),
        ("LATENT", "replicate_latent", "T4"),
    ):
        if flags[flag]:
            target = output / "three_seed" / family
            analyze_three_seed_family(
                phases[stage], phases["T5"], family=family, output=target, **common
            )
            path = target / "THREE_SEED_ANALYSIS.json"
            three_seed[family] = {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256(path),
            }
    interface = output / "TEMPORAL_CANDIDATE_FREEZE.json"
    publish_candidate_interface(
        interface,
        freeze=freeze,
        freeze_sha256=freeze_sha256,
        roots=roots,
        phases={stage: phases[stage] for stage in ("T2", "T4") if stage in phases},
        validate_authority_and_qa=validate_authority_and_qa,
        resource_ok=resource_ok,
    )
    if verify() != coverage:
        raise ValueError("scientific graph changed during postprocessing")
    result = {
        "schema": "simplex_t_campaign_postprocessing_v1",
        "status": "SEALED_ANALYSES_WEIGHTS_INTERFACE_ASSEMBLED_NOT_TRANSPORT_DELIVERY",
        "freeze_sha256": freeze_sha256,
        "graph_coverage_sha256": sha256(output / "SCIENTIFIC_GRAPH_COVERAGE.json"),
        "analyses": analyses,
        "compact_weights": weights,
        "three_seed": three_seed,
        "future_interface_sha256": sha256(interface),
        "scientific_fits_completed": coverage["fits_completed"],
        "scientific_updates_completed": coverage["scientific_updates_completed"],
        "optimizer_updates_executed": 0,
        "technical_ledger_and_resource_accounting_complete": False,
        "transport_bundle_complete": False,
        "campaign_complete": False,
        "holdout_opened": False,
    }
    write_new_json(output / "POSTPROCESSING.json", result)
    return result
