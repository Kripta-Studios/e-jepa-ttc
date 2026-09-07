"""Reconcile every completed scientific phase with frozen technical/practical gates."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .campaign_sources import CampaignSources
from .history_support import frozen_train_history_support
from .practical_decisions import canonical_practical_decisions
from .registry import registered_graph
from .scientific_freeze import read_scientific_freeze
from .sealed_analysis import load_sealed_analysis_arms
from .stage_gate import CanonicalPublication


def verify_completed_scientific_graph(
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
    """Verify actual sealed fits, including controls, against independently derived gates.

    Caller supplies OLD from the independent historical loader and a RISK17 loader
    bound to frozen acknowledged router/table inputs. This does not substitute for
    admission, analyses, compact weights or delivery: it proves scientific graph
    coverage only. The caller owns and releases source caches. Missing evidence is
    never a failed practical gate. No optimizer is constructed or holdout opened.
    """
    record = read_scientific_freeze(
        freeze,
        expected_sha256=freeze_sha256,
        roots=roots,
        validate_prerequisites=validate_authority_and_qa,
    )
    flags = dict(record["source_contract"]["availability"])
    if "T2" not in phases or (flags["latent"] and "T4" not in phases):
        raise ValueError("WAITING_PRIMARY_PHASES: complete scalar and enabled latent required")
    if set(phases) - {"T2", "T3", "T4", "T5"}:
        raise ValueError("unregistered scientific phase")

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: scientific graph completion")
        validate_authority_and_qa()
        if sha256(freeze) != freeze_sha256:
            raise ValueError("completion freeze changed")
        for binding in phases.values():
            if (
                sha256(binding.publication) != binding.publication_sha256
                or sha256(binding.endpoints) != binding.endpoints_sha256
            ):
                raise ValueError("completion publication or endpoints changed")

    boundary()
    support = frozen_train_history_support(
        sources,
        record,
        validate_frozen_sources=boundary,
    )
    arms = {
        stage: load_sealed_analysis_arms(
            binding,
            stage=stage,
            freeze=freeze,
            freeze_sha256=freeze_sha256,
            roots=roots,
            expected_queries=expected_queries,
            validate_authority_and_qa=boundary,
            resource_ok=resource_ok,
        )
        for stage, binding in phases.items()
    }
    risk17 = load_verified_risk17()
    primary = "D1" if flags["d1"] else "D0"
    decisions = {}
    for family, stage in (("TPR", "T2"), ("LATENT", "T4")):
        if family == "LATENT" and not flags["latent"]:
            continue
        decisions[family] = canonical_practical_decisions(
            arms[stage][f"{family}-{primary}-H8-C160", 7],
            arms[stage][f"{family}-{primary}-H1-C160", 7],
            risk17,
            primary_pool=primary,
            family=family,
            fraction_train_h8=support["fraction_train_h8"],
            validate_publication_and_lineage=boundary,
        )
    resolved = dict(flags)
    for flag, family, gate in (
        ("t3", "TPR", "t3_practically_eligible"),
        ("replicate_scalar", "TPR", "t5_practically_eligible"),
        ("replicate_latent", "LATENT", "t5_practically_eligible"),
    ):
        if flags[flag]:
            decision = decisions.get(family, {}).get(gate)
            if type(decision) is not bool:
                raise ValueError("WAITING_PRACTICAL_EVIDENCE: missing completion gate")
            resolved[flag] = decision
    graph = registered_graph(**resolved)
    required_stages = {spec.stage for spec in graph}
    if set(phases) != required_stages:
        raise ValueError("completed phases omit required work or include a failed-gate branch")
    for stage, binding in phases.items():
        if any(binding.availability[key] != resolved[key] for key in ("d1", "density", "latent")):
            raise ValueError("completed phase changes technical data or latent availability")
        if stage == "T5" and any(
            binding.availability[key] != resolved[key]
            for key in ("replicate_scalar", "replicate_latent")
        ):
            raise ValueError("replication phase omits or adds a practically eligible family")
        expected = {(s.name, s.seed) for s in graph if s.stage == stage}
        if set(arms[stage]) != expected:
            raise ValueError("completed phase omits a registered arm or seed")
    boundary()
    return {
        "schema": "simplex_t_completed_scientific_graph_v1",
        "status": "SCIENTIFIC_GRAPH_COVERAGE_VERIFIED_NOT_FINAL_DELIVERY",
        "freeze_sha256": freeze_sha256,
        "resolved_availability": resolved,
        "decisions": decisions,
        "train_history_support": support,
        "fits_completed": len(graph),
        "scientific_updates_completed": 2500 * len(graph),
        "phase_fit_counts": {
            stage: sum(spec.stage == stage for spec in graph) for stage in sorted(required_stages)
        },
        "optimizer_updates_executed_by_verification": 0,
        "technical_updates_included": False,
        "analyses_and_delivery_verified": False,
        "campaign_complete": False,
        "holdout_opened": False,
    }
