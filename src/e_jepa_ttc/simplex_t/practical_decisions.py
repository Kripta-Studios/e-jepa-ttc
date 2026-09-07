"""Canonical seed7 practical decisions derived from paired development outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

import pandas as pd

from .evaluation import may_explore_context
from .gates import Evidence, may_replicate
from .practical_comparison import paired_practical_comparison


def canonical_practical_decisions(
    candidate: pd.DataFrame,
    h1: pd.DataFrame,
    risk17: pd.DataFrame | None,
    *,
    primary_pool: str,
    family: str,
    fraction_train_h8: tuple[float, float, float],
    validate_publication_and_lineage: Callable[[], None],
) -> dict:
    """No best-arm substitution or CI prerequisite for exploration/replication.

    Caller must validate frozen publication bytes, complete phase endpoints,
    independent OLD identities and RISK17 producer lineage before supplying data.
    The result measures practical eligibility, not resource or holdout authority.
    T4 technical availability must be determined independently of this function.
    """
    validate_publication_and_lineage()
    if primary_pool not in {"D0", "D1"} or family not in {"TPR", "LATENT"}:
        raise ValueError("registered canonical family and frozen primary pool required")
    names = [f"{family}-{primary_pool}-H{h}-C160" for h in (8, 1)]
    for frame, name in zip((candidate, h1), names, strict=True):
        if (
            not {"arm", "seed"} <= set(frame)
            or not frame.arm.eq(name).all()
            or not frame.seed.eq(7).all()
        ):
            raise ValueError("canonical seed7 H8 and same-family H1 required")
    current = paired_practical_comparison(candidate, h1)
    context = family == "TPR" and may_explore_context(
        delta_h1=current["point_delta"],
        finite=current["candidate_finite_fraction"] == 1.0,
        sign_delta=current["weighted_sign_error_delta"],
        crucial_delta=current["crucial_bucket_mid_delta"],
        fraction_train_h8=fraction_train_h8,
        integrity=True,
    )
    risk_comparison, evidence = None, None
    replication = None
    if risk17 is not None:
        risk_comparison = paired_practical_comparison(candidate, risk17)
        evidence = Evidence(
            delta_risk17=risk_comparison["point_delta"],
            delta_current_control=current["point_delta"],
            sequence_wins=risk_comparison["sequence_wins"],
            sequence_count=risk_comparison["sequence_count"],
            finite_fraction=risk_comparison["candidate_finite_fraction"],
            sign_error_delta=risk_comparison["weighted_sign_error_delta"],
            crucial_delta=risk_comparison["crucial_bucket_mid_delta"],
            integrity=True,
        )
        replication = may_replicate(evidence)
    validate_publication_and_lineage()
    return {
        "canonical_candidate": names[0],
        "h1_control": names[1],
        "seed": 7,
        "versus_h1": current,
        "versus_risk17": risk_comparison,
        "replication_evidence": asdict(evidence) if evidence is not None else None,
        "t3_practically_eligible": context if family == "TPR" else None,
        "t5_practically_eligible": replication,
        "t5_status": "WAITING_RISK17_COMPARATOR" if risk17 is None else "PRACTICAL_GATE_EVALUATED",
        "t4_enabled_by_this_result": False,
        "confidence_interval_required_for_replication": False,
        "scientific_execution_authorized": False,
    }
