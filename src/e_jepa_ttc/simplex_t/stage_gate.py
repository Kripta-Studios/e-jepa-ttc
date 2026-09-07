"""Bind execution-stage admission to canonical practical prediction evidence."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .campaign_sources import CampaignSources
from .history_support import frozen_train_history_support
from .practical_decisions import canonical_practical_decisions
from .registry import registered_graph


def stage_gate_from_frozen_sources(
    *,
    freeze: dict,
    sources: CampaignSources,
    scalar: tuple[pd.DataFrame, pd.DataFrame] | None,
    latent: tuple[pd.DataFrame, pd.DataFrame] | None,
    risk17: pd.DataFrame | None,
    validate_frozen_publications_and_lineage: Callable[[], None],
) -> Callable[[str, dict[str, bool]], None]:
    """Connect immutable TRAIN history support to the canonical practical gate.

    Caller validates actual freeze/source bytes and publication lineage at every
    boundary. No fraction, denominator, primary pool or candidate is hand-entered.
    """
    support = frozen_train_history_support(
        sources, freeze, validate_frozen_sources=validate_frozen_publications_and_lineage
    )
    return stage_gate_from_predictions(
        frozen_availability=freeze["source_contract"]["availability"],
        scalar=scalar,
        latent=latent,
        risk17=risk17,
        fraction_train_h8=support["fraction_train_h8"],
        validate_frozen_publications_and_lineage=validate_frozen_publications_and_lineage,
    )


def stage_gate_from_predictions(
    *,
    frozen_availability: dict[str, bool],
    scalar: tuple[pd.DataFrame, pd.DataFrame] | None,
    latent: tuple[pd.DataFrame, pd.DataFrame] | None,
    risk17: pd.DataFrame | None,
    fraction_train_h8: tuple[float, float, float],
    validate_frozen_publications_and_lineage: Callable[[], None],
) -> Callable[[str, dict[str, bool]], None]:
    """Build the stage callback consumed by ``run_frozen_phase``.

    Pairs are canonical seed7 H8/H1 predictions from sealed full-stage outputs,
    not best arms. The mandatory validator must bind the supplied frames to
    immutable publications, validate technical latent availability independently,
    and recheck those pins at every call. This function computes gates once from
    TTC, retaining only the resulting booleans, not mutable caller dataframes.
    Missing scalar results never prevent technically available T4. A missing
    RISK17 comparator is unresolved replication evidence, not a negative result.
    This validates stage allocation only, not fit, resource or holdout authority.
    """
    keys = {"d1", "density", "t3", "latent", "replicate_scalar", "replicate_latent"}
    if set(frozen_availability) != keys or any(
        type(value) is not bool for value in frozen_availability.values()
    ):
        raise ValueError("complete boolean frozen availability required")
    frozen = dict(frozen_availability)
    registered_graph(**frozen)
    validate_frozen_publications_and_lineage()
    primary = "D1" if frozen["d1"] else "D0"
    gates: dict[str, tuple[bool | None, bool | None]] = {}
    for family, pair in (("TPR", scalar), ("LATENT", latent)):
        if pair is None:
            continue
        if family == "LATENT" and not frozen["latent"]:
            raise ValueError("latent predictions supplied outside frozen technical availability")
        decision = canonical_practical_decisions(
            pair[0],
            pair[1],
            risk17,
            primary_pool=primary,
            family=family,
            fraction_train_h8=fraction_train_h8,
            validate_publication_and_lineage=validate_frozen_publications_and_lineage,
        )
        gates[family] = (decision["t3_practically_eligible"], decision["t5_practically_eligible"])

    def require(family: str, gate: int) -> None:
        if family not in gates or gates[family][gate] is None:
            raise ValueError(f"WAITING_PRACTICAL_EVIDENCE: {family}")
        if gates[family][gate] is not True:
            raise ValueError(f"PRACTICAL_GATE_NOT_PASSED: {family}")

    def validate(stage: str, availability: dict[str, bool]) -> None:
        validate_frozen_publications_and_lineage()
        if (
            stage not in {"T2", "T3", "T4", "T5"}
            or set(availability) != keys
            or any(type(value) is not bool for value in availability.values())
            or any(availability[key] and not frozen[key] for key in keys)
            or any(availability[key] != frozen[key] for key in ("d1", "density"))
        ):
            raise ValueError("execution stage changes frozen availability")
        if not any(spec.stage == stage for spec in registered_graph(**availability)):
            raise ValueError("stage has no registered enabled fits")
        if stage == "T3":
            require("TPR", 0)
        elif stage == "T5":
            for flag, family in (("replicate_scalar", "TPR"), ("replicate_latent", "LATENT")):
                if availability[flag]:
                    require(family, 1)
        # T2 is the registered factorial; T4 needs technical lineage, not T2 gains.

    return validate
