"""Bind execution-stage admission to canonical practical prediction evidence."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .campaign_sources import CampaignSources
from .canonical_publication import load_canonical_publication
from .history_support import frozen_train_history_support
from .practical_decisions import canonical_practical_decisions
from .registry import registered_graph


@dataclass(frozen=True)
class CanonicalPublication:
    """Complete scientific phase output, not an individual winning candidate."""

    publication: Path
    publication_sha256: str
    endpoints: Path
    endpoints_sha256: str
    checkpoint_root: Path
    availability: dict[str, bool]


def stage_gate_from_publications(
    *,
    stage: str,
    availability: dict[str, bool],
    freeze: dict,
    freeze_sha256: str,
    sources: CampaignSources,
    publications: dict[str, CanonicalPublication],
    expected_queries: pd.DataFrame,
    load_verified_risk17: Callable[[], pd.DataFrame] | None,
    validate_frozen_sources_cohort_and_lineage: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> Callable[[str, dict[str, bool]], None]:
    """Resolve only this stage's canonical evidence and derive its practical gate.

    T2/T4 never read scalar publications or require a RISK17 comparison. T3 reads
    only the complete scalar seed7 phase. T5 independently reads each enabled
    family's phase plus an independently validated historical RISK17 replay.
    Every callback invocation revalidates the actual phase files and endpoints;
    no caller-entered score, best-arm name or significance flag is accepted.
    The mandatory lineage validator binds the freeze, sources, cohort and the
    RISK17 loader's independent router/table references to acknowledged inputs.
    """
    flags = dict(availability)
    if stage not in {"T2", "T3", "T4", "T5"}:
        raise ValueError("registered scientific stage required")
    keys = {"d1", "density", "t3", "latent", "replicate_scalar", "replicate_latent"}
    frozen_flags = freeze["source_contract"]["availability"]
    if (
        set(flags) != keys
        or set(frozen_flags) != keys
        or any(type(value) is not bool for value in (*flags.values(), *frozen_flags.values()))
        or any(flags[key] and not frozen_flags[key] for key in keys)
        or any(flags[key] != frozen_flags[key] for key in ("d1", "density"))
        or not any(spec.stage == stage for spec in registered_graph(**flags))
    ):
        raise ValueError("publication gate changes frozen execution allocation")
    required = (
        ["TPR"]
        if stage == "T3"
        else [
            family
            for key, family in (("replicate_scalar", "TPR"), ("replicate_latent", "LATENT"))
            if flags[key]
        ]
        if stage == "T5"
        else []
    )
    if any(family not in publications for family in required):
        raise ValueError("WAITING_CANONICAL_PUBLICATION: incomplete stage evidence")
    if stage == "T5" and load_verified_risk17 is None:
        raise ValueError("WAITING_RISK17_COMPARATOR: independently validated replay required")
    # Copy mutable availability dictionaries, retaining only the needed families.
    pins = {
        family: CanonicalPublication(
            pin.publication,
            pin.publication_sha256,
            pin.endpoints,
            pin.endpoints_sha256,
            pin.checkpoint_root,
            dict(pin.availability),
        )
        for family in required
        for pin in [publications[family]]
    }
    cohort = expected_queries.copy(deep=True)

    def read_family(family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        pin = pins[family]
        frozen = freeze["source_contract"]["availability"]
        if any(pin.availability[key] != frozen[key] for key in ("d1", "density")):
            raise ValueError("canonical publication changes frozen data pools")
        return load_canonical_publication(
            pin.publication,
            publication_sha256=pin.publication_sha256,
            endpoints=pin.endpoints,
            endpoints_sha256=pin.endpoints_sha256,
            checkpoint_root=pin.checkpoint_root,
            freeze_sha256=freeze_sha256,
            availability=pin.availability,
            family=family,
            expected_queries=cohort,
            validate_authority=validate_frozen_sources_cohort_and_lineage,
            resource_ok=resource_ok,
        )

    def validate() -> None:
        validate_frozen_sources_cohort_and_lineage()
        if not expected_queries.equals(cohort):
            raise ValueError("independently verified gate cohort changed")
        for family in required:
            read_family(family)
        if stage == "T5":
            assert load_verified_risk17 is not None
            load_verified_risk17()

    validate_frozen_sources_cohort_and_lineage()
    pairs = {family: read_family(family) for family in required}
    risk17 = load_verified_risk17() if stage == "T5" and load_verified_risk17 else None
    gate = stage_gate_from_frozen_sources(
        freeze=freeze,
        sources=sources,
        scalar=pairs.get("TPR"),
        latent=pairs.get("LATENT"),
        risk17=risk17,
        validate_frozen_publications_and_lineage=validate,
    )

    def check(requested_stage: str, requested_flags: dict[str, bool]) -> None:
        if requested_stage != stage or requested_flags != flags:
            raise ValueError("stage gate cannot be reused for a different execution allocation")
        gate(requested_stage, requested_flags)

    return check


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
