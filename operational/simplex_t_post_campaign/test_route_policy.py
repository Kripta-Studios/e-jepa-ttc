"""Fixture-only dispatch tests; no raw data, encoders or optimization are executed."""

from __future__ import annotations

import pytest

from operational.simplex_t_post_campaign.route_policy import (
    ALLOWED,
    FrozenProducers,
    execute_producers,
)


def callbacks(calls: list[str]) -> FrozenProducers:
    """Return counting fixtures; they are not scientific expert implementations."""

    def encoder(value: object) -> object:
        calls.append("A5_ENCODER")
        return value

    def a5(value: object) -> object:
        calls.append("A5_HEAD")
        return value

    def pair(value: object) -> object:
        calls.append("PAIR_HEAD")
        return value

    def c2f(value: object) -> object:
        calls.append("C2F")
        return value

    return FrozenProducers(encoder, a5, pair, c2f)


@pytest.mark.parametrize("model", list(ALLOWED))
def test_dispatch_calls_exactly_allowed_producers(model: str) -> None:
    calls: list[str] = []
    result = execute_producers(
        model,
        object(),
        callbacks(calls),
        exclusive_slot_validated=True,
        sources_and_runtime_verified=True,
        history_dependencies=ALLOWED[model],
        normalization_dependencies=ALLOWED[model],
    )
    assert frozenset(result) == ALLOWED[model]
    assert list(result) == [key for key in ("A5", "C2F", "PAIR") if key in ALLOWED[model]]
    assert ("C2F" in calls) == ("C2F" in ALLOWED[model])
    assert ("A5_ENCODER" in calls) == ("A5" in ALLOWED[model])
    assert ("PAIR_HEAD" in calls) == ("PAIR" in ALLOWED[model])
    assert calls.count("A5_ENCODER") <= 1


def test_pair_reuses_same_a5_encoding_identity() -> None:
    marker = object()
    seen = []
    cb = FrozenProducers(
        lambda _: marker, lambda x: seen.append(x), lambda x: seen.append(x), lambda _: None
    )
    execute_producers(
        "A5_PAIR_C0",
        None,
        cb,
        exclusive_slot_validated=True,
        sources_and_runtime_verified=True,
        history_dependencies=frozenset({"A5", "PAIR"}),
        normalization_dependencies=frozenset({"A5"}),
    )
    assert len(seen) == 2 and all(x is marker for x in seen)


@pytest.mark.parametrize("slot,sources", [(False, True), (True, False), (False, False)])
def test_unadmitted_route_calls_nothing(slot: bool, sources: bool) -> None:
    calls: list[str] = []
    with pytest.raises(PermissionError):
        execute_producers(
            "FULL_C0",
            None,
            callbacks(calls),
            exclusive_slot_validated=slot,
            sources_and_runtime_verified=sources,
            history_dependencies=frozenset(),
            normalization_dependencies=frozenset(),
        )
    assert calls == []


@pytest.mark.parametrize("dependency", ["history", "normalization"])
def test_inherited_full_cache_cannot_claim_independent_reduced_route(dependency: str) -> None:
    calls: list[str] = []
    with pytest.raises(ValueError, match="excluded producer"):
        execute_producers(
            "C2F_ONLY_C0",
            None,
            callbacks(calls),
            exclusive_slot_validated=True,
            sources_and_runtime_verified=True,
            history_dependencies=frozenset({"A5"})
            if dependency == "history"
            else frozenset({"C2F"}),
            normalization_dependencies=frozenset({"A5"})
            if dependency == "normalization"
            else frozenset({"C2F"}),
        )
    assert calls == []
