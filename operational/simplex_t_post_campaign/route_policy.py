"""Producer dispatch for future admitted profiling, with explicit dependency exclusions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# The interface transports canonical producer payloads without changing their types.
# ruff: noqa: ANN401

FULL = frozenset({"A5", "C2F", "PAIR"})
ALLOWED = {
    "H1_SEED7": FULL,
    "H8_SEED7": FULL,
    "H16_SEED7": FULL,
    "FULL_C0": FULL,
    "A5_ONLY_C0": frozenset({"A5"}),
    "C2F_ONLY_C0": frozenset({"C2F"}),
    "A5_PAIR_C0": frozenset({"A5", "PAIR"}),
    "SET_AGE_C0": FULL,
    "SET_NOTIME_C0": FULL,
}


@dataclass(frozen=True)
class FrozenProducers:
    """Canonical callbacks; PAIR consumes the same A5 encoding rather than a new encoder."""

    encode_a5: Callable[[Any], Any]
    a5_head: Callable[[Any], Any]
    pair_head: Callable[[Any], Any]
    c2f: Callable[[Any], Any]


def execute_producers(
    model: str,
    context: Any,
    callbacks: FrozenProducers,
    *,
    exclusive_slot_validated: bool,
    sources_and_runtime_verified: bool,
    history_dependencies: frozenset[str],
    normalization_dependencies: frozenset[str],
) -> dict[str, Any]:
    """Reject unavailable or excluded dependencies before invoking any producer.

    Cached ablation inputs do not certify an independent upstream history path.
    Real callers must bind normalizers, validity and context to the declared
    producer subset. This function does not acquire a lease or load raw data.
    """
    if model not in ALLOWED:
        raise ValueError("unregistered frozen profiling model")
    if not exclusive_slot_validated or not sources_and_runtime_verified:
        raise PermissionError(
            "live exclusive slot plus verified canonical sources/runtime required"
        )
    allowed = ALLOWED[model]
    if not history_dependencies <= allowed or not normalization_dependencies <= allowed:
        raise ValueError("history/validity/normalization depends on excluded producer")
    results = {}
    if "A5" in allowed:
        encoding = callbacks.encode_a5(context)
        results["A5"] = callbacks.a5_head(encoding)
        if "PAIR" in allowed:
            results["PAIR"] = callbacks.pair_head(encoding)
    if "C2F" in allowed:
        results["C2F"] = callbacks.c2f(context)
    if frozenset(results) != allowed:
        raise ValueError("producer interface did not emit exactly the allowed set")
    return {key: results[key] for key in ("A5", "C2F", "PAIR") if key in allowed}
