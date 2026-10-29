"""Derive practical history support from frozen TRAIN sources, never TTC scores."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .campaign_sources import CampaignSources
from .phase_manifest import fit_key
from .registry import registered_graph


def history_support(history: np.ndarray) -> dict:
    """Count full H8 histories over every TRAIN query, including cold starts."""
    if (
        history.ndim != 2
        or history.shape[1] != 16
        or not len(history)
        or history.dtype.kind != "i"
        or (history < -1).any()
    ):
        raise ValueError("complete nonempty H16 TRAIN history index required")
    valid = history >= 0
    if not valid[:, -1].all() or (valid[:, :-1] & ~valid[:, 1:]).any():
        raise ValueError("history must be a causal suffix retaining current observations")
    full = int(valid[:, -8:].all(axis=1).sum())
    return {
        "queries": len(history),
        "full_h8": full,
        "fraction_train_h8": full / len(history),
        "full_h16": int(valid.all(axis=1).sum()),
        "cold_start_h8": len(history) - full,
    }


def frozen_train_history_support(
    sources: CampaignSources,
    freeze: dict,
    *,
    validate_frozen_sources: Callable[[], None],
) -> dict:
    """Require canonical T2 TRAIN identities for all three folds before returning.

    The caller owns sources and must release them. This does not enable T3;
    the separate practical gain/guardrail condition still requires sealed scores.
    """
    validate_frozen_sources()
    flags = freeze["source_contract"]["availability"]
    graph = registered_graph(**flags)
    if sources.graph != graph:
        raise ValueError("history sources differ from frozen graph")
    primary = "D1" if flags["d1"] else "D0"
    canonical = f"TPR-{primary}-H8-C160"
    if freeze["canonical_scalar"] != canonical:
        raise ValueError("history support cannot choose a different primary")
    records = []
    for spec in sorted(
        (s for s in graph if s.stage == "T2" and s.name == canonical), key=lambda s: s.fold
    ):
        validate_frozen_sources()
        source = sources.source(spec, "inner_oof")
        expected = freeze["source_identities"][fit_key(spec)]["inner_oof"]
        if source.identity_sha256 != expected or source.length != 8 or source.control != "NONE":
            raise ValueError("canonical frozen TRAIN history source differs")
        records.append(
            dict(fold=spec.fold, source_sha256=expected, **history_support(source.history))
        )
    if [item["fold"] for item in records] != [0, 1, 2]:
        raise ValueError("all three TRAIN folds required for history condition")
    validate_frozen_sources()
    return {
        "primary_pool": primary,
        "canonical": canonical,
        "folds": records,
        "fraction_train_h8": tuple(item["fraction_train_h8"] for item in records),
        "scientific_gate_authorized": False,
    }
