"""Registered matched-diversity selection over an already validated D1 source."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .cache import CachedQueries
from .matched_sources import matched_source_view


def diverse_source_view(
    sources: dict[str, CachedQueries],
    *,
    pool_manifest: Path,
    pool_sha256: str,
    outer: int,
    train_tokens: np.ndarray,
    train_sequences: np.ndarray,
    train_target_ttc: np.ndarray,
    sequence_families: dict[str, str],
) -> dict[str, CachedQueries]:
    """Resolve frozen query order, not a target-dependent resample of D1.

    Caller supplies the D1 TRAIN token/sequence/target order and verified producer
    family hashes. Both controls must retain the registered equal query count.
    This function executes no experts, fits or evaluations.
    """
    if outer not in range(3) or set(sources) != {"inner_oof", "outer_dev"}:
        raise ValueError("registered fold and separate D1 source roles required")
    if (
        len(pool_sha256) != 64
        or set(pool_sha256) - set("0123456789abcdef")
        or compute_file_hash(str(pool_manifest)) != pool_sha256
    ):
        raise ValueError("matched pool manifest changed")
    population = sources["inner_oof"].population
    if (
        train_tokens.shape != (population,)
        or train_sequences.shape != train_tokens.shape
        or train_target_ttc.shape != train_tokens.shape
        or len(np.unique(train_tokens)) != population
    ):
        raise ValueError("D1 TRAIN identity alignment mismatch")
    plan = json.loads(pool_manifest.read_text(encoding="utf-8"))
    folds = [record for record in plan["folds"] if record["outer"] == outer]
    if len(folds) != 1:
        raise ValueError("ambiguous matched fold")
    fold = folds[0]
    selection = fold["pools"]["DIVERSE_MATCHED"]
    tokens = selection["tokens"]
    if (
        not tokens
        or len(set(tokens)) != len(tokens)
        or len(tokens) != fold["nominal_common_count"]
        or len(tokens) != len(fold["pools"]["DENSE_OLD"]["tokens"])
    ):
        raise ValueError("matched count or query uniqueness changed")
    positions = {str(token): row for row, token in enumerate(train_tokens)}
    if not set(tokens) <= set(positions):
        raise ValueError("matched query outside validated D1 TRAIN")
    rows = np.asarray([positions[token] for token in tokens], np.int64)
    groups = set(train_sequences[rows])
    if groups != set(selection["sequences"]):
        raise ValueError("matched sequence membership changed")
    if set(selection["sequence_family_sha256"]) != groups or any(
        selection["sequence_family_sha256"][group] != sequence_families.get(str(group))
        for group in groups
    ):
        raise ValueError("matched history producer family changed")
    return matched_source_view(
        sources,
        selected_rows=rows,
        train_target_ttc=train_target_ttc,
        train_sequences=train_sequences,
        pool="DIVERSE_MATCHED",
        pool_manifest_sha256=pool_sha256,
    )
