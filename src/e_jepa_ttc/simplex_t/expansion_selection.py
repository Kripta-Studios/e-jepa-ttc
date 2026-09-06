"""Bind the registered D1 pool to input-only index rows and historical producers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .pools import expansion_producer


def selected_expansion_rows(
    index: dict[str, np.ndarray],
    history: np.ndarray,
    *,
    outer: int,
    pool: dict[str, Any],
    families: list[dict[str, Any]],
    allowed_expansion_sequences: set[str],
) -> np.ndarray:
    """Return rows in frozen pool order, without accepting or inspecting labels.

    The caller must verify manifest hashes and role authority before calling.
    Inactive queries stay in the index but cannot enter training or normalization.
    Every historical slot is assigned to the current query's same producer family.
    """
    if outer not in range(3):
        raise ValueError("invalid outer fold")
    tokens, sequences = index["tokens"], index["sequences"]
    count = len(tokens)
    if tokens.ndim != 1 or sequences.shape != (count,) or len(set(tokens)) != count:
        raise ValueError("duplicate or misaligned expansion query identities")
    if not count or set(sequences) != allowed_expansion_sequences:
        raise ValueError("expansion sequence inventory differs from authorized TRAIN")
    assignments = index["producer_family"]
    if assignments.shape != (3, count) or assignments.dtype.kind != "i":
        raise ValueError("producer assignment schema mismatch")
    if history.shape != (count, 16) or history.dtype != np.int64:
        raise ValueError("history schema mismatch")
    valid = index["valid"]
    if valid.shape != history.shape or valid.dtype != bool or (history < -1).any():
        raise ValueError("history support schema mismatch")
    family_ids = assignments[outer]
    active = family_ids >= 0
    if (family_ids < -1).any() or not np.array_equal(history >= 0, valid & active[:, None]):
        raise ValueError("inactive query or history support differs from input-only index")
    fold = pool["folds"][str(outer)]
    selected = fold["additional_train_tokens"]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("empty or duplicate registered expansion pool")
    if set(selected) != set(tokens[active]):
        raise ValueError("active index differs from frozen expansion pool")
    if set(selected) & set(fold["original_train_tokens"]):
        raise ValueError("D0 and D1 token identities overlap")
    if fold["D1_train_queries"] != fold["D0_train_queries"] + len(selected):
        raise ValueError("registered D1 pool count mismatch")
    if set(fold["additional_groups"]) != allowed_expansion_sequences:
        raise ValueError("registered D1 group inventory mismatch")
    inner = [
        row for row in families if row["outer_fold"] == outer and row["role"].startswith("inner")
    ]
    by_role = {row["role"]: row["family_sha256"] for row in inner}
    if len(inner) != 3 or set(by_role) != {"inner0", "inner1", "inner2"}:
        raise ValueError("three unique historical inner families required")
    family_order = (by_role["inner0"], by_role["inner1"], by_role["inner2"])
    expected = {
        sequence: expansion_producer(str(sequence), family_order)
        for sequence in allowed_expansion_sequences
    }
    if expected != fold["sequence_family_sha256"]:
        raise ValueError("registered deterministic producer assignment changed")
    positions = {str(token): i for i, token in enumerate(tokens)}
    rows = np.asarray([positions[token] for token in selected], dtype=np.int64)
    for row in rows:
        family_id = int(family_ids[row])
        if family_id >= len(families):
            raise ValueError("unknown historical producer family")
        family = families[family_id]
        if (
            family["outer_fold"] != outer
            or family["role"] not in by_role
            or family["family_sha256"] != expected[str(sequences[row])]
        ):
            raise ValueError("query producer differs from its whole-history family")
    if not (history[rows, -1] >= 0).all():
        raise ValueError("selected expansion query lacks current observation")
    return rows
