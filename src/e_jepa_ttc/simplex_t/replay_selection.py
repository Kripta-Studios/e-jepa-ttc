"""Label-independent selection of the mandatory 64-row producer replay cohort."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence


def select_replay_rows(
    populations: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Cover all twelve producer families with 64 globally distinct TRAIN tokens.

    Each selected token is replayed through A5, C2F and its matching PAIR teacher
    family. Membership and order depend exclusively on visible input identities.
    The caller must validate TRAIN roles, table hashes and ancestor exclusions.
    """
    expected = {
        f"outer{outer}/{role}"
        for outer in range(3)
        for role in ("inner0", "inner1", "inner2", "outer_dev")
    }
    if set(populations) != expected:
        raise ValueError("replay requires all twelve exact producer families")
    used: set[str] = set()
    selected: dict[str, list[str]] = {}
    for index, family in enumerate(sorted(expected)):
        tokens = populations[family]
        if len(tokens) != len(set(tokens)) or any(not token for token in tokens):
            raise ValueError("duplicate or empty replay input identity")
        ordered = sorted(
            (token for token in tokens if token not in used),
            key=lambda token: (hashlib.sha256(token.encode("utf-8")).hexdigest(), token),
        )
        count = 6 if index < 4 else 5
        if len(ordered) < count:
            raise ValueError("insufficient distinct TRAIN rows for replay coverage")
        selected[family] = ordered[:count]
        used.update(selected[family])
    return selected
