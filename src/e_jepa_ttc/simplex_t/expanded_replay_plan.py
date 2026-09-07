"""Input-only work scheduling for D1 and matched-density inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


def expanded_family_queries(
    producer_family: np.ndarray, families: list[dict], *, queries: int
) -> dict[int, np.ndarray]:
    """Return active query IDs per inner family; never replay held-out outer families.

    D1 and DENSE indexes retain twelve historical family descriptors but assign
    only the nine inner families. A -1 assignment means inactive in that fold,
    not a missing producer that may be replaced with an outer-dev checkpoint.
    """
    if (
        type(queries) is not int
        or queries < 1
        or producer_family.shape != (3, queries)
        or producer_family.dtype.kind not in "iu"
        or len(families) != 12
    ):
        raise ValueError("expanded family index shape or dtype invalid")
    for family_id, family in enumerate(families):
        outer, slot = divmod(family_id, 4)
        if (
            family["outer_fold"] != outer
            or family["role"] != (f"inner{slot}" if slot < 3 else "outer_dev")
            or set(family["experts"]) != {"A5", "C2F", "PAIR"}
        ):
            raise ValueError("historical producer family order changed")
    result = {}
    for outer in range(3):
        assignments = producer_family[outer]
        if not np.isin(assignments, [-1, outer * 4, outer * 4 + 1, outer * 4 + 2]).all():
            raise ValueError("expanded query references outer-dev or another outer family")
        for slot in range(3):
            family_id = outer * 4 + slot
            result[family_id] = np.flatnonzero(assignments == family_id)
    return result


def inspect_expanded_replay(index_root: Path, *, manifest_sha256: str, pool: str) -> dict:
    """Inspect existing input index; no raw media, labels, checkpoints or GPU access."""
    statuses = {
        "D1": "D1_INPUT_INDEX_PREPARED_PENDING_OWNER_TIME_ACK_AND_REPLAY",
        "DENSE_OLD": "DENSE_INPUT_INDEX_PREPARED_PENDING_TIME_ACK_AND_REPLAY",
    }
    if pool not in statuses:
        raise ValueError("registered expanded pool required")
    path = index_root / "INDEX_MANIFEST.json"
    if path.stat().st_size > 1_048_576 or sha256(path) != manifest_sha256:
        raise ValueError("expanded index manifest changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] != statuses[pool]:
        raise ValueError("unexpected expanded index schema")
    array_path = index_root / "query_context_index.npz"
    if sha256(array_path) != manifest["index_sha256"]:
        raise ValueError("expanded input index changed")
    with np.load(array_path, allow_pickle=False) as archive:
        groups = expanded_family_queries(
            archive["producer_family"], manifest["families"], queries=manifest["queries"]
        )
    counts = {str(family): len(ids) for family, ids in groups.items()}
    return {
        "pool": pool,
        "index_manifest_sha256": manifest_sha256,
        "queries": manifest["queries"],
        "family_query_blocks": counts,
        "total_query_blocks": sum(counts.values()),
        "replay_authorized": False,
        "time_authority_verified": False,
        "optimizer_updates": 0,
        "scope": "Input-only scheduling; retain inactive -1 assignments and historical teachers",
    }
