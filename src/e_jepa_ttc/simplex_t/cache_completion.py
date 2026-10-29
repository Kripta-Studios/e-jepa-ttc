"""Exact query-family coverage checks before expensive cache payload verification."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np


def complete_d0_receipts(
    cache: Path,
    producer_family: np.ndarray,
    *,
    resource_ok: Callable[[], bool],
) -> dict[tuple[int, int], Path] | None:
    """Return all canonical receipts or None for an incomplete D0 population.

    This is coverage, not payload or authority verification. The caller must
    bind the supplied assignments to the acknowledged index and verify every
    payload before treating the replay as complete. No files are written.
    """
    if (
        producer_family.ndim != 2
        or producer_family.shape[0] != 3
        or producer_family.dtype.kind not in "iu"
    ):
        raise ValueError("three integer D0 outer-fold assignments required")
    queries = producer_family.shape[1]
    if queries < 1:
        raise ValueError("nonempty D0 query population required")
    expected = set()
    for outer, assignments in enumerate(producer_family):
        if not np.isin(assignments, np.arange(outer * 4, outer * 4 + 4)).all():
            raise ValueError("D0 assignments change outer producer family")
        expected.update((int(family), query) for query, family in enumerate(assignments))
    # Fewer files already proves incompleteness. Do not spend minutes opening
    # every small receipt just to reach that same negative coverage result.
    # Exact-size populations still require every identity and backing payload.
    paths = []
    for path in cache.glob("family*_query*.json"):
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: cache receipt coverage")
        paths.append(path)
        if len(paths) > len(expected):
            raise ValueError("too many D0 cache receipts")
    if len(paths) < len(expected):
        return None
    receipts = {}
    for path in paths:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: cache receipt coverage")
        with path.open("rb") as stream:
            payload = stream.read(65_537)
        if len(payload) > 65_536:
            raise ValueError("cache receipt exceeds metadata bound")
        record = json.loads(payload)
        family, query = record["family"], record["query"]
        if type(family) is not int or type(query) is not int:
            raise ValueError("integer cache receipt identity required")
        key = (family, query)
        if (
            key not in expected
            or key in receipts
            or path.name != f"family{family:02d}_query{query:05d}.json"
            or not path.with_suffix(".npz").is_file()
        ):
            raise ValueError("noncanonical, duplicate or unbacked D0 receipt")
        receipts[key] = path
    return receipts if set(receipts) == expected else None
