"""Resolve matched queries to exact producer-bound input indices, not anonymous rows."""

from __future__ import annotations

from typing import Any

import numpy as np

SOURCE_NAMES = ("D0", "D1", "DENSE")


def control_references(
    *,
    pool: str,
    selection: dict[str, Any],
    outer: int,
    indices: dict[str, dict[str, np.ndarray]],
    families: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Prefer D0 reuse only after exact equality with its dense input counterpart.

    The caller verifies all index/pool manifest hashes first. References establish
    required inputs, not cache completeness, extraction authorization or fit gates.
    """
    if pool not in {"DENSE_OLD", "DIVERSE_MATCHED"} or outer not in range(3):
        raise ValueError("unregistered control pool or fold")
    if set(indices) != set(SOURCE_NAMES):
        raise ValueError("D0/D1/dense input bindings required")
    positions = {}
    for name, index in indices.items():
        tokens = index["tokens"]
        if tokens.ndim != 1 or len(set(tokens)) != len(tokens):
            raise ValueError("ambiguous query identity in input index")
        positions[name] = {str(token): i for i, token in enumerate(tokens)}
    tokens = selection["tokens"]
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("unique nonempty matched query selection required")
    codes, rows, producer_ids = [], [], []
    for token in tokens:
        present = [name for name in SOURCE_NAMES if token in positions[name]]
        if pool == "DENSE_OLD":
            if "DENSE" not in present or "D1" in present:
                raise ValueError("dense query outside original TRAIN input universe")
            source = "D0" if "D0" in present else "DENSE"
        else:
            candidates = [name for name in ("D0", "D1") if name in present]
            if len(candidates) != 1:
                raise ValueError("diverse query lacks one original/expansion input identity")
            source = candidates[0]
        row = positions[source][token]
        index = indices[source]
        if source == "D0" and "DENSE" in present:
            other = positions["DENSE"][token]
            for field in (
                "sequences",
                "base_windows_us",
                "square_xyxy",
                "anchor_us",
                "roi_available_us",
                "valid",
            ):
                if not np.array_equal(index[field][row], indices["DENSE"][field][other]):
                    raise ValueError("D0 reuse changes producer input dependencies")
            if not np.array_equal(index["lag_us"], indices["DENSE"]["lag_us"]):
                raise ValueError("D0 reuse changes lag schedule")
        sequence = str(index["sequences"][row])
        family_id = int(index["producer_family"][outer, row])
        if family_id < 0 or family_id >= len(families):
            raise ValueError("control query has no active producer")
        family = families[family_id]
        if (
            family["outer_fold"] != outer
            or family["role"] == "outer_dev"
            or family["family_sha256"] != selection["sequence_family_sha256"].get(sequence)
            or sequence not in selection["sequences"]
        ):
            raise ValueError("control query producer differs from registered TRAIN family")
        if not bool(index["valid"][row, -1]):
            raise ValueError("control query lacks current source support")
        codes.append(SOURCE_NAMES.index(source))
        rows.append(row)
        producer_ids.append(family_id)
    return {
        "tokens": np.asarray(tokens),
        "source": np.asarray(codes, np.int8),
        "query_row": np.asarray(rows, np.int64),
        "producer_family": np.asarray(producer_ids, np.int16),
    }
