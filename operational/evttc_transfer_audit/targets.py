"""Explicit target semantics and bounded interpolation; never extrapolate."""

from __future__ import annotations

import bisect
import math
from typing import Any

from .common import canonical_sha


def align_targets(
    queries: list[dict[str, Any]],
    labels: dict[str, list[list[Any]]],
    clock_map: dict[str, int],
    target_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return one row per query with independent target validity and provenance."""
    required = {
        "unit",
        "timestamp_unit",
        "reference",
        "geometry",
        "sign",
        "interpolation",
        "zero_policy",
        "max_gap_us",
        "official_equivalence",
    }
    if not required.issubset(target_contract):
        raise ValueError("Incomplete target contract")
    if (
        target_contract["unit"] != "seconds"
        or target_contract["timestamp_unit"] != "microseconds"
        or target_contract["reference"] != "query"
        or not target_contract["geometry"]
        or target_contract["interpolation"] != "linear_ttc"
        or target_contract["sign"] not in {"signed", "positive_only"}
        or target_contract["zero_policy"] != "mae_only"
        or target_contract["official_equivalence"] != "unverified"
    ):
        raise ValueError("Unsupported target semantics; this evaluator is diagnostic, not official")
    max_gap = target_contract["max_gap_us"]
    if isinstance(max_gap, bool) or not isinstance(max_gap, int) or max_gap <= 0:
        raise ValueError("max_gap_us must be a positive integer")
    series = {}
    for sid, values in labels.items():
        times = [row[0] for row in values]
        if any(isinstance(t, bool) or not isinstance(t, int) for t in times):
            raise ValueError("Labels require integer microseconds")
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise ValueError(
                "Label timestamps must be strictly increasing; duplicates are ambiguous"
            )
        series[sid] = (times, [float(row[1]) for row in values])
    ids = [q["query_id"] for q in queries]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate query IDs")
    rows = []
    for query in queries:
        row = dict(
            query,
            target_ttc=None,
            target_status="missing_gt",
            reason="missing_labels",
            target_us=None,
            target_contract_sha256=canonical_sha(target_contract),
        )
        sid = query["sequence_id"]
        timestamp = query["query_us"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError("Queries require integer microseconds")
        if sid not in series or not series[sid][0]:
            rows.append(row)
            continue
        if sid not in clock_map:
            row["reason"] = "missing_clock_map"
            rows.append(row)
            continue
        offset = clock_map[sid]
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError("Clock offsets must be integer microseconds")
        t = timestamp + offset
        row["target_us"] = t
        times, values = series[sid]
        i = bisect.bisect_left(times, t)
        if i < len(times) and times[i] == t:
            value = values[i]
        elif i == 0 or i == len(times):
            row["reason"] = "outside_label_support"
            rows.append(row)
            continue
        elif times[i] - times[i - 1] > max_gap:
            row["reason"] = "label_gap"
            rows.append(row)
            continue
        else:
            weight = (t - times[i - 1]) / (times[i] - times[i - 1])
            value = values[i - 1] * (1 - weight) + values[i] * weight
        if not math.isfinite(value):
            row["reason"] = "nonfinite_label"
        elif value < 0 and target_contract["sign"] == "positive_only":
            row["reason"] = "negative_target_outside_contract"
        else:
            row.update(target_ttc=value, target_status="valid", reason="available")
        rows.append(row)
    return rows
