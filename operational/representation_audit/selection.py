"""At most one conditional replication family across the remaining queues."""

from __future__ import annotations

from pathlib import Path

from operational.range_transfer.common import freeze, read, sha

from .run import RUN


def claim_family(family: str, decision: Path) -> None:
    """Keep the first admitted registered family; never replace it after later results."""
    value = read(decision)
    if value["status"] != "REPLICATE" or not all(v["eligible"] for v in value["gates"].values()):
        raise ValueError("Conditional replication requires every fixed screen gate")
    freeze(
        RUN / "SELECTED_REPLICATION_FAMILY.json",
        {
            "family": family,
            "screen_sha256": sha(decision),
            "rule": "first admitted queue in execution order; at most one conditional family",
        },
    )
