"""User-authorized 1 GiB commit reserve, preserving all other resource gates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

COMMIT_RESERVE_BYTES = 1024**3


def resource_decision(values: Mapping[str, Any], *, reserve_bytes: int) -> bool:
    """Evaluate the original full/fast gates with one changed bound."""
    commit = values["windows_commit_headroom_bytes"]
    return bool(
        values["host_available_bytes"] >= 2 * 1024**3
        and (commit is None or commit >= reserve_bytes)
        and values["disk_free_after_reservation_bytes"] >= 10_000_000_000
        and values.get("project_tree_rss_bytes", 0) <= 23_000_000_000
    )


def with_requested_reserve(
    original: tuple[bool, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Reject unknown upstream policy changes before applying the requested reserve."""
    allowed, values = original
    if allowed != resource_decision(values, reserve_bytes=3 * 1024**3):
        raise RuntimeError("Upstream resource policy changed; cannot apply reserve overlay")
    return resource_decision(values, reserve_bytes=COMMIT_RESERVE_BYTES), {
        **values,
        "commit_reserve_bytes": COMMIT_RESERVE_BYTES,
        "previous_commit_reserve_bytes": 3 * 1024**3,
        "commit_reserve_authorization": "explicit_user_request_20261010",
    }
