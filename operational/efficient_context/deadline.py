"""Pause work at a complete optimizer boundary when the user's72-hour budget expires."""

from datetime import UTC, datetime

from .common import Campaign, atomic_json, digest, read


def permits(c: Campaign) -> bool:
    """Keep unfinished fits resumable; never turn a deadline into a negative comparator result."""
    path = c.out / "EXECUTION_DEADLINE.json"
    if not path.exists():
        return True
    value = read(path)
    if value["parent_protocol_sha256"] != digest(c.out / "PROTOCOL.json"):
        raise ValueError("execution deadline differs from the authorized campaign")
    if datetime.now(UTC) < datetime.fromisoformat(value["deadline_utc"]):
        return True
    atomic_json(
        c.out / "TIME_BUDGET_PAUSE.json",
        {
            "status": "PAUSED_USER_TIME_BUDGET",
            "deadline_utc": value["deadline_utc"],
            "scientific_negative": False,
            "partial_producers_not_admissible_as_completed_comparators": True,
        },
    )
    return False
