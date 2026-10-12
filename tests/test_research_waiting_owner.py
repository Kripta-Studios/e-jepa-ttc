"""An idle context exception must reject stale states and active GPU reservations."""

from datetime import UTC, datetime, timedelta

import pytest

from operational.multimodal_research.waiting_owner import validate_idle


@pytest.mark.parametrize("status", ["WAIT_P_SENSOR", "RESOURCE_WAIT"])
def test_fresh_idle_supervisor(status: str) -> None:
    now = datetime.now(UTC)
    state = {"status": status, "reason": "GPU_OWNER_WAIT: owned", "utc": now.isoformat()}
    validate_idle(state, {"entries": [{"status": "SETTLED"}]}, now)


@pytest.mark.parametrize("reason", ["active", "stale", "reservation"])
def test_unsafe_supervisor_rejected(reason: str) -> None:
    now = datetime.now(UTC)
    state = {"status": "WAIT_P_SENSOR", "utc": now.isoformat()}
    budget = {"entries": [{"status": "SETTLED"}]}
    if reason == "active":
        state["status"] = "TRAINING"
    elif reason == "stale":
        state["utc"] = (now - timedelta(seconds=121)).isoformat()
    else:
        budget["entries"][0]["status"] = "RESERVED"
    with pytest.raises(AssertionError):
        validate_idle(state, budget, now)
