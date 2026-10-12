"""Frozen cache recovery retains full resource failures and per-item deadline enforcement."""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from operational.multimodal_research import test12_cached_resume as resume


def test_full_checks_at_bounded_batch_boundaries() -> None:
    authorization = {"utc": datetime.now(UTC).isoformat(), "max_elapsed_seconds": 60}
    with patch.object(resume, "AUTHORIZATION", authorization), patch.object(
        resume.common, "guard"
    ) as guard:
        for index in range(65):
            resume.check(index, 32)
        assert guard.call_count == 3


def test_deadline_is_enforced_inside_batch() -> None:
    with patch.object(
        resume, "AUTHORIZATION", {"utc": "2000-01-01T00:00:00+00:00", "max_elapsed_seconds": 1}
    ), pytest.raises(RuntimeError, match="window exhausted"):
        resume.check(1, 32)


def test_native_guard_rejection_propagates() -> None:
    authorization = {"utc": datetime.now(UTC).isoformat(), "max_elapsed_seconds": 60}
    with patch.object(resume, "AUTHORIZATION", authorization), patch.object(
        resume.common, "guard", side_effect=MemoryError("resource floor")
    ), pytest.raises(MemoryError, match="resource floor"):
        resume.check(32, 32)
