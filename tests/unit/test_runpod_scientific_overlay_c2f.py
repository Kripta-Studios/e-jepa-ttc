"""Scope, budget and cached-quota rules for three C2F cloud producers."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import pytest

from scripts.runpod_scientific_overlay_c2f_v1 import (
    C2F_MIGRATION_BUDGET_BYTES,
    QUOTA_INFLIGHT_RESERVE,
    QuotaPublisher,
    main,
    validate_budget,
    validate_runtime_budget,
)


def _documents(rotation: int) -> tuple[dict, dict]:
    end = datetime.now(UTC) + timedelta(hours=1)
    return (
        {
            "schema": "ejepa_cloud_assignment_v1",
            "maximum_compute_usd": 24,
            "execution_authorized": True,
            "execution_deadline_utc": end.isoformat(),
            "tasks": [{"rotation": rotation, "fit": "R_C2F"}],
        },
        {
            "schema": "ejepa_cloud_budget_authorization_v1",
            "maximum_total_usd": 24,
            "execution_deadline_utc": end.isoformat(),
        },
    )


@pytest.mark.parametrize("rotation", [2, 3, 4])
def test_budget_admits_exactly_one_c2f_rotation(rotation: int) -> None:
    assignment, budget = _documents(rotation)
    validate_budget(assignment, budget, rotation)
    with pytest.raises(ValueError):
        validate_budget(
            {**assignment, "tasks": [{"rotation": rotation, "fit": "R_A5"}]},
            budget,
            rotation,
        )


def test_quota_snapshot_keeps_four_gib_reserve(tmp_path) -> None:
    publisher = QuotaPublisher(tmp_path, 20_000_000_000, {"scope": "shared"})
    available, report = publisher.read(tmp_path)
    assert available == 20_000_000_000 - QUOTA_INFLIGHT_RESERVE
    assert report["inventory_off_optimizer_critical_path"] is True


def test_main_rejects_a5_before_reading_assignment(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["overlay", "producer", "--rotation", "2", "--fit", "R_A5"],
    )
    with pytest.raises(ValueError, match="R_C2F"):
        main()


def test_new_c2f_migration_requires_exact_discovered_cgroup_budget() -> None:
    validate_runtime_budget(C2F_MIGRATION_BUDGET_BYTES)
    for invalid in (46_000_000_000, 61_000_000_000):
        with pytest.raises(ValueError, match="exact"):
            validate_runtime_budget(invalid)
