"""Guard caching never bypasses stale/error checks or expands the cloud budget."""

from datetime import UTC, datetime, timedelta

import pytest

from scripts.runpod_scientific_overlay_v4 import (
    QUOTA_INFLIGHT_RESERVE,
    QuotaPublisher,
    validate_budget,
)


def test_cached_guard_retains_reserve_without_inventory_call(tmp_path, monkeypatch):
    from scripts import runpod_scientific_overlay_v4 as overlay

    def forbidden(_):
        raise AssertionError("Synchronous inventory must not run inside guard")

    monkeypatch.setattr(overlay.resources, "storage_available", forbidden)
    publisher = QuotaPublisher(tmp_path, 20_000_000_000, {"quota": 50_000_000_000})
    available, report = publisher.read(tmp_path)
    assert available == 20_000_000_000 - QUOTA_INFLIGHT_RESERVE
    assert report["inventory_off_optimizer_critical_path"] is True


def test_cached_guard_denies_stale_errors_and_wrong_root(tmp_path):
    publisher = QuotaPublisher(tmp_path, 20_000_000_000, {})
    observed, free, report, _ = publisher.snapshot
    for snapshot in [(observed - 121, free, report, None), (observed, 0, {}, "du failed")]:
        publisher.snapshot = snapshot
        with pytest.raises(RuntimeError):
            publisher.read(tmp_path)
    with pytest.raises(ValueError):
        publisher.read(tmp_path / "elsewhere")


def test_budget_cap_and_deadline_never_expand():
    end = datetime.now(UTC) + timedelta(hours=1)
    budget = {
        "schema": "ejepa_cloud_budget_authorization_v1",
        "maximum_total_usd": 24,
        "execution_deadline_utc": end.isoformat(),
    }
    assignment = {
        "maximum_compute_usd": 24,
        "execution_authorized": True,
        "execution_deadline_utc": end.isoformat(),
        "tasks": [{"rotation": 4, "fit": "R_A5"}],
    }
    validate_budget(assignment, budget)
    for change in [
        {"maximum_compute_usd": 25},
        {"execution_authorized": False},
        {"tasks": [{"rotation": 3, "fit": "R_A5"}]},
        {"execution_deadline_utc": (end + timedelta(hours=1)).isoformat()},
    ]:
        with pytest.raises(ValueError):
            validate_budget({**assignment, **change}, budget)
