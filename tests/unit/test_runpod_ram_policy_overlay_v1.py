"""Bounded tests for the additive real-cgroup RAM policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import runpod_ram_policy_overlay_v1 as policy


def test_effective_limit_is_minimum_finite_ancestor() -> None:
    reports = [
        {"kernel_limit": 61_000_000_000},
        {"kernel_limit": 80_000_000_000},
        {"kernel_limit": 2**63 - 1},
    ]
    assert policy.effective_finite_limit(reports) == 61_000_000_000


@pytest.mark.parametrize(
    "reports",
    [
        [],
        [{"kernel_limit": "max"}],
        [{"kernel_limit": 2**63 - 1}],
        [{"kernel_limit": 2 * 1024**3}],
    ],
)
def test_effective_limit_rejects_missing_invalid_or_unlimited(reports: list[dict]) -> None:
    with pytest.raises(ValueError):
        policy.effective_finite_limit(reports)


def test_runtime_monitor_replaces_historical_cap_and_preserves_base_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reports = [{"kernel_limit": 61_000_000_000, "path": "/cgroup"}]
    monkeypatch.setattr(policy, "discover_effective_limit", lambda: (61_000_000_000, reports))
    monkeypatch.setattr(policy.resources.os, "name", "posix")

    observed = {}

    def base_guard(self, run_root, *, checkpoint_reservation, completed_updates):
        observed.update(
            budget=self.budget_bytes,
            run_root=run_root,
            checkpoint_reservation=checkpoint_reservation,
            completed_updates=completed_updates,
        )
        return True, {
            "schema": "ejepa_linux_resources_v3",
            "host_available_bytes": 30_000_000_000,
        }

    monkeypatch.setattr(policy.resources.LinuxResourceMonitor, "guard", base_guard)
    monitor = policy.RealCgroupLinuxResourceMonitor(budget_bytes=46_000_000_000)
    allowed, report = monitor.guard(tmp_path, checkpoint_reservation=1024**3, completed_updates=123)

    assert allowed is True
    assert observed["budget"] == 61_000_000_000
    assert report["historical_migration_budget_bytes"] == 46_000_000_000
    assert report["runtime_effective_cgroup_limit_bytes"] == 61_000_000_000
    assert report["physical_reserve_bytes"] == 2 * 1024**3
    assert report["ram_policy_override_active"] is True


def test_discovery_uses_memory_snapshot_ancestry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    membership = tmp_path / "cgroup"
    membership.write_text("0::/pod\n", encoding="utf-8")
    reports = [{"kernel_limit": 61_000_000_000}, {"kernel_limit": 70_000_000_000}]
    seen = {}

    def snapshot(root, text, budget):
        seen.update(root=root, text=text, budget=budget)
        return 10, reports

    monkeypatch.setattr(policy.resources, "memory_snapshot", snapshot)
    effective, returned = policy.discover_effective_limit(tmp_path, membership)

    assert effective == 61_000_000_000
    assert returned == reports
    assert seen["text"] == "0::/pod\n"
    assert seen["budget"] == policy.UNLIMITED_LIMIT_THRESHOLD
