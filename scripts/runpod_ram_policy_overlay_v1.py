"""Add an explicit real-cgroup RAM policy without changing scientific identity.

The historical 46 GB command-line value remains part of MIGRATION_BINDING.json.
Only the live resource monitor is replaced: it uses the smallest finite kernel
limit visible across the process' cgroup ancestry and retains every other v3
guard (2 GiB reserve, host MemAvailable, strict commit, deadline and disk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

if __package__:
    from . import runpod_resources_v3 as resources
    from . import runpod_scientific_overlay_v5 as v5
else:
    import runpod_resources_v3 as resources
    import runpod_scientific_overlay_v5 as v5

from operational.runpod_migration.inventory import digest

GIB = 1024**3
PHYSICAL_RESERVE_BYTES = 2 * GIB
UNLIMITED_LIMIT_THRESHOLD = 2**60


def effective_finite_limit(reports: list[dict[str, Any]]) -> int:
    """Return the tightest finite kernel limit across all cgroup ancestors."""
    if not reports:
        raise ValueError("Cgroup memory ancestry is empty")
    limits = []
    for report in reports:
        limit = report.get("kernel_limit")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("Cgroup memory ancestry contains an invalid limit")
        limits.append(limit)
    effective = min(limits)
    if effective >= UNLIMITED_LIMIT_THRESHOLD:
        raise ValueError("Cgroup memory ancestry has no finite effective limit")
    if effective <= PHYSICAL_RESERVE_BYTES:
        raise ValueError("Finite cgroup limit does not exceed the 2 GiB reserve")
    return effective


def discover_effective_limit(
    root: Path = Path("/sys/fs/cgroup"),
    membership_path: Path = Path("/proc/self/cgroup"),
) -> tuple[int, list[dict[str, Any]]]:
    """Discover v1/v2 ancestry using the already tested v3 cgroup parser."""
    _headroom, reports = resources.memory_snapshot(
        root,
        membership_path.read_text(encoding="utf-8"),
        UNLIMITED_LIMIT_THRESHOLD,
    )
    return effective_finite_limit(reports), reports


class RealCgroupLinuxResourceMonitor(resources.LinuxResourceMonitor):
    """Dynamically replace the historical cap with the real finite cgroup limit."""

    def __init__(
        self, *args: object, budget_bytes: int = resources.DEFAULT_BUDGET_BYTES, **kwargs: object
    ) -> None:
        self.historical_budget_bytes = budget_bytes
        effective, _reports = discover_effective_limit()
        super().__init__(*args, budget_bytes=effective, **kwargs)

    def guard(
        self, run_root: Path, *, checkpoint_reservation: int, completed_updates: int
    ) -> tuple[bool, dict[str, Any]]:
        """Refresh the effective ancestry limit at every optimizer boundary."""
        try:
            effective, discovery = discover_effective_limit()
        except (OSError, ValueError, KeyError, RuntimeError, IndexError) as error:
            return False, {
                "schema": "ejepa_linux_resources_real_cgroup_v1",
                "error": str(error),
                "host_available_bytes": 0,
                "windows_commit_headroom_bytes": None,
                "ram_policy_override_active": True,
            }
        self.budget_bytes = effective
        allowed, report = super().guard(
            run_root,
            checkpoint_reservation=checkpoint_reservation,
            completed_updates=completed_updates,
        )
        report.update(
            schema="ejepa_linux_resources_real_cgroup_v1",
            historical_migration_budget_bytes=self.historical_budget_bytes,
            runtime_effective_cgroup_limit_bytes=effective,
            runtime_limit_source="minimum finite kernel limit across accessible cgroup ancestry",
            runtime_limit_refreshed_each_guard=True,
            ram_policy_override_active=True,
            physical_reserve_bytes=PHYSICAL_RESERVE_BYTES,
            cgroup_discovery=discovery,
        )
        return allowed, report


def _write_admission(
    run: Path, setup: dict[str, Any], historical_budget_bytes: int
) -> dict[str, Any]:
    from operational.runpod_migration import runner

    migration_path = run / "MIGRATION_BINDING.json"
    if not migration_path.is_file():
        raise ValueError("Historical migration binding must exist before RAM policy admission")
    report = setup.get("resources", {})
    effective = report.get("runtime_effective_cgroup_limit_bytes")
    if not isinstance(effective, int) or effective <= PHYSICAL_RESERVE_BYTES:
        raise ValueError("Technical prepare did not use the real-cgroup RAM monitor")
    admission = {
        "schema": "ejepa_runpod_ram_policy_admission_v1",
        "status": "PASS",
        "policy_source_sha256": digest(Path(__file__)),
        "resource_v3_source_sha256": digest(Path(str(resources.__file__))),
        "historical_migration_binding_sha256": digest(migration_path),
        "historical_migration_budget_bytes": historical_budget_bytes,
        "runtime_effective_cgroup_limit_bytes": effective,
        "effective_limit_source": report.get("runtime_limit_source"),
        "user_authorized_real_kernel_limit": True,
        "physical_reserve_bytes": PHYSICAL_RESERVE_BYTES,
        "workspace_quota_bytes": resources.WORKSPACE_QUOTA_BYTES,
        "deadline_guard_preserved": True,
        "host_memavailable_guard_preserved": True,
        "strict_commit_guard_preserved": True,
        "disk_guard_preserved": True,
        "checkpoint_identity_extended": False,
        "scientific_identity_unchanged": True,
        "migration_binding_rewritten": False,
    }
    runner._write_once(run / "RAM_POLICY_ADMISSION.json", admission)
    return admission


def main() -> int:
    """Install the additive runtime policy before delegating to scientific v5."""
    original_monitor = resources.LinuxResourceMonitor
    original_prepare = v5.technical.prepare

    def prepare(bundle: Path, run: Path, budget_bytes: int) -> dict[str, Any]:
        setup = original_prepare(bundle, run, budget_bytes)
        _write_admission(run, setup, budget_bytes)
        return setup

    resources.LinuxResourceMonitor = RealCgroupLinuxResourceMonitor
    v5.technical.prepare = prepare
    try:
        return v5.main()
    finally:
        v5.technical.prepare = original_prepare
        resources.LinuxResourceMonitor = original_monitor


if __name__ == "__main__":
    raise SystemExit(main())
