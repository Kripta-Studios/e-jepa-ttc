"""Retry only transient GNU-du ENOENT races before delegating to C2F v1.

Snapshot retention can remove a file after ``du`` enumerates it.  GNU du then
returns status 1 even though the workspace remains readable.  This operational
overlay retries that one narrowly identified race for a bounded duration.  It
does not accept stale inventory and does not weaken quota, TTL or disk reserves.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__:
    from . import runpod_resources_v3 as resources
else:
    import runpod_resources_v3 as resources

from operational.runpod_migration.inventory import digest

RETRY_DELAYS_SECONDS = (0.1, 0.25, 0.5)
_DU_ENOENT = re.compile(
    r"^du: cannot access [\'\"](?P<path>/workspace(?:/[^\'\"]*)?)[\'\"]: "
    r"No such file or directory$"
)


def _delegate() -> tuple[Any, Any]:
    """Consume an operational selector and return the delegate plus its technical module."""
    selected = "c2f"
    if "--quota-entrypoint" in sys.argv:
        if sys.argv.count("--quota-entrypoint") != 1:
            raise ValueError("Exactly one --quota-entrypoint is allowed")
        index = sys.argv.index("--quota-entrypoint")
        if index + 1 >= len(sys.argv):
            raise ValueError("Missing --quota-entrypoint value")
        selected = sys.argv[index + 1]
        del sys.argv[index : index + 2]
    if selected == "c2f":
        if __package__:
            from . import runpod_scientific_overlay_c2f_v1 as delegate
        else:
            import runpod_scientific_overlay_c2f_v1 as delegate
        return delegate, delegate.technical
    if selected == "ram-a5":
        if __package__:
            from . import runpod_ram_policy_overlay_v1 as delegate
        else:
            import runpod_ram_policy_overlay_v1 as delegate
        return delegate, delegate.v5.technical
    raise ValueError("--quota-entrypoint must be c2f or ram-a5")


def transient_workspace_enoent(error: subprocess.CalledProcessError) -> bool:
    """Accept only status-1 GNU-du ENOENT diagnostics wholly under /workspace."""
    if error.returncode != 1 or not isinstance(error.stderr, str):
        return False
    lines = [line.strip() for line in error.stderr.splitlines() if line.strip()]
    return bool(lines) and all(_DU_ENOENT.fullmatch(line) is not None for line in lines)


class RetryingStorageInventory:
    """Retry a transient traversal race, then fail closed without stale fallback."""

    def __init__(
        self,
        original: Callable[[Path], tuple[int, dict[str, Any]]],
        *,
        delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if any(delay < 0 for delay in delays):
            raise ValueError("Inventory retry delays must be nonnegative")
        self.original = original
        self.delays = delays
        self.sleeper = sleeper

    def __call__(self, root: Path) -> tuple[int, dict[str, Any]]:
        races = 0
        for attempt in range(len(self.delays) + 1):
            try:
                free, report = self.original(root)
                return free, {
                    **report,
                    "workspace_inventory_enoent_retries": races,
                    "workspace_inventory_retry_bounded": True,
                    "workspace_inventory_stale_fallback_used": False,
                }
            except subprocess.CalledProcessError as error:
                if not transient_workspace_enoent(error):
                    raise
                races += 1
                if attempt == len(self.delays):
                    raise RuntimeError(
                        "Workspace inventory remained unstable after bounded ENOENT retries"
                    ) from error
                self.sleeper(self.delays[attempt])
        raise AssertionError("unreachable")


def main() -> int:
    """Install retry policy before C2F captures its quota inventory callable."""
    delegate, technical = _delegate()
    original_storage = resources.storage_available
    original_prepare = technical.prepare
    retrying = RetryingStorageInventory(original_storage)

    def prepare(bundle: Path, run: Path, budget_bytes: int) -> dict[str, Any]:
        setup = original_prepare(bundle, run, budget_bytes)
        from operational.runpod_migration import runner

        migration = run / "MIGRATION_BINDING.json"
        admission = {
            "schema": "ejepa_workspace_inventory_retry_admission_v1",
            "status": "PASS",
            "policy_source_sha256": digest(Path(__file__)),
            "resource_v3_source_sha256": digest(Path(str(resources.__file__))),
            "migration_binding_sha256": digest(migration),
            "retry_delays_seconds": list(RETRY_DELAYS_SECONDS),
            "retry_scope": "GNU du status 1; every stderr line is ENOENT below /workspace",
            "generic_exit_1_ignored": False,
            "stale_inventory_fallback": False,
            "quota_ttl_changed": False,
            "quota_inflight_reserve_changed": False,
            "workspace_quota_changed": False,
            "scientific_identity_changed": False,
            "migration_binding_rewritten": False,
        }
        runner._write_once(run / "QUOTA_RACE_POLICY_ADMISSION.json", admission)
        return setup

    resources.storage_available = retrying
    technical.prepare = prepare
    try:
        return delegate.main()
    finally:
        technical.prepare = original_prepare
        resources.storage_available = original_storage


if __name__ == "__main__":
    raise SystemExit(main())
