"""Bounded retries at data-read boundaries and immutable overlay provenance."""

from __future__ import annotations

import errno
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port.recipe import canonical_sha256

ROOT = Path(__file__).resolve().parents[2]
FREEZE_NAME = "IO_RECOVERY_FREEZE.json"
RETRY_COUNT = 3
RETRY_DELAY_SECONDS = 2.0
DISK_WINERRORS = {21, 55, 1117, 1167}
DISK_ERRNOS = {errno.EINVAL, errno.EIO, errno.ENOTCONN, errno.ENODEV, errno.ENOENT}
T = TypeVar("T")


def retryable_read_error(error: OSError) -> bool:
    """Classify only errors already scoped to input reads or supervisor disk I/O."""
    return error.errno in DISK_ERRNOS or getattr(error, "winerror", None) in DISK_WINERRORS


def retry_input_read(
    read: Callable[[], T],
    record: Callable[[dict[str, Any]], None],
    *,
    scope: str,
    attempts: int = RETRY_COUNT,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Reopen the original input, or trigger the trainer's full-state pause path.

    No data substitution, sample skipping, corruption suppression or optimizer
    operations occur here. ValueError/BadZipFile/CRC failures still propagate.
    """
    if attempts < 1:
        raise ValueError("At least one read attempt is required")
    for attempt in range(1, attempts + 1):
        try:
            return read()
        except OSError as error:
            if not retryable_read_error(error):
                raise
            record(
                {
                    "scope": scope,
                    "attempt": attempt,
                    "maximum_attempts": attempts,
                    "errno": error.errno,
                    "winerror": getattr(error, "winerror", None),
                    "message": str(error),
                    "checked_utc": datetime.now(UTC).isoformat(),
                    "action": "retry_same_input" if attempt < attempts else "full_state_pause",
                }
            )
            if attempt == attempts:
                raise OSError(
                    errno.ENOTCONN, f"Input unavailable after bounded retries: {scope}"
                ) from error
            sleep(RETRY_DELAY_SECONDS)
    raise AssertionError("Unreachable retry state")


def freeze_payload(run: Path, qa: Path) -> dict[str, Any]:
    """Bind this additive overlay, its tests and the unchanged upstream contracts."""
    files = [
        *sorted(Path(__file__).parent.glob("*.py")),
        ROOT / "tests/test_rgb_port_io_recovery.py",
    ]
    value = {
        "schema": "rgb_port_io_recovery_freeze_v1",
        "scientific_changes": False,
        "optimizer_updates_for_qa": 0,
        "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in files},
        "qa": {"path": str(qa.resolve(strict=True)), "sha256": sha256_file(qa)},
        "upstream": {
            name: sha256_file(run / name)
            for name in (
                "SOURCE_FREEZE.json",
                "ACCELERATION_FREEZE.json",
                "PIPELINE_FREEZE.json",
                "CONCURRENT_FREEZE.json",
                "C2F_GRAPH_FREEZE.json",
                "AUDIT_CODE_MIGRATION.json",
            )
        },
        "retry_count": RETRY_COUNT,
        "retry_delay_seconds": RETRY_DELAY_SECONDS,
        "scope": ["event_shard_decode", "rgb_producer_batch", "supervisor_disk_io"],
    }
    value["identity_sha256"] = canonical_sha256(value)
    return value


def validate_freeze(run: Path) -> dict[str, Any]:
    """Reject unreviewed changes rather than bypassing historical freezes."""
    value = read_json_shared(run / FREEZE_NAME)
    identity = canonical_sha256({k: v for k, v in value.items() if k != "identity_sha256"})
    if (
        value.get("schema") != "rgb_port_io_recovery_freeze_v1"
        or value.get("identity_sha256") != identity
    ):
        raise ValueError("I/O recovery overlay identity changed")
    if value.get("scientific_changes") is not False or value.get("optimizer_updates_for_qa") != 0:
        raise ValueError("I/O overlay must not change science or consume training updates")
    for relative, expected in value["files"].items():
        path = (ROOT / relative).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha256_file(path) != expected:
            raise ValueError(f"Frozen I/O recovery source changed: {relative}")
    for name, expected in value["upstream"].items():
        if sha256_file(run / name) != expected:
            raise ValueError(f"Upstream contract changed: {name}")
    qa = value["qa"]
    if sha256_file(Path(qa["path"])) != qa["sha256"]:
        raise ValueError("I/O recovery QA changed")
    return value


def write_event(run: Path, fit_id: str, event: dict[str, Any]) -> None:
    """Keep one durable record per failure; fits never share a mutable journal."""
    root = run / "io_recovery" / fit_id
    atomic_write_json(root / f"read_{time.time_ns()}.json", event)
