"""Single off-critical-path logical-volume publisher for all event workers."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from operational.rgb_port.accounting import atomic_write_json

from .event_resources import EVENT_VOLUME_BYTES

RETRY_DELAYS_SECONDS = (0.1, 0.25, 0.5)
_DU_ENOENT = re.compile(
    r"^du: cannot access [\'\"](?P<path>/workspace(?:/[^\'\"]*)?)[\'\"]: "
    r"No such file or directory$"
)


class TransientInventoryRaceError(RuntimeError):
    """All bounded retries observed only a transient /workspace ENOENT race."""


def _transient_workspace_enoent(error: subprocess.CalledProcessError) -> bool:
    if error.returncode != 1 or not isinstance(error.stderr, str):
        return False
    lines = [line.strip() for line in error.stderr.splitlines() if line.strip()]
    return bool(lines) and all(_DU_ENOENT.fullmatch(line) is not None for line in lines)


def _measure(volume: Path) -> int:
    for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
        try:
            measured = subprocess.run(
                ["du", "-sb", "--one-file-system", str(volume)],
                check=True,
                capture_output=True,
                text=True,
                timeout=90,
            )
            return int(measured.stdout.split()[0])
        except subprocess.CalledProcessError as error:
            if not _transient_workspace_enoent(error):
                raise
            if attempt == len(RETRY_DELAYS_SECONDS):
                raise TransientInventoryRaceError(
                    "Workspace inventory remained unstable after bounded ENOENT retries"
                ) from error
            time.sleep(RETRY_DELAYS_SECONDS[attempt])
    raise AssertionError("Unreachable inventory retry state")


def publish_once(volume: Path, run: Path) -> dict[str, object]:
    volume, run = volume.resolve(strict=True), run.resolve()
    if volume != Path("/workspace") or not run.is_relative_to(volume):
        raise ValueError("Event quota publisher is restricted to /workspace")
    used = _measure(volume)
    if not 0 <= used <= EVENT_VOLUME_BYTES:
        raise ValueError("Event volume logical usage exceeds its declared allocation")
    value: dict[str, object] = {
        "schema": "ejepa_event_volume_snapshot_v1",
        "run_root": str(run),
        "volume_root": str(volume),
        "quota_bytes": EVENT_VOLUME_BYTES,
        "used_bytes": used,
        "observed_timestamp": time.time(),
        "publisher_pid": os.getpid(),
        "recursive_scan_is_off_training_critical_path": True,
    }
    run.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run / "EVENT_VOLUME_SNAPSHOT.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, default=Path("/workspace"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 10:
        raise ValueError("Quota publication interval must be at least ten seconds")
    lock = args.run.parent / f".{args.run.name}.EVENT_VOLUME_PUBLISHER.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
        while True:
            try:
                value = publish_once(args.volume, args.run)
            except TransientInventoryRaceError:
                if args.once:
                    raise
                time.sleep(10)
                continue
            print(json.dumps(value), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval_seconds)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
