"""Deploy watchdog V6 while preserving the live V7 task state."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

OPS = Path("/workspace/ejepa_event_operations")
OLD_CONFIG = OPS / "EVENT_WATCHDOG_CONFIG_V7.json"
OLD_STATUS = OPS / "EVENT_WATCHDOG_STATUS_V7.json"
NEW_CONFIG = OPS / "EVENT_WATCHDOG_CONFIG_V8.json"
NEW_STATUS = OPS / "EVENT_WATCHDOG_STATUS_V8.json"
SOURCE = OPS / "runpod_event_resource_watchdog_v6.py"
SOURCE_SHA = "576b38484ae5f63932cf85d07fa0f9049f717cf572b08867c34f4425c66626f9"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path: Path, value: object) -> None:
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def argv(pid: int) -> list[str]:
    return [x.decode() for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]


def main() -> None:
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Watchdog V6 source differs")
    config = json.loads(OLD_CONFIG.read_text())
    status = json.loads(OLD_STATUS.read_text())
    matches = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            args = argv(int(item.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if str(OLD_CONFIG) in args and "runpod_event_resource_watchdog_v5.py" in " ".join(args):
            matches.append(int(item.name))
    if len(matches) > 1:
        raise RuntimeError(f"Expected at most one V7 watchdog, got {matches}")
    if matches:
        config["status_path"] = str(NEW_STATUS)
        config["lock_path"] = str(OPS / ".EVENT_WATCHDOG_V8.lock")
        config["watchdog_exec_strict_overlay_source_sha256"] = SOURCE_SHA
        atomic(NEW_CONFIG, config)
        status["config_sha256"] = sha(NEW_CONFIG)
        status["source_sha256"] = SOURCE_SHA
        atomic(NEW_STATUS, status)
        os.kill(matches[0], signal.SIGTERM)
        for _ in range(100):
            if not Path(f"/proc/{matches[0]}").exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("V7 watchdog did not exit")
    elif NEW_CONFIG.is_file() and NEW_STATUS.is_file():
        config = json.loads(NEW_CONFIG.read_text())
    else:
        raise RuntimeError("Neither the V7 watchdog nor complete V8 state exists")
    log = (OPS / "EVENT_WATCHDOG_V8.log").open("ab", buffering=0)
    environment = os.environ.copy()
    environment.update(config["required_environment"])
    proc = subprocess.Popen(
        ["/opt/ejepa_env/bin/python", "-u", str(SOURCE), "--config", str(NEW_CONFIG)],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
    time.sleep(3)
    if proc.poll() is not None:
        raise RuntimeError(f"V8 watchdog exited {proc.returncode}")
    atomic(
        OPS / "EVENT_WATCHDOG_V8_DEPLOYMENT.json",
        {
            "schema": "ejepa_event_watchdog_deployment_v1",
            "status": "RUNNING",
            "pid": proc.pid,
            "source_sha256": SOURCE_SHA,
            "config_sha256": sha(NEW_CONFIG),
            "previous_pid": matches[0] if matches else None,
            "optimizer_updates_performed": 0,
        },
    )


if __name__ == "__main__":
    main()
