"""Recover R3 E_A5 after the operator watchdog environment deployment fault."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

OPS = Path("/workspace/ejepa_event_operations")
STATUS = OPS / "EVENT_WATCHDOG_STATUS_V8.json"
CONFIG = OPS / "EVENT_WATCHDOG_CONFIG_V8.json"
WATCHDOG = OPS / "runpod_event_resource_watchdog_v6.py"


def atomic(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def identity(pid: int) -> dict[str, object]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    arguments = [item for item in raw.split(b"\0") if item]
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    boot = int(
        next(
            x.split()[1]
            for x in Path("/proc/stat").read_text().splitlines()
            if x.startswith("btime ")
        )
    )
    ticks = int(fields[21])
    return {
        "pid": pid,
        "start_ticks": ticks,
        "started_epoch": boot + ticks / os.sysconf("SC_CLK_TCK"),
        "cmdline_sha256": hashlib.sha256(b"\0".join(arguments)).hexdigest(),
    }


def env(config: dict[str, object]) -> dict[str, str]:
    value = os.environ.copy()
    value.update(config["required_environment"])
    value.update(
        {
            "EJEPA_EVENT_SUPPLEMENT_ROOT": "/opt/ejepa_event_supplement",
            "EJEPA_EVENT_SUPPLEMENT_ADMISSION": str(
                OPS / "supplement/EVENT_DOWNSTREAM_SUPPLEMENTAL_ADMISSION.json"
            ),
            "EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ROOT": "/opt/ejepa_event_authority_supplement_v2",
            "EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ADMISSION": str(
                OPS / "authority_supplement_v2/EVENT_AUTHORITY_SUPPLEMENT_ADMISSION.json"
            ),
        }
    )
    return value


def main() -> None:
    config = json.loads(CONFIG.read_text())
    status = json.loads(STATUS.read_text())
    w = int((OPS / "EVENT_WATCHDOG_V8.pid").read_text())
    expected = ["/opt/ejepa_env/bin/python", "-u", str(WATCHDOG), "--config", str(CONFIG)]
    if Path(f"/proc/{w}/cmdline").is_file():
        if Path(f"/proc/{w}/cmdline").read_bytes().split(b"\0")[:-1] != [
            x.encode() for x in expected
        ]:
            raise ValueError("watchdog identity differs")
        os.kill(w, signal.SIGTERM)
        for _ in range(100):
            if not Path(f"/proc/{w}").exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("watchdog did not exit")
    task = next(x for x in config["tasks"] if x["id"] == "R3_E_A5")
    existing = int(status["tasks"]["R3_E_A5"]["pid"])
    if Path(f"/proc/{existing}/stat").is_file():
        observed = identity(existing)
    else:
        log = Path(task["log"]).open("ab", buffering=0)
        proc = subprocess.Popen(
            task["argv"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env(config),
        )
        time.sleep(2)
        observed = identity(proc.pid)
    if (
        observed["cmdline_sha256"]
        != hashlib.sha256(b"\0".join(x.encode() for x in task["argv"])).hexdigest()
    ):
        raise ValueError("producer argv differs")
    status["tasks"]["R3_E_A5"].update(
        observed, state="RUNNING", reason="OPERATOR_VERIFIED_DURABLE_ENV_RECOVERY"
    )
    atomic(STATUS, status)
    wd = subprocess.Popen(
        expected,
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V8.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env(config),
    )
    (OPS / "EVENT_WATCHDOG_V8.pid").write_text(f"{wd.pid}\n")
    time.sleep(3)
    if wd.poll() is not None:
        raise RuntimeError("watchdog relaunch failed")
    atomic(
        OPS / "R3_E_A5_ENV_RECOVERY.json",
        {
            "schema": "ejepa_event_operator_recovery_v1",
            "status": "RUNNING",
            "producer": observed,
            "watchdog_pid": wd.pid,
            "durable_updates": 26683,
            "optimizer_updates_reset": False,
        },
    )


if __name__ == "__main__":
    main()
