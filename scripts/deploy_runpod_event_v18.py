"""Move the still-pre-head R4 queue onto admitted V25 routing."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

OPS = Path("/workspace/ejepa_event_operations")
OLD_CONFIG = OPS / "EVENT_WATCHDOG_CONFIG_V17.json"
OLD_STATE = OPS / "EVENT_WATCHDOG_STATUS_V17.json"
CONFIG = OPS / "EVENT_WATCHDOG_CONFIG_V18.json"
STATE = OPS / "EVENT_WATCHDOG_STATUS_V18.json"
WD = OPS / "runpod_event_resource_watchdog_v10.py"
WD_SHA = "3015df79d8220041328382133447e30c4053e2646f74eff2b40f2f4c08bc03ee"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atom(path: Path, value: object) -> None:
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def identity(pid: int) -> tuple[dict[str, object], list[str]]:
    argv = [
        part.decode() for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part
    ]
    stat = Path(f"/proc/{pid}/stat").read_text().split()
    ticks = int(stat[21])
    boot = int(
        next(
            line.split()[1]
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
    )
    value = {
        "pid": pid,
        "start_ticks": ticks,
        "started_epoch": boot + ticks / os.sysconf("SC_CLK_TCK"),
        "cmdline_sha256": hashlib.sha256(b"\0".join(x.encode() for x in argv)).hexdigest(),
    }
    return value, argv


def wait_exit(pid: int, label: str) -> None:
    for _ in range(100):
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.1)
    raise RuntimeError(f"{label} did not exit")


def main() -> None:
    if sha(WD) != WD_SHA:
        raise ValueError("watchdog source differs")
    watchdog_pid = int((OPS / "EVENT_WATCHDOG_V17.pid").read_text())
    os.kill(watchdog_pid, signal.SIGTERM)
    wait_exit(watchdog_pid, "V17 watchdog")

    # Re-read only after the old supervisor has stopped.
    config = json.loads(OLD_CONFIG.read_text())
    state = json.loads(OLD_STATE.read_text())
    task = next(item for item in config["tasks"] if item["id"] == "R4_EVENT_DOWNSTREAM")
    current = state["tasks"]["R4_EVENT_DOWNSTREAM"]
    old_pid = int(current["pid"])
    seen, argv = identity(old_pid)
    if seen["start_ticks"] != current["start_ticks"] or not any(
        "event_downstream_runtime_v22/scripts/runpod_event_queue_bootstrap_v6.py" in item
        for item in argv
    ):
        raise ValueError("R4 old queue identity differs")
    os.kill(old_pid, signal.SIGTERM)
    wait_exit(old_pid, "R4 V22 queue")

    env = os.environ.copy()
    env.update(config["required_environment"])
    child = subprocess.Popen(
        task["argv"],
        stdin=subprocess.DEVNULL,
        stdout=Path(task["log"]).open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    time.sleep(6)
    if child.poll() is not None:
        raise RuntimeError("R4 V25 queue failed during launch")
    replacement, actual = identity(child.pid)
    if actual not in task["adopted_argv"]:
        raise ValueError("R4 V25 argv is not admitted")
    task.update(
        pid=replacement["pid"],
        pid_start_ticks=replacement["start_ticks"],
        pid_started_epoch=replacement["started_epoch"],
        pid_cmdline_sha256=replacement["cmdline_sha256"],
    )
    current.update(
        replacement,
        state="RUNNING",
        reason="R4_V25_FULL_RAW_ROOT_RELOCATION",
    )
    config.update(status_path=str(STATE), lock_path=str(OPS / ".EVENT_WATCHDOG_V18.lock"))
    atom(CONFIG, config)
    state.update(config_sha256=sha(CONFIG), source_sha256=WD_SHA)
    atom(STATE, state)

    watchdog = subprocess.Popen(
        ["/opt/ejepa_env/bin/python", "-u", str(WD), "--config", str(CONFIG)],
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V18.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (OPS / "EVENT_WATCHDOG_V18.pid").write_text(f"{watchdog.pid}\n")
    time.sleep(5)
    if watchdog.poll() is not None:
        raise RuntimeError("V18 watchdog failed")
    atom(
        OPS / "EVENT_V18_DEPLOYMENT.json",
        {
            "schema": "ejepa_event_v18_deployment_v1",
            "status": "RUNNING",
            "watchdog_pid": watchdog.pid,
            "watchdog_source_sha256": WD_SHA,
            "config_sha256": sha(CONFIG),
            "r4_previous_pid": old_pid,
            "r4_replacement": replacement,
            "optimizer_updates_reset": False,
        },
    )


if __name__ == "__main__":
    main()
