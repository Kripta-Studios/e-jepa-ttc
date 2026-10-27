"""Resume both durable C2F handoffs after the verified R2 PAIR_E seal."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

OPS = Path("/workspace/ejepa_event_operations")
RUN = Path("/workspace/ejepa_event_runs")
CONFIG = OPS / "EVENT_WATCHDOG_CONFIG_V9.json"
STATUS = OPS / "EVENT_WATCHDOG_STATUS_V9.json"
WATCHDOG = OPS / "runpod_event_resource_watchdog_v7.py"


def atomic(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def identity(pid: int) -> dict[str, object]:
    parts = [x for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    ticks = int(fields[21])
    boot = int(
        next(
            x.split()[1]
            for x in Path("/proc/stat").read_text().splitlines()
            if x.startswith("btime ")
        )
    )
    return {
        "pid": pid,
        "start_ticks": ticks,
        "started_epoch": boot + ticks / os.sysconf("SC_CLK_TCK"),
        "cmdline_sha256": hashlib.sha256(b"\0".join(parts)).hexdigest(),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    status = json.loads(STATUS.read_text())
    cache = RUN / "rotations/2/caches/PAIR_E.npz"
    if not (cache.is_file() and cache.with_suffix(".json").is_file()):
        raise ValueError("PAIR_E seal absent")
    w = int((OPS / "EVENT_WATCHDOG_V9.pid").read_text())
    expected = ["/opt/ejepa_env/bin/python", "-u", str(WATCHDOG), "--config", str(CONFIG)]
    if Path(f"/proc/{w}/cmdline").read_bytes().split(b"\0")[:-1] != [x.encode() for x in expected]:
        raise ValueError("watchdog differs")
    os.kill(w, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{w}").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("watchdog did not exit")
    environment = os.environ.copy()
    environment.update(config["required_environment"])
    environment.update(
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
    observed = {}
    for task_id in ("R3_E_C2F", "R4_E_C2F"):
        task = next(x for x in config["tasks"] if x["id"] == task_id)
        fit = RUN / f"rotations/{task['rotation']}/fits/E_C2F"
        receipt = json.loads((fit / "CHECKPOINT_RECEIPT.json").read_text())
        journal = json.loads((fit / "UPDATE_JOURNAL.json").read_text())
        if not (
            receipt["status"] == "PAUSED_RESOURCE"
            and receipt["completed_updates"]
            == journal["completed_updates"]
            == journal["durable_updates"]
            and journal["pending_update_upper"] == journal["recovery_upper"] == 0
        ):
            raise ValueError(f"{task_id} durable boundary differs")
        proc = subprocess.Popen(
            task["argv"],
            stdin=subprocess.DEVNULL,
            stdout=Path(task["log"]).open("ab", buffering=0),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        )
        time.sleep(2)
        item = identity(proc.pid)
        if item["cmdline_sha256"] != task["pid_cmdline_sha256"]:
            raise ValueError(f"{task_id} argv differs")
        status["tasks"][task_id].update(
            item, state="RUNNING", reason="PAIR_E_CACHE_SEALED_DURABLE_HANDOFF_RESUME"
        )
        observed[task_id] = {**item, "durable_updates": receipt["completed_updates"]}
    atomic(STATUS, status)
    wd = subprocess.Popen(
        expected,
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V9.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
    (OPS / "EVENT_WATCHDOG_V9.pid").write_text(f"{wd.pid}\n")
    time.sleep(4)
    if wd.poll() is not None:
        raise RuntimeError("watchdog relaunch failed")
    atomic(
        OPS / "PAIR_E_CACHE_HANDOFF_RESUME.json",
        {
            "schema": "ejepa_event_pair_cache_handoff_resume_v1",
            "status": "RUNNING",
            "tasks": observed,
            "watchdog_pid": wd.pid,
            "optimizer_updates_reset": False,
        },
    )


if __name__ == "__main__":
    main()
