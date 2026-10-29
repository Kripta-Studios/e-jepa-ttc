"""Adopt every live producer into the final V12 watchdog state."""

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
CFG = OPS / "EVENT_WATCHDOG_CONFIG_V12.json"
STATE = OPS / "EVENT_WATCHDOG_STATUS_V12.json"
WD = OPS / "runpod_event_resource_watchdog_v8.py"


def atom(p: Path, v: object) -> None:
    t = p.with_name(f".{p.name}.{os.getpid()}.pending")
    with t.open("x") as f:
        json.dump(v, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)


def ident(pid: int) -> dict[str, object]:
    a = [x for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]
    s = Path(f"/proc/{pid}/stat").read_text().split()
    ticks = int(s[21])
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
        "cmdline_sha256": hashlib.sha256(b"\0".join(a)).hexdigest(),
    }


def main() -> None:
    c = json.loads(CFG.read_text())
    s = json.loads(STATE.read_text())
    env = os.environ.copy()
    env.update(c["required_environment"])
    env.update(
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
    w = int((OPS / "EVENT_WATCHDOG_V12.pid").read_text())
    os.kill(w, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{w}").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("watchdog did not exit")
    adopted = {}
    for task in c["tasks"]:
        if task["fit"] not in {"E_A5", "E_C2F"} or task["id"] == "R2_E_A5":
            continue
        fit = RUN / f"rotations/{task['rotation']}/fits/{task['fit']}"
        pid = json.loads((fit / "PROGRESS.json").read_text()).get("pid")
        seen = ident(pid) if isinstance(pid, int) and Path(f"/proc/{pid}/stat").is_file() else None
        if seen is None or seen["cmdline_sha256"] != task["pid_cmdline_sha256"]:
            r = json.loads((fit / "CHECKPOINT_RECEIPT.json").read_text())
            j = json.loads((fit / "UPDATE_JOURNAL.json").read_text())
            if not (
                r["completed_updates"] == j["completed_updates"] == j["durable_updates"]
                and j["pending_update_upper"] == j["recovery_upper"] == 0
            ):
                raise ValueError(task["id"])
            p = subprocess.Popen(
                task["argv"],
                stdin=subprocess.DEVNULL,
                stdout=Path(task["log"]).open("ab", buffering=0),
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
            time.sleep(2)
            seen = ident(p.pid)
        if seen["cmdline_sha256"] != task["pid_cmdline_sha256"]:
            raise ValueError(f"{task['id']} argv")
        s["tasks"][task["id"]].update(
            seen, state="RUNNING", reason="FINAL_V12_LIVE_PROCESS_RECONCILIATION"
        )
        adopted[task["id"]] = seen
    atom(STATE, s)
    cmd = ["/opt/ejepa_env/bin/python", "-u", str(WD), "--config", str(CFG)]
    wd = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V12.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (OPS / "EVENT_WATCHDOG_V12.pid").write_text(f"{wd.pid}\n")
    time.sleep(4)
    if wd.poll() is not None:
        raise RuntimeError("watchdog failed")
    atom(
        OPS / "EVENT_V12_FINAL_RECONCILIATION.json",
        {
            "schema": "ejepa_event_v12_final_reconciliation_v1",
            "status": "RUNNING",
            "tasks": adopted,
            "watchdog_pid": wd.pid,
            "operator_stop_requests_remaining": 0,
            "optimizer_updates_reset": False,
        },
    )


if __name__ == "__main__":
    main()
