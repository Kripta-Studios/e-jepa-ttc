"""Repair A5 watchdog state after V11 rollover."""

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
CFG = OPS / "EVENT_WATCHDOG_CONFIG_V11.json"
STATE = OPS / "EVENT_WATCHDOG_STATUS_V11.json"
WD = OPS / "runpod_event_resource_watchdog_v8.py"


def atom(p: Path, v: object) -> None:
    t = p.with_name(f".{p.name}.{os.getpid()}.pending")
    with t.open("x") as f:
        json.dump(v, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)


def identity(pid: int) -> dict[str, object]:
    args = [x for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]
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
        "cmdline_sha256": hashlib.sha256(b"\0".join(args)).hexdigest(),
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
    w = int((OPS / "EVENT_WATCHDOG_V11.pid").read_text())
    if Path(f"/proc/{w}").exists():
        os.kill(w, signal.SIGTERM)
        for _ in range(100):
            if not Path(f"/proc/{w}").exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("watchdog did not exit")
    for tid in ("R3_E_A5", "R4_E_A5"):
        task = next(x for x in c["tasks"] if x["id"] == tid)
        fit = RUN / f"rotations/{task['rotation']}/fits/E_A5"
        pid = json.loads((fit / "PROGRESS.json").read_text()).get("pid")
        seen = (
            identity(pid) if isinstance(pid, int) and Path(f"/proc/{pid}/stat").is_file() else None
        )
        if seen is None or seen["cmdline_sha256"] != task["pid_cmdline_sha256"]:
            r = json.loads((fit / "CHECKPOINT_RECEIPT.json").read_text())
            j = json.loads((fit / "UPDATE_JOURNAL.json").read_text())
            if not (
                r["completed_updates"] == j["completed_updates"] == j["durable_updates"]
                and j["pending_update_upper"] == j["recovery_upper"] == 0
            ):
                raise ValueError(tid)
            p = subprocess.Popen(
                task["argv"],
                stdin=subprocess.DEVNULL,
                stdout=Path(task["log"]).open("ab", buffering=0),
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
            time.sleep(2)
            seen = identity(p.pid)
        if seen["cmdline_sha256"] != task["pid_cmdline_sha256"]:
            raise ValueError(f"{tid} argv")
        s["tasks"][tid].update(seen, state="RUNNING", reason="DURABLE_A5_V11_STATE_REPAIR")
    atom(STATE, s)
    cmd = ["/opt/ejepa_env/bin/python", "-u", str(WD), "--config", str(CFG)]
    wd = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V11.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (OPS / "EVENT_WATCHDOG_V11.pid").write_text(f"{wd.pid}\n")
    time.sleep(4)
    if wd.poll() is not None:
        raise RuntimeError("watchdog failed")
    atom(
        OPS / "A5_V11_STATE_REPAIR.json",
        {
            "schema": "ejepa_event_a5_state_repair_v1",
            "status": "RUNNING",
            "watchdog_pid": wd.pid,
            "optimizer_updates_reset": False,
        },
    )


if __name__ == "__main__":
    main()
