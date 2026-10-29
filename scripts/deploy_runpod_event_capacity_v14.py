"""Deploy the final four-producer EVENT capacity policy from a stopped snapshot."""

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
OLD_CFG = OPS / "EVENT_WATCHDOG_CONFIG_V13.json"
OLD_STATE = OPS / "EVENT_WATCHDOG_STATUS_V13.json"
CFG = OPS / "EVENT_WATCHDOG_CONFIG_V14.json"
STATE = OPS / "EVENT_WATCHDOG_STATUS_V14.json"
WD = OPS / "runpod_event_resource_watchdog_v9.py"
WD_SHA = "5e04f4b50e633b1d8df995307a7d168e932ceaaab956fc8a122e60458d03f53d"


def sha(p: Path) -> str:
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


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
    if sha(WD) != WD_SHA:
        raise ValueError("watchdog V9 differs")
    oldpid = int((OPS / "EVENT_WATCHDOG_V13.pid").read_text())
    os.kill(oldpid, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{oldpid}").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("V13 watchdog did not exit")
    c = json.loads(OLD_CFG.read_text())
    s = json.loads(OLD_STATE.read_text())
    tasks = {x["id"]: x for x in c["tasks"]}
    for tid in ("R2_E_C2F", "R3_E_C2F", "R4_E_A5", "R4_E_C2F"):
        task = tasks[tid]
        progress = json.loads(
            (RUN / f"rotations/{task['rotation']}/fits/{task['fit']}/PROGRESS.json").read_text()
        )
        pid = progress["pid"]
        seen = ident(pid)
        if seen["cmdline_sha256"] != task["pid_cmdline_sha256"]:
            raise ValueError(f"{tid} live identity differs")
        s["tasks"][tid].update(seen, state="RUNNING", reason="CAPACITY_V14_LIVE_RECONCILIATION")
    fit = RUN / "rotations/3/fits/E_A5"
    r = json.loads((fit / "CHECKPOINT_RECEIPT.json").read_text())
    j = json.loads((fit / "UPDATE_JOURNAL.json").read_text())
    p = json.loads((fit / "PROGRESS.json").read_text())
    if not (
        r["status"] == "PAUSED_RESOURCE"
        and r["completed_updates"]
        == j["completed_updates"]
        == j["durable_updates"]
        == p["completed_updates"]
        == 28021
        and j["pending_update_upper"] == j["recovery_upper"] == 0
        and p["pid"] == 238376
    ):
        raise ValueError("R3_E_A5 deferred boundary differs")
    s["tasks"]["R3_E_A5"].update(
        pid=238376,
        state="WAITING_CAPACITY",
        reason="CAPACITY_WAIT",
        retries=0,
        retry_anchor_completed_updates=28021,
    )
    c.update(
        status_path=str(STATE),
        lock_path=str(OPS / ".EVENT_WATCHDOG_V14.lock"),
        deferred_task="R3_E_A5",
        release_candidates=["R4_E_A5", "R2_E_C2F", "R3_E_C2F", "R4_E_C2F"],
        capacity_max_active=4,
        watchdog_capacity_overlay_source_sha256=WD_SHA,
    )
    atom(CFG, c)
    s["config_sha256"] = sha(CFG)
    s["source_sha256"] = WD_SHA
    atom(STATE, s)
    env = os.environ.copy()
    env.update(c["required_environment"])
    cmd = ["/opt/ejepa_env/bin/python", "-u", str(WD), "--config", str(CFG)]
    wd = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V14.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (OPS / "EVENT_WATCHDOG_V14.pid").write_text(f"{wd.pid}\n")
    time.sleep(5)
    if wd.poll() is not None:
        raise RuntimeError("V14 watchdog failed")
    atom(
        OPS / "EVENT_CAPACITY_V14_DEPLOYMENT.json",
        {
            "schema": "ejepa_event_capacity_deployment_v1",
            "status": "RUNNING",
            "capacity_max_active": 4,
            "deferred_task": "R3_E_A5",
            "deferred_durable_updates": 28021,
            "release_candidates": c["release_candidates"],
            "watchdog_pid": wd.pid,
            "watchdog_source_sha256": WD_SHA,
            "config_sha256": sha(CFG),
            "optimizer_updates_reset": False,
        },
    )


if __name__ == "__main__":
    main()
