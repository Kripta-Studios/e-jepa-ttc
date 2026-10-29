"""Move the waiting R2 queue to the V22 cloud-admission gate."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

OPS = Path("/workspace/ejepa_event_operations")
OLD_CFG = OPS / "EVENT_WATCHDOG_CONFIG_V11.json"
OLD_STATE = OPS / "EVENT_WATCHDOG_STATUS_V11.json"
CFG = OPS / "EVENT_WATCHDOG_CONFIG_V12.json"
STATE = OPS / "EVENT_WATCHDOG_STATUS_V12.json"
V21 = OPS / "event_downstream_runtime_v21"
V22 = OPS / "event_downstream_runtime_v22"
WAITER = OPS / "scripts/runpod_event_queue_waiter_v2.py"
BOOT = V22 / "scripts/runpod_event_queue_bootstrap_v6.py"
WD = OPS / "runpod_event_resource_watchdog_v8.py"


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


def identity(pid: int) -> dict[str, object]:
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
    c = json.loads(OLD_CFG.read_text())
    s = json.loads(OLD_STATE.read_text())
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
    os.kill(w, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{w}").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("watchdog did not exit")
    task = next(x for x in c["tasks"] if x["id"] == "R2_EVENT_DOWNSTREAM")
    old = int(s["tasks"]["R2_EVENT_DOWNSTREAM"]["pid"])
    seen = identity(old)
    if (
        seen["start_ticks"] != s["tasks"]["R2_EVENT_DOWNSTREAM"]["start_ticks"]
        or seen["cmdline_sha256"] != s["tasks"]["R2_EVENT_DOWNSTREAM"]["cmdline_sha256"]
    ):
        raise ValueError("R2 identity differs")
    os.kill(old, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{old}").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("R2 queue did not exit")

    def repl(a: list[str]) -> list[str]:
        return [
            x.replace(str(OPS / "scripts/runpod_event_queue_waiter_v1.py"), str(WAITER)).replace(
                str(V21 / "scripts/runpod_event_queue_bootstrap_v5.py"), str(BOOT)
            )
            for x in a
        ]

    task["argv"] = repl(task["argv"])
    task["adopted_argv"] = [repl(x) for x in task["adopted_argv"]]
    task["entrypoint"] = str(WAITER)
    task["entrypoint_sha256"] = sha(WAITER)
    files = []
    for item in task["runtime_files"]:
        p = Path(item["path"])
        if p.name == "runpod_event_queue_waiter_v1.py":
            p = WAITER
        elif p.name == "runpod_event_queue_bootstrap_v5.py":
            p = BOOT
        elif str(p).startswith(str(V21)):
            p = Path(str(p).replace(str(V21), str(V22), 1))
        files.append({"path": str(p), "sha256": sha(p)})
    task["runtime_files"] = files
    p = subprocess.Popen(
        task["argv"],
        stdin=subprocess.DEVNULL,
        stdout=Path(task["log"]).open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    time.sleep(3)
    new = identity(p.pid)
    task.update(
        pid=new["pid"],
        pid_start_ticks=new["start_ticks"],
        pid_started_epoch=new["started_epoch"],
        pid_cmdline_sha256=new["cmdline_sha256"],
    )
    s["tasks"]["R2_EVENT_DOWNSTREAM"].update(
        new, state="RUNNING", reason="CLOUD_ADMISSION_GATED_R2_QUEUE_V22"
    )
    c["status_path"] = str(STATE)
    c["lock_path"] = str(OPS / ".EVENT_WATCHDOG_V12.lock")
    atom(CFG, c)
    s["config_sha256"] = sha(CFG)
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
        OPS / "EVENT_R2_CLOUD_GATE_V12_DEPLOYMENT.json",
        {
            "schema": "ejepa_event_r2_cloud_gate_deployment_v1",
            "status": "RUNNING",
            "queue_pid": p.pid,
            "watchdog_pid": wd.pid,
            "optimizer_updates_reset": False,
        },
    )


if __name__ == "__main__":
    main()
