"""Deploy watchdog V10 and V23 queue routing, restarting only dead R2 queue."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

OPS = Path("/workspace/ejepa_event_operations")
OLD_CFG = OPS / "EVENT_WATCHDOG_CONFIG_V14.json"
OLD_STATE = OPS / "EVENT_WATCHDOG_STATUS_V14.json"
CFG = OPS / "EVENT_WATCHDOG_CONFIG_V15.json"
STATE = OPS / "EVENT_WATCHDOG_STATUS_V15.json"
V22 = OPS / "event_downstream_runtime_v22"
V23 = OPS / "event_downstream_runtime_v23"
BOOT_SRC = OPS / "runpod_event_queue_bootstrap_v7.py"
BOOT = V23 / "scripts/runpod_event_queue_bootstrap_v7.py"
WD = OPS / "runpod_event_resource_watchdog_v10.py"
WD_SHA = "3015df79d8220041328382133447e30c4053e2646f74eff2b40f2f4c08bc03ee"


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
    if sha(WD) != WD_SHA:
        raise ValueError("watchdog V10 differs")
    w = int((OPS / "EVENT_WATCHDOG_V14.pid").read_text())
    os.kill(w, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{w}").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("V14 watchdog did not exit")
    if V23.exists():
        raise FileExistsError(V23)
    shutil.copytree(V22, V23, symlinks=True)
    shutil.copy2(BOOT_SRC, BOOT)
    c = json.loads(OLD_CFG.read_text())
    s = json.loads(OLD_STATE.read_text())
    env = os.environ.copy()
    env.update(c["required_environment"])
    for task in c["tasks"]:
        if task["fit"] != "EVENT_DOWNSTREAM":
            continue
        old_wait = list(task["argv"])
        new_wait = [
            x.replace(str(V22 / "scripts/runpod_event_queue_bootstrap_v6.py"), str(BOOT))
            for x in old_wait
        ]
        old_boot = task["adopted_argv"][-1]
        new_boot = [
            x.replace(str(V22 / "scripts/runpod_event_queue_bootstrap_v6.py"), str(BOOT))
            for x in old_boot
        ]
        task["argv"] = new_wait
        for candidate in (new_wait, new_boot):
            if candidate not in task["adopted_argv"]:
                task["adopted_argv"].append(candidate)
        task["runtime_files"].append({"path": str(BOOT), "sha256": sha(BOOT)})
        if task["id"] == "R2_EVENT_DOWNSTREAM":
            if Path(f"/proc/{s['tasks'][task['id']]['pid']}").exists():
                raise RuntimeError("R2 queue unexpectedly alive")
            p = subprocess.Popen(
                new_wait,
                stdin=subprocess.DEVNULL,
                stdout=Path(task["log"]).open("ab", buffering=0),
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
            time.sleep(3)
            seen = identity(p.pid)
            expected = hashlib.sha256(b"\0".join(x.encode() for x in new_wait)).hexdigest()
            if seen["cmdline_sha256"] not in {
                expected,
                hashlib.sha256(b"\0".join(x.encode() for x in new_boot)).hexdigest(),
            }:
                raise ValueError("R2 queue identity differs")
            task.update(
                pid=seen["pid"],
                pid_start_ticks=seen["start_ticks"],
                pid_started_epoch=seen["started_epoch"],
                pid_cmdline_sha256=seen["cmdline_sha256"],
            )
            s["tasks"][task["id"]].update(
                seen, state="RUNNING", reason="R2_V23_RAW_ROUTING_RECOVERY"
            )
    c.update(
        status_path=str(STATE),
        lock_path=str(OPS / ".EVENT_WATCHDOG_V15.lock"),
        watchdog_exec_exit_overlay_source_sha256=WD_SHA,
    )
    atom(CFG, c)
    s["config_sha256"] = sha(CFG)
    s["source_sha256"] = WD_SHA
    atom(STATE, s)
    cmd = ["/opt/ejepa_env/bin/python", "-u", str(WD), "--config", str(CFG)]
    wd = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V15.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (OPS / "EVENT_WATCHDOG_V15.pid").write_text(f"{wd.pid}\n")
    time.sleep(5)
    if wd.poll() is not None:
        raise RuntimeError("V15 watchdog failed")
    atom(
        OPS / "EVENT_V15_DEPLOYMENT.json",
        {
            "schema": "ejepa_event_v15_deployment_v1",
            "status": "RUNNING",
            "watchdog_pid": wd.pid,
            "watchdog_source_sha256": WD_SHA,
            "queue_bootstrap_sha256": sha(BOOT),
            "config_sha256": sha(CFG),
            "optimizer_updates_reset": False,
        },
    )


if __name__ == "__main__":
    main()
