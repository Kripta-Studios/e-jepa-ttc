"""Recover the failed V16 handoff with full admitted EVENT raw-root relocation."""

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
CFG = OPS / "EVENT_WATCHDOG_CONFIG_V17.json"
STATE = OPS / "EVENT_WATCHDOG_STATUS_V17.json"
V24 = OPS / "event_downstream_runtime_v24"
V25 = OPS / "event_downstream_runtime_v25"
BOOT_SRC = OPS / "runpod_event_queue_bootstrap_v7_fixed2.py"
BOOT = V25 / "scripts/runpod_event_queue_bootstrap_v7.py"
BOOT_SHA = "7a2ed37dc21a75156ab2f3c7f52c4c192f19fd7790061f1b6c9815ce82450b9e"
WD = OPS / "runpod_event_resource_watchdog_v10.py"
WD_SHA = "3015df79d8220041328382133447e30c4053e2646f74eff2b40f2f4c08bc03ee"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def argv_sha(argv: list[str]) -> str:
    return hashlib.sha256(b"\0".join(item.encode() for item in argv)).hexdigest()


def atom(path: Path, value: object) -> None:
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def process(pid: int) -> tuple[dict[str, object], list[str]]:
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
    return (
        {
            "pid": pid,
            "start_ticks": ticks,
            "started_epoch": boot + ticks / os.sysconf("SC_CLK_TCK"),
            "cmdline_sha256": argv_sha(argv),
        },
        argv,
    )


def stop_waiter(pid: int) -> None:
    if not Path(f"/proc/{pid}").exists():
        return
    _, argv = process(pid)
    if not any("runpod_event_queue_waiter_v2.py" in item for item in argv):
        raise ValueError("R3 process is no longer a waiter")
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.1)
    raise RuntimeError("R3 V24 waiter did not exit")


def launch(task: dict[str, object], env: dict[str, str]) -> tuple[dict[str, object], list[str]]:
    child = subprocess.Popen(
        task["argv"],
        stdin=subprocess.DEVNULL,
        stdout=Path(str(task["log"])).open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    time.sleep(6)
    if child.poll() is not None:
        raise RuntimeError(f"{task['id']} failed V25 launch")
    return process(child.pid)


def main() -> None:
    if sha(WD) != WD_SHA or sha(BOOT_SRC) != BOOT_SHA:
        raise ValueError("V17 sources differ")
    for name in ("EVENT_WATCHDOG_V15.pid", "EVENT_WATCHDOG_V16.pid"):
        path = OPS / name
        if path.exists() and Path(f"/proc/{int(path.read_text())}").exists():
            raise RuntimeError(f"Prior watchdog remains alive: {name}")
    config = json.loads((OPS / "EVENT_WATCHDOG_CONFIG_V16.json").read_text())
    state = json.loads((OPS / "EVENT_WATCHDOG_STATUS_V16.json").read_text())
    if V25.exists():
        raise FileExistsError(V25)
    shutil.copytree(V24, V25, symlinks=True)
    shutil.copy2(BOOT_SRC, BOOT)
    if sha(BOOT) != BOOT_SHA:
        raise ValueError("installed V25 bootstrap differs")

    env = os.environ.copy()
    env.update(config["required_environment"])
    launched: dict[str, dict[str, object]] = {}
    for task in config["tasks"]:
        if task["fit"] != "EVENT_DOWNSTREAM":
            continue
        waiter = [
            item.replace(str(V24 / "scripts/runpod_event_queue_bootstrap_v7.py"), str(BOOT))
            for item in task["argv"]
        ]
        entry = waiter.index("--queue-entrypoint")
        bootstrap = [waiter[0], waiter[entry + 1], *waiter[entry + 2 :]]
        poll_index = bootstrap.index("--poll-seconds") + 1
        bootstrap[poll_index] = str(float(bootstrap[poll_index]))
        task["argv"] = waiter
        for admitted in (waiter, bootstrap):
            if admitted not in task["adopted_argv"]:
                task["adopted_argv"].append(admitted)
        runtime = {"path": str(BOOT), "sha256": BOOT_SHA}
        if runtime not in task["runtime_files"]:
            task["runtime_files"].append(runtime)

        task_id = str(task["id"])
        if task_id == "R3_EVENT_DOWNSTREAM":
            stop_waiter(int(state["tasks"][task_id]["pid"]))
        if task_id in {"R2_EVENT_DOWNSTREAM", "R3_EVENT_DOWNSTREAM"}:
            seen, actual_argv = launch(task, env)
            if actual_argv not in task["adopted_argv"]:
                raise ValueError(f"{task_id} launched an unadmitted argv")
            task.update(
                pid=seen["pid"],
                pid_start_ticks=seen["start_ticks"],
                pid_started_epoch=seen["started_epoch"],
                pid_cmdline_sha256=seen["cmdline_sha256"],
            )
            state["tasks"][task_id].update(
                seen, state="RUNNING", reason=f"{task_id}_V25_FULL_RAW_ROOT_RELOCATION"
            )
            launched[task_id] = seen
        elif task_id == "R4_EVENT_DOWNSTREAM":
            current = state["tasks"][task_id]
            seen, _ = process(int(current["pid"]))
            if seen["start_ticks"] != current["start_ticks"]:
                raise ValueError("R4 PID generation changed")
            current.update(seen, state="RUNNING", reason="ADOPTED_OR_STILL_RUNNING")

    config.update(status_path=str(STATE), lock_path=str(OPS / ".EVENT_WATCHDOG_V17.lock"))
    atom(CFG, config)
    state.update(config_sha256=sha(CFG), source_sha256=WD_SHA)
    atom(STATE, state)
    watchdog = subprocess.Popen(
        ["/opt/ejepa_env/bin/python", "-u", str(WD), "--config", str(CFG)],
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V17.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (OPS / "EVENT_WATCHDOG_V17.pid").write_text(f"{watchdog.pid}\n")
    time.sleep(5)
    if watchdog.poll() is not None:
        raise RuntimeError("V17 watchdog failed")
    atom(
        OPS / "EVENT_V17_DEPLOYMENT.json",
        {
            "schema": "ejepa_event_v17_deployment_v1",
            "status": "RUNNING",
            "watchdog_pid": watchdog.pid,
            "watchdog_source_sha256": WD_SHA,
            "queue_bootstrap_sha256": BOOT_SHA,
            "config_sha256": sha(CFG),
            "launched": launched,
            "optimizer_updates_reset": False,
        },
    )


if __name__ == "__main__":
    main()
