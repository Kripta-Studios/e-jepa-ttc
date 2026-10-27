"""Deploy corrected EVENT raw-root routing without disturbing active science."""

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
OLD_CFG = OPS / "EVENT_WATCHDOG_CONFIG_V15.json"
OLD_STATE = OPS / "EVENT_WATCHDOG_STATUS_V15.json"
CFG = OPS / "EVENT_WATCHDOG_CONFIG_V16.json"
STATE = OPS / "EVENT_WATCHDOG_STATUS_V16.json"
V23 = OPS / "event_downstream_runtime_v23"
V24 = OPS / "event_downstream_runtime_v24"
BOOT_SRC = OPS / "runpod_event_queue_bootstrap_v7_fixed.py"
BOOT = V24 / "scripts/runpod_event_queue_bootstrap_v7.py"
WD = OPS / "runpod_event_resource_watchdog_v10.py"
WD_SHA = "3015df79d8220041328382133447e30c4053e2646f74eff2b40f2f4c08bc03ee"
BOOT_SHA = "cda0b2e690df02beb5565602413c891bc674c9b2755ab0f297dff42064a847bc"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atom(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def identity(pid: int) -> dict[str, object]:
    argv = [part for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
    stat = Path(f"/proc/{pid}/stat").read_text().split()
    ticks = int(stat[21])
    boot = int(
        next(
            line.split()[1]
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
    )
    return {
        "pid": pid,
        "start_ticks": ticks,
        "started_epoch": boot + ticks / os.sysconf("SC_CLK_TCK"),
        "cmdline_sha256": hashlib.sha256(b"\0".join(argv)).hexdigest(),
    }


def argv_sha(argv: list[str]) -> str:
    return hashlib.sha256(b"\0".join(item.encode() for item in argv)).hexdigest()


def stop_exact_waiter(task: dict[str, object], current: dict[str, object]) -> None:
    pid = int(current["pid"])
    if not Path(f"/proc/{pid}").exists():
        return
    seen = identity(pid)
    admitted = {argv_sha(argv) for argv in task["adopted_argv"]}  # type: ignore[index]
    if seen["start_ticks"] != current["start_ticks"] or seen["cmdline_sha256"] not in admitted:
        raise ValueError("R3 waiter identity changed before controlled replacement")
    if b"runpod_event_queue_waiter_v2.py" not in Path(f"/proc/{pid}/cmdline").read_bytes():
        raise ValueError("R3 is no longer an idle waiter")
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.1)
    raise RuntimeError("R3 waiter did not exit")


def launch(task: dict[str, object], argv: list[str], env: dict[str, str]) -> dict[str, object]:
    log = Path(str(task["log"]))
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=log.open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    time.sleep(3)
    if process.poll() is not None:
        raise RuntimeError(f"{task['id']} exited during launch; inspect {log}")
    return identity(process.pid)


def main() -> None:
    if sha(WD) != WD_SHA or sha(BOOT_SRC) != BOOT_SHA:
        raise ValueError("V16 source bytes differ")
    watchdog_pid = int((OPS / "EVENT_WATCHDOG_V15.pid").read_text())
    os.kill(watchdog_pid, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{watchdog_pid}").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("V15 watchdog did not exit")

    # The previous supervisor is now frozen: read its final state, never a stale copy.
    config = json.loads(OLD_CFG.read_text())
    state = json.loads(OLD_STATE.read_text())
    if V24.exists():
        raise FileExistsError(V24)
    shutil.copytree(V23, V24, symlinks=True)
    shutil.copy2(BOOT_SRC, BOOT)
    if sha(BOOT) != BOOT_SHA:
        raise ValueError("installed V24 bootstrap differs")

    env = os.environ.copy()
    env.update(config["required_environment"])
    launched: dict[str, dict[str, object]] = {}
    for task in config["tasks"]:
        if task["fit"] != "EVENT_DOWNSTREAM":
            continue
        new_wait = [
            item.replace(
                str(V23 / "scripts/runpod_event_queue_bootstrap_v7.py"), str(BOOT)
            ).replace(
                str(
                    OPS / "event_downstream_runtime_v22/scripts/runpod_event_queue_bootstrap_v6.py"
                ),
                str(BOOT),
            )
            for item in task["argv"]
        ]
        queue_index = new_wait.index("--queue-entrypoint") + 1
        new_boot = [new_wait[0], new_wait[queue_index], *new_wait[queue_index + 1 :]]
        task["argv"] = new_wait
        for candidate in (new_wait, new_boot):
            if candidate not in task["adopted_argv"]:
                task["adopted_argv"].append(candidate)
        runtime = {"path": str(BOOT), "sha256": BOOT_SHA}
        if runtime not in task["runtime_files"]:
            task["runtime_files"].append(runtime)

        task_id = str(task["id"])
        if task_id == "R3_EVENT_DOWNSTREAM":
            stop_exact_waiter(task, state["tasks"][task_id])
        if task_id in {"R2_EVENT_DOWNSTREAM", "R3_EVENT_DOWNSTREAM"}:
            seen = launch(task, new_wait, env)
            task.update(
                pid=seen["pid"],
                pid_start_ticks=seen["start_ticks"],
                pid_started_epoch=seen["started_epoch"],
                pid_cmdline_sha256=seen["cmdline_sha256"],
            )
            state["tasks"][task_id].update(
                seen, state="RUNNING", reason=f"{task_id}_V24_RAW_ROOT_ROUTING_RECOVERY"
            )
            launched[task_id] = seen
        elif task_id == "R4_EVENT_DOWNSTREAM":
            current = state["tasks"][task_id]
            pid = int(current["pid"])
            if not Path(f"/proc/{pid}").exists():
                raise RuntimeError("R4 active queue disappeared during deployment")
            seen = identity(pid)
            if seen["start_ticks"] != current["start_ticks"]:
                raise ValueError("R4 PID generation changed during deployment")
            current.update(seen, state="RUNNING", reason="ADOPTED_OR_STILL_RUNNING")

    config.update(status_path=str(STATE), lock_path=str(OPS / ".EVENT_WATCHDOG_V16.lock"))
    atom(CFG, config)
    state["config_sha256"] = sha(CFG)
    state["source_sha256"] = WD_SHA
    atom(STATE, state)
    command = ["/opt/ejepa_env/bin/python", "-u", str(WD), "--config", str(CFG)]
    watchdog = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V16.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (OPS / "EVENT_WATCHDOG_V16.pid").write_text(f"{watchdog.pid}\n")
    time.sleep(5)
    if watchdog.poll() is not None:
        raise RuntimeError("V16 watchdog failed")
    atom(
        OPS / "EVENT_V16_DEPLOYMENT.json",
        {
            "schema": "ejepa_event_v16_deployment_v1",
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
