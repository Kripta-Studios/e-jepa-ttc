"""Deploy cloud-admission-gated R3/R4 waiters without touching the live R2 queue."""

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
OLD_CONFIG = OPS / "EVENT_WATCHDOG_CONFIG_V10.json"
OLD_STATUS = OPS / "EVENT_WATCHDOG_STATUS_V10.json"
NEW_CONFIG = OPS / "EVENT_WATCHDOG_CONFIG_V11.json"
NEW_STATUS = OPS / "EVENT_WATCHDOG_STATUS_V11.json"
V21 = OPS / "event_downstream_runtime_v21"
V22 = OPS / "event_downstream_runtime_v22"
WAITER = OPS / "scripts/runpod_event_queue_waiter_v2.py"
BOOTSTRAP = OPS / "runpod_event_queue_bootstrap_v6.py"
WATCHDOG = OPS / "runpod_event_resource_watchdog_v8.py"
WAITER_SHA = "2ecd5f7f4b41ae569ca9670b555dd3617d0a0f2849a73c44a8761dfc2a140292"
BOOTSTRAP_SHA = "f1e2aef60b0d628d472fcce218a0c03c7caa5778d01c121317db7b46f7f2bda4"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def ident(pid: int) -> dict[str, object]:
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
        "pid_start_ticks": ticks,
        "pid_started_epoch": boot + ticks / os.sysconf("SC_CLK_TCK"),
        "pid_cmdline_sha256": hashlib.sha256(b"\0".join(args)).hexdigest(),
    }


def environment(config: dict[str, object]) -> dict[str, str]:
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
    if sha(WAITER) != WAITER_SHA or sha(BOOTSTRAP) != BOOTSTRAP_SHA:
        raise ValueError("cloud gate source differs")
    if V22.exists():
        raise FileExistsError(V22)
    shutil.copytree(V21, V22, symlinks=True)
    shutil.copy2(BOOTSTRAP, V22 / "scripts/runpod_event_queue_bootstrap_v6.py")
    config = json.loads(OLD_CONFIG.read_text())
    status = json.loads(OLD_STATUS.read_text())
    env = environment(config)
    w = int((OPS / "EVENT_WATCHDOG_V10.pid").read_text())
    expected_w = ["/opt/ejepa_env/bin/python", "-u", str(WATCHDOG), "--config", str(OLD_CONFIG)]
    if Path(f"/proc/{w}/cmdline").read_bytes().split(b"\0")[:-1] != [
        x.encode() for x in expected_w
    ]:
        raise ValueError("watchdog identity differs")
    os.kill(w, signal.SIGTERM)
    for _ in range(100):
        if not Path(f"/proc/{w}").exists():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("watchdog did not exit")
    for task in config["tasks"]:
        if task["id"] not in {"R3_EVENT_DOWNSTREAM", "R4_EVENT_DOWNSTREAM"}:
            continue
        oldpid = int(status["tasks"][task["id"]]["pid"])
        old_identity = ident(oldpid)
        if not (
            old_identity["pid_start_ticks"] == status["tasks"][task["id"]]["start_ticks"]
            and old_identity["pid_cmdline_sha256"] == status["tasks"][task["id"]]["cmdline_sha256"]
        ):
            raise ValueError(f"{task['id']} live waiter identity differs")
        os.kill(oldpid, signal.SIGTERM)
        for _ in range(100):
            if not Path(f"/proc/{oldpid}").exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"{task['id']} waiter did not exit")

        def replace(items: list[str]) -> list[str]:
            return [
                x.replace(
                    str(OPS / "scripts/runpod_event_queue_waiter_v1.py"), str(WAITER)
                ).replace(
                    str(V21 / "scripts/runpod_event_queue_bootstrap_v5.py"),
                    str(V22 / "scripts/runpod_event_queue_bootstrap_v6.py"),
                )
                for x in items
            ]

        task["argv"] = replace(task["argv"])
        task["adopted_argv"] = [replace(x) for x in task["adopted_argv"]]
        task["entrypoint"] = str(WAITER)
        task["entrypoint_sha256"] = WAITER_SHA
        files = []
        for item in task["runtime_files"]:
            path = Path(item["path"])
            if path.name == "runpod_event_queue_waiter_v1.py":
                path = WAITER
            elif path.name == "runpod_event_queue_bootstrap_v5.py":
                path = V22 / "scripts/runpod_event_queue_bootstrap_v6.py"
            elif str(path).startswith(str(V21)):
                path = Path(str(path).replace(str(V21), str(V22), 1))
            files.append({"path": str(path), "sha256": sha(path)})
        task["runtime_files"] = files
        proc = subprocess.Popen(
            task["argv"],
            stdin=subprocess.DEVNULL,
            stdout=Path(task["log"]).open("ab", buffering=0),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        time.sleep(2)
        seen = ident(proc.pid)
        if (
            seen["pid_cmdline_sha256"]
            != hashlib.sha256(b"\0".join(x.encode() for x in task["argv"])).hexdigest()
        ):
            raise ValueError(f"{task['id']} argv differs")
        task.update(seen)
        status["tasks"][task["id"]].update(
            pid=seen["pid"],
            start_ticks=seen["pid_start_ticks"],
            started_epoch=seen["pid_started_epoch"],
            cmdline_sha256=seen["pid_cmdline_sha256"],
            state="RUNNING",
            reason="CLOUD_ADMISSION_GATED_WAITER_V2",
        )
    config["status_path"] = str(NEW_STATUS)
    config["lock_path"] = str(OPS / ".EVENT_WATCHDOG_V11.lock")
    atomic(NEW_CONFIG, config)
    status["config_sha256"] = sha(NEW_CONFIG)
    atomic(NEW_STATUS, status)
    expected = ["/opt/ejepa_env/bin/python", "-u", str(WATCHDOG), "--config", str(NEW_CONFIG)]
    wd = subprocess.Popen(
        expected,
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V11.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (OPS / "EVENT_WATCHDOG_V11.pid").write_text(f"{wd.pid}\n")
    time.sleep(4)
    if wd.poll() is not None:
        raise RuntimeError("watchdog V11 failed")
    atomic(
        OPS / "EVENT_CLOUD_GATE_V11_DEPLOYMENT.json",
        {
            "schema": "ejepa_event_cloud_gate_deployment_v1",
            "status": "RUNNING",
            "watchdog_pid": wd.pid,
            "waiter_sha256": WAITER_SHA,
            "bootstrap_sha256": BOOTSTRAP_SHA,
            "tasks": ["R3_EVENT_DOWNSTREAM", "R4_EVENT_DOWNSTREAM"],
            "optimizer_updates_performed": 0,
        },
    )


if __name__ == "__main__":
    main()
