"""Resume the three admitted downstream queues on the replacement Pod."""

import hashlib
import importlib.util
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPS = Path("/workspace/ejepa_event_operations")
OUT = OPS / "resume_20261027"
RUN = Path("/workspace/ejepa_event_runs")
OLD = OPS / "manual_pause_20261026"
BOOT = OUT / "runpod_event_relocated_bootstrap_20261027.py"
CONFIG = OPS / "EVENT_WATCHDOG_CONFIG_V20.json"
STATE = OPS / "EVENT_WATCHDOG_STATUS_V20.json"


def digest(p: Path) -> str:
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atom(path: Path, value: Any) -> None:
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def identity(pid: int) -> tuple[dict[str, Any], list[str]]:
    root = Path("/proc") / str(pid)
    tail = (root / "stat").read_text().rsplit(")", 1)[-1].strip().split()
    assert tail[0] not in ["Z", "X"]
    ticks = int(tail[19])
    args = [s.decode() for s in (root / "cmdline").read_bytes().split(b"\0") if s]
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
        "cmdline_sha256": hashlib.sha256(b"\0".join(a.encode() for a in args)).hexdigest(),
    }, args


def main() -> None:
    assert json.loads((OUT / "RESTORE_STATUS.json").read_text())["stage"] == "RESTORED"
    assert not CONFIG.exists(), "Resume already attempted; inspect instead of duplicate launch"
    config = json.loads((OLD / "EVENT_WATCHDOG_CONFIG_V18.json").read_text())
    state = json.loads((OLD / "EVENT_WATCHDOG_STATUS_V18.json").read_text())
    deadline = datetime.fromisoformat(config["authority"]["global_deadline_utc"])
    assert datetime.now(UTC) < deadline
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            a = (p / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError):
            continue
        assert b"/opt/ejepa_env/bin/python\0" not in a, (
            "Existing scientific Python process: inspect"
        )
    checks = json.loads((OLD / "SCIENTIFIC_PAUSE_CHECK.json").read_text())
    for c in checks["checkpoints"]:
        assert digest(Path(c["path"])) == c["sha256"]
    env = os.environ.copy()
    env.update(config["required_environment"])
    env["EJEPA_PHYSICAL_RELOCATION_SHA256"] = digest(OUT / "PHYSICAL_RELOCATION.json")
    env["EJEPA_ORIGINAL_V25_BOOTSTRAP_SHA256"] = digest(
        OPS / "event_downstream_runtime_v25/scripts/runpod_event_queue_bootstrap_v7.py"
    )
    config["required_environment"].update(
        {
            k: env[k]
            for k in ["EJEPA_PHYSICAL_RELOCATION_SHA256", "EJEPA_ORIGINAL_V25_BOOTSTRAP_SHA256"]
        }
    )
    good = subprocess.run(
        ["/opt/ejepa_env/bin/python", str(BOOT), "--preflight"],
        env=env,
        capture_output=True,
        text=True,
    )
    (OUT / "preflight.log").write_text(good.stdout + good.stderr)
    assert good.returncode == 0, good.stdout + good.stderr
    badenv = env.copy()
    badenv["EJEPA_PHYSICAL_RELOCATION_SHA256"] = "0" * 64
    bad = subprocess.run(
        ["/opt/ejepa_env/bin/python", str(BOOT), "--preflight"],
        env=badenv,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0, "Relocation mismatch must fail closed"
    for task in config["tasks"]:
        if task["fit"] != "EVENT_DOWNSTREAM":
            continue
        argv = list(task["argv"])
        argv[argv.index(task["entrypoint"])] = str(BOOT)
        if "--queue-entrypoint" in argv:
            index = argv.index("--queue-entrypoint")
            del argv[index : index + 2]
        task.update(
            argv=argv,
            adopted_argv=[argv],
            entrypoint=str(BOOT),
            entrypoint_sha256=digest(BOOT),
            pid_cmdline_sha256=hashlib.sha256(b"\0".join(a.encode() for a in argv)).hexdigest(),
        )
        task["log"] = str(OUT / (task["id"] + ".log"))
        task["runtime_files"] += [
            {"path": str(p), "sha256": digest(p)} for p in [BOOT, OUT / "PHYSICAL_RELOCATION.json"]
        ]
    config.update(status_path=str(STATE), lock_path=str(OPS / ".EVENT_WATCHDOG_V20.lock"))
    atom(CONFIG, config)
    # Validate the entire frozen authority/runtime closure before removing STOP.
    wd = OPS / "runpod_event_resource_watchdog_v10.py"
    spec = importlib.util.spec_from_file_location("resume_watchdog_v10", wd)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validated = module.install_v10()
    validated.validate_config(CONFIG)
    stop = RUN / "STOP_REQUEST"
    assert stop.read_text() == "USER_MANUAL_PAUSE_20261026_LOW_CREDIT_DO_NOT_AUTORESTART\n"
    os.replace(stop, OUT / "USER_STOP_REQUEST_REVOKED")
    started = []
    for task in config["tasks"]:
        if task["fit"] != "EVENT_DOWNSTREAM":
            continue
        log = Path(task["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(
            task["argv"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log.open("ab", buffering=0),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(1)
        observed, argv = identity(p.pid)
        assert argv == task["argv"]
        task.update(
            pid=p.pid,
            pid_start_ticks=observed["start_ticks"],
            pid_started_epoch=observed["started_epoch"],
            pid_cmdline_sha256=observed["cmdline_sha256"],
        )
        state["tasks"][task["id"]].update(
            **observed,
            state="RUNNING",
            reason="USER_RESUMED_ON_REPLACEMENT_POD",
            retries=0,
            next_retry_epoch=0,
        )
        started.append({"id": task["id"], **observed})
    atom(CONFIG, config)
    state.update(config_sha256=digest(CONFIG), source_sha256=digest(wd))
    atom(STATE, state)
    aux = []
    for i, entry in enumerate(json.loads((OLD / "ANCILLARY_PROCESSES.json").read_text())):
        argv = entry["argv"][:]
        if any("runpod_event_cache_controller" in a for a in argv):
            lock = Path(argv[argv.index("--lock") + 1])
            if lock.exists():
                os.replace(lock, OUT / "PREVIOUS_CACHE_CONTROLLER.lock")
            argv += ["--seal", str(OUT / "ejepa_event_bundle.RELOCATED_SEAL.json")]
        p = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=(OUT / f"auxiliary_{i}.log").open("ab", buffering=0),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        aux.append({"pid": p.pid, "argv": argv})
    p = subprocess.Popen(
        ["/opt/ejepa_env/bin/python", "-u", str(wd), "--config", str(CONFIG)],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=(OPS / "EVENT_WATCHDOG_V20.log").open("ab", buffering=0),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (OPS / "EVENT_WATCHDOG_V20.pid").write_text(str(p.pid) + "\n")
    atom(
        OUT / "RESUME_RECEIPT.json",
        {
            "status": "LAUNCHED_VERIFY_PROGRESS",
            "pod_id": "lreiivpf18vxu7",
            "host": "213.173.108.97",
            "port": 17473,
            "previous_pod_id": "8mqxzys5bz6218",
            "network_volume": "3mritxijmp",
            "utc": datetime.now(UTC).isoformat(),
            "deadline_unchanged": deadline.isoformat(),
            "historical_budget_not_reset": True,
            "user_last_reported_balance_usd": 7,
            "current_provider_balance_not_visible": True,
            "scientific_config_modified": False,
            "tasks": started,
            "auxiliaries": aux,
            "watchdog_pid": p.pid,
        },
    )
    print(json.dumps({"tasks": started, "watchdog_pid": p.pid}), flush=True)


if __name__ == "__main__":
    main()
