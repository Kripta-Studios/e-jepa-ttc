"""Recover missing TRAIN40 supervisors at each local clock hour without loading Torch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

from operational.train40_system.durable_io import atomic_json

CONTROLLER = "operational.train40_system.controller_c2f_graph"
FOLLOWUP = "operational.train40_system.followup"
PAUSES = ("COORDINATION_PAUSE_REQUEST.json", "STOP_REQUEST")


def read(path: Path) -> dict[str, Any]:
    """Read a small boundary record; never open training tensors or raw data."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Hash a bound source or configuration using bounded memory."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def next_hour(now: datetime) -> datetime:
    """Return the next wall-clock hour, preserving the supplied local timezone."""
    return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)


def inventory(root: Path, output: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Inspect Python processes once; ambiguous access prevents automatic relaunch."""
    records, errors = [], []
    for process in psutil.process_iter(["name"]):
        if not (process.info["name"] or "").lower().startswith("python"):
            continue
        try:
            command = process.cmdline()
            if "-m" not in command:
                continue
            module = command[command.index("-m") + 1]
            if not module.startswith("operational.train40_system."):
                continue
            if Path(process.cwd()).resolve() != root:
                continue
            if "--output" in command:
                destination = Path(command[command.index("--output") + 1]).resolve()
                if destination != output:
                    continue
            records.append(
                {"pid": process.pid, "create_time": process.create_time(), "module": module}
            )
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, IndexError, OSError) as exc:
            errors.append(f"PID {process.pid}: {type(exc).__name__}")
    return records, errors


def owner_state(path: Path) -> str:
    """Match lease PID and creation time before considering its owner missing."""
    if not path.exists():
        return "MISSING"
    try:
        owner = read(path)
        process = psutil.Process(int(owner["pid"]))
        return "LIVE" if process.create_time() == owner["create_time"] else "DEAD"
    except psutil.NoSuchProcess:
        return "DEAD"
    except (psutil.AccessDenied, OSError, ValueError, KeyError):
        return "UNKNOWN"


def launch(config: dict[str, Any], command: list[str], tag: str) -> dict[str, Any]:
    """Restart only an admitted supervisor; its existing queue owns all GPU work."""
    directory = Path(config["output"]) / "hourly_supervisor"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    log_path = directory / f"{tag}_{stamp}.log"
    environment = dict(os.environ)
    environment.update(config["environment"])
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=config["root"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    return {"pid": process.pid, "command": command, "log_path": str(log_path)}


def review(
    config: dict[str, Any],
    *,
    now: datetime | None = None,
    inspect: Callable = inventory,
    owner: Callable = owner_state,
    start: Callable = launch,
) -> dict[str, Any]:
    """Respect completion, deadline, pauses, exact processes and supervisor leases."""
    now = now or datetime.now(UTC)
    root, output = Path(config["root"]).resolve(), Path(config["output"]).resolve()
    record: dict[str, Any] = {
        "checked_utc": now.isoformat(),
        "checked_local": now.astimezone().isoformat(),
        "status": "HEALTHY",
        "optimizer_updates": 0,
        "actions": [],
        "scientific_negative": False,
    }
    deadline = datetime.fromisoformat(read(output / "AUTHORIZATION.json")["deadline_utc"])
    bundle = output / "BUNDLE_VERIFICATION.json"
    if bundle.exists() and read(bundle).get("status") == "PASSED":
        record["status"] = "COMPLETE_VERIFIED"
        return record
    if now >= deadline:
        record["status"] = "DEADLINE_STOP_PRESERVED"
        return record
    pauses = [name for name in (*PAUSES, "HOURLY_SUPERVISOR_STOP") if (output / name).exists()]
    if pauses:
        record.update(status="PAUSED_BY_REQUEST", pause_markers=pauses)
        return record
    for item in config["bound_files"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Bound supervisor contract changed: {item['path']}")
    processes, errors = inspect(root, output)
    record.update(processes=processes, inspection_errors=errors)
    if errors:
        record["status"] = "PROCESS_INSPECTION_BLOCKED"
        return record
    for name in ("CONTROLLER_PROGRESS", "POST_QUEUE_PROGRESS"):
        path = output / f"{name}.json"
        if path.exists():
            record[name] = read(path)
    record["fits"] = {}
    for fit in ("a5_seed7", "c2f_seed7", "pair_seed7", "h8_seed7", "h8_seed13", "h8_seed23"):
        path = output / "fits" / fit / "CHECKPOINT_RECEIPT.json"
        if path.exists():
            record["fits"][fit] = read(path)
    completed = output / "TRAINING_QUEUE_COMPLETION.json"
    queue_complete = completed.exists() and read(completed).get("status") == "COMPLETE"
    for tag, module, lease in (
        ("controller", CONTROLLER, "controller"),
        ("followup", FOLLOWUP, "post_queue"),
    ):
        if tag == "controller" and queue_complete:
            continue
        live = [item for item in processes if item["module"] == module]
        state = owner(output / lease / "WRITER.lock")
        record[f"{tag}_lease"] = state
        if live or state == "LIVE":
            continue
        if state == "UNKNOWN":
            record["status"] = "OWNER_INSPECTION_BLOCKED"
            continue
        missing = [path for path in config["required_paths"][tag] if not Path(path).exists()]
        if missing:
            record.update(status="DEPENDENCY_UNAVAILABLE", missing_paths=missing)
            continue
        command = config["commands"][tag]
        if command[1:3] != ["-m", module]:
            raise ValueError("Supervisor command is outside the admitted queue")
        action = start(config, command, tag)
        record["actions"].append({"task": tag, **action})
        record["status"] = "SUPERVISOR_RESTART_REQUESTED"
    return record


@contextmanager
def review_lock(directory: Path) -> Iterator[bool]:
    """Hold an OS lock during one check; crashes release it without deleting history."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "REVIEW.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check(config: dict[str, Any]) -> dict[str, Any]:
    """Persist every check and dependency failure without interpreting it as a result."""
    directory = Path(config["output"]) / "hourly_supervisor"
    with review_lock(directory) as acquired:
        if not acquired:
            return {"status": "CHECK_ALREADY_RUNNING"}
        try:
            result = review(config)
        except Exception as exc:
            result = {
                "status": "CHECK_BLOCKED_PRESERVED",
                "checked_utc": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "optimizer_updates": 0,
                "scientific_negative": False,
            }
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        atomic_json(directory / f"CHECK_{stamp}.json", result)
        atomic_json(directory / "HOURLY_PROGRESS.json", result)
        return result


def main() -> None:
    """Run once for Windows Task Scheduler, or use a clock-aligned local timer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if digest(args.config) != args.config_sha256:
        raise ValueError("Hourly supervisor configuration changed")
    config = read(args.config)
    if args.once:
        check(config)
        return
    while True:
        result = check(config)
        if result["status"] in {"COMPLETE_VERIFIED", "DEADLINE_STOP_PRESERVED"}:
            return
        target = next_hour(datetime.now().astimezone())
        atomic_json(
            Path(config["output"]) / "hourly_supervisor/TIMER_PROGRESS.json",
            {"pid": os.getpid(), "next_check_local": target.isoformat(), "torch_loaded": False},
        )
        while datetime.now().astimezone() < target:
            time.sleep(min(30, max(0.01, target.timestamp() - time.time())))


if __name__ == "__main__":
    main()
