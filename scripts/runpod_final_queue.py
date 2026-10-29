"""Durable final-fit queue; scientific guards remain authoritative in each worker."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil

OPS = Path("/workspace/ejepa_final_operations")
RUN = Path("/workspace/ejepa_final_runs")
ENTRY = OPS / "runpod_final_bootstrap.py"
PYTHON = "/opt/ejepa_env/bin/python"
DEADLINE = 1793145600.0
STAGES = ("PAIR_E_MATCHED", "E_CTX_MATCHED", "R_CTX", "PHASE_ONLY_EVENT_S7_TRAIN40", "PAIR_R")
HEAD_SLOTS = 5
EXPORT_STATUS = "TRAINING_EXPORTED_TEST12_PENDING"


def publish(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def complete(stage: str) -> bool:
    return read(RUN / "final/fits" / stage / "CHECKPOINT_RECEIPT.json").get("status") == "COMPLETE"


def job_complete(name: str) -> bool:
    if name == "export_test12":
        receipt = read(RUN / "final/TRAINING_EXPORT_STATUS.json")
        return (
            receipt.get("status") == EXPORT_STATUS
            and receipt.get("test12_predictions_executed") is False
            and all(complete(stage) for stage in STAGES)
        )
    if name == "pair_cache":
        return (RUN / "final/cache/PAIR_R.json").is_file() and (
            RUN / "final/cache/PAIR_R.npz"
        ).is_file()
    return complete(name)


def job_name(command: list[str]) -> str | None:
    """Recognize the exact bootstrap and registered jobs, not substring matches."""
    if str(ENTRY) not in command:
        return None
    arguments = command[command.index(str(ENTRY)) + 1 :]
    if arguments == ["pair-cache"]:
        return "pair_cache"
    if arguments == ["finish"]:
        return "export_test12"
    if len(arguments) == 3 and arguments[:2] == ["train", "--stage"] and arguments[2] in STAGES:
        return arguments[2]
    raise ValueError(f"Unrecognized concurrent final bootstrap command: {command}")


def process_record(process: psutil.Process) -> dict[str, Any] | None:
    command = process.cmdline()
    name = job_name(command)
    if name is None or process.status() == psutil.STATUS_ZOMBIE:
        return None
    return {
        "name": name,
        "stage": name if name in STAGES else None,
        "pid": process.pid,
        "created": process.create_time(),
        "cmdline": command,
    }


class Adopted:
    """Track detached workers without trusting PID reuse or a successful exit alone."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        self.pid = record["pid"]

    def poll(self) -> int | None:
        try:
            process = psutil.Process(self.pid)
            if (
                process.create_time() == self.record["created"]
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                if process.cmdline() != self.record["cmdline"]:
                    raise ValueError("Adopted final worker changed its command")
                return None
        except psutil.NoSuchProcess:
            pass
        return 0 if job_complete(self.record["name"]) else 1


def discover_workers(saved: dict[str, Any]) -> dict[str, Adopted]:
    """The inventory also closes a crash between spawning and persisting a worker."""
    workers = {}
    for process in psutil.process_iter():
        try:
            record = process_record(process)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            raise RuntimeError("Cannot establish exclusive final worker inventory") from None
        if record is None:
            continue
        name = record["name"]
        if name in workers:
            raise RuntimeError(f"Multiple live final workers for {name}; refusing launches")
        previous = saved.get(name)
        if (
            previous
            and previous["pid"] == record["pid"]
            and previous["created"] == record["created"]
        ):
            if previous.get("cmdline", record["cmdline"]) != record["cmdline"]:
                raise ValueError(f"Persisted worker command differs: {name}")
        workers[name] = Adopted(record)
    return workers


def main() -> None:
    import fcntl

    OPS.mkdir(exist_ok=True)
    with (OPS / "FINAL_QUEUE.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = OPS / "FINAL_QUEUE_WORKERS.json"
        prior = read(registry)
        saved = {**read(OPS / "EXTERNAL_JOBS.json"), **prior.get("workers", {})}
        children = discover_workers(saved)
        failed = dict(prior.get("failed", {}))
        done_aux = {name for name in prior.get("auxiliary_completed", []) if job_complete(name)}
        for name in children:
            failed.pop(name, None)

        def persist() -> None:
            publish(
                registry,
                {
                    "schema": "ejepa_final_queue_workers_v1",
                    "timestamp": time.time(),
                    "deadline_timestamp": DEADLINE,
                    "workers": {name: child.record for name, child in children.items()},
                    "failed": failed,
                    "auxiliary_completed": sorted(done_aux),
                },
            )

        persist()

        def start(name: str, arguments: list[str]) -> None:
            if time.time() >= DEADLINE or (RUN / "STOP_REQUEST").exists():
                return
            # Adopt manual launches or a spawn/persist orphan before starting anything.
            current = discover_workers({key: child.record for key, child in children.items()})
            for key, child in current.items():
                children.setdefault(key, child)
            if name in children:
                persist()
                return
            with (OPS / (name + ".log")).open("a") as log:
                child = subprocess.Popen(
                    [PYTHON, str(ENTRY), *arguments],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    env=dict(
                        os.environ,
                        OMP_NUM_THREADS="2",
                        MKL_NUM_THREADS="2",
                        CUBLAS_WORKSPACE_CONFIG=":4096:8",
                    ),
                )
            try:
                record = process_record(psutil.Process(child.pid))
            except psutil.NoSuchProcess:
                record = None
            if record is None:
                failed[name] = {"reason": "WORKER_EXITED_BEFORE_REGISTRATION", "pid": child.pid}
            else:
                if record["name"] != name:
                    raise ValueError("Launched final worker identity differs")
                child.record = record
                children[name] = child
            persist()

        while time.time() < DEADLINE and not (RUN / "STOP_REQUEST").exists():
            for name, child in list(children.items()):
                code = child.poll()
                if code is not None:
                    del children[name]
                    if job_complete(name):
                        done_aux.add(name)
                        failed.pop(name, None)
                    else:
                        failed[name] = {
                            "returncode": code,
                            "reason": "MISSING_COMPLETION_RECEIPT",
                            "log": str(OPS / (name + ".log")),
                        }
                    persist()
            prepared = (RUN / "final/OOF_PREPARATION.json").exists()
            if prepared:
                for stage in STAGES:
                    if sum(name in STAGES for name in children) >= HEAD_SLOTS:
                        break
                    cache = RUN / (
                        "cache/PAIR_E_MATCHED.npz"
                        if stage == "PAIR_E_MATCHED"
                        else f"final/cache/{stage}.npz"
                    )
                    if (
                        cache.exists()
                        and not complete(stage)
                        and stage not in children
                        and stage not in failed
                    ):
                        start(stage, ["train", "--stage", stage])
            rgb_ready = Path("/opt/ejepa_final_rgb/RGB_READY.json").exists()
            pair_ready = job_complete("pair_cache")
            if (
                rgb_ready
                and not pair_ready
                and len(children) < HEAD_SLOTS + 1
                and "pair_cache" not in children
                and "pair_cache" not in failed
                and "pair_cache" not in done_aux
            ):
                start("pair_cache", ["pair-cache"])
            fits_complete = all(complete(stage) for stage in STAGES)
            if (
                fits_complete
                and not children
                and not job_complete("export_test12")
                and "export_test12" not in failed
                and "export_test12" not in done_aux
            ):
                start("export_test12", ["finish"])
            exported = job_complete("export_test12")
            state = {
                "schema": "ejepa_final_train40_queue_v1",
                "pid": os.getpid(),
                "timestamp": time.time(),
                "deadline_utc": "2026-10-28T00:00:00+00:00",
                "prepared": prepared,
                "rgb_ready": rgb_ready,
                "completed": [stage for stage in STAGES if complete(stage)],
                "running": {key: child.pid for key, child in children.items()},
                "failed": failed,
                "auxiliary_completed": sorted(done_aux),
                "submission_authorized": False,
                "test12_predictions_executed": False,
                "status": EXPORT_STATUS
                if exported
                else ("BLOCKED" if failed and not children else "RUNNING"),
            }
            publish(OPS / "FINAL_QUEUE_STATUS.json", state)
            if exported:
                return
            time.sleep(10)
        persist()
        publish(
            OPS / "FINAL_QUEUE_STOP.json",
            {
                "timestamp": time.time(),
                "reason": "DEADLINE_OR_MANUAL",
                "checkpoints_preserved": True,
            },
        )


if __name__ == "__main__":
    main()
