"""Complete the authorized dual/triple windows and resume the best valid observed mode."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

OPS = Path("/workspace/ejepa_operations")
PYTHON = "/opt/ejepa_env/bin/python"
SAMPLER_SHA = "8bb610974fdd72e6a3f84f45a7abdb986518eab5cdae63c6cb3948f1a5078af4"


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def run_root(rotation: int) -> Path:
    return Path(f"/workspace/ejepa_technical/r{rotation}_rgb_v3")


def fit_root(rotation: int) -> Path:
    return run_root(rotation) / "rotations" / str(rotation) / "fits/R_A5"


def live(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def durable(receipt: dict, journal: dict) -> bool:
    return bool(
        receipt
        and journal
        and receipt.get("completed_updates")
        == journal.get("completed_updates")
        == journal.get("durable_updates")
        and journal.get("pending_update_upper") == 0
        and receipt.get("status") in {"PAUSED_RESOURCE", "COMPLETE"}
    )


def main() -> None:
    import fcntl

    with (OPS / "CONCURRENCY_SEQUENCE.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = datetime.fromisoformat(
            read(OPS / "USER_BUDGET_24USD.json")["execution_deadline_utc"]
        )
        sampler = OPS / "runpod_measure_window.py"
        if hashlib.sha256(sampler.read_bytes()).hexdigest() != SAMPLER_SHA:
            raise ValueError("Measurement source changed")
        env = dict(os.environ)
        repo = "/workspace/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement"
        env.update(PYTHONPATH=f"{repo}/src:{repo}", PYTHONDONTWRITEBYTECODE="1")
        children: dict[int, subprocess.Popen] = {}

        def check() -> None:
            if (OPS / "STOP_CONCURRENCY_SEQUENCE").exists() or datetime.now(UTC) >= deadline:
                raise RuntimeError("Sequence stopped by operator or total authorization deadline")

        def status(phase: str, **values: object) -> None:
            record = {
                "phase": phase,
                "utc": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "billing_stop_implied": False,
                **values,
            }
            pending = OPS / "CONCURRENCY_SEQUENCE_STATUS.pending"
            pending.write_text(json.dumps(record, indent=2) + "\n")
            os.replace(pending, OPS / "CONCURRENCY_SEQUENCE_STATUS.json")

        def wait_paused(rotations: tuple[int, ...]) -> dict[int, int]:
            while True:
                check()
                result = {}
                for rotation in rotations:
                    folder = fit_root(rotation)
                    receipt, journal = (
                        read(folder / "CHECKPOINT_RECEIPT.json"),
                        read(folder / "UPDATE_JOURNAL.json"),
                    )
                    pid = read(folder / "PROGRESS.json").get("pid")
                    if durable(receipt, journal) and not live(pid):
                        result[rotation] = receipt["completed_updates"]
                if len(result) == len(rotations):
                    return result
                time.sleep(2)

        def start(rotation: int) -> tuple[int, int]:
            check()
            root = run_root(rotation)
            progress = read(fit_root(rotation) / "PROGRESS.json")
            if live(progress.get("pid")):
                raise RuntimeError(f"Rotation {rotation} still has a live worker")
            marker = root / "STOP_REQUEST"
            if marker.exists():
                if not marker.read_text().startswith("Bounded performance window completed"):
                    raise ValueError("Refusing to clear a non-benchmark stop request")
                os.replace(marker, marker.with_name(f"STOP_AFTER_WINDOW_{time.time_ns()}.txt"))
            directory = OPS if rotation == 4 else OPS / f"r{rotation}"
            entrypoint = OPS / f"runpod_scientific_overlay_v{4 if rotation == 4 else 5}.py"
            command = [
                PYTHON,
                "-u",
                str(entrypoint),
                "producer",
                "--bundle",
                "/workspace/ejepa_bundle",
                "--run",
                str(root),
                "--rotation",
                str(rotation),
                "--fit",
                "R_A5",
                "--budget-bytes",
                "46000000000",
                "--assignment",
                str(directory / f"SCIENTIFIC_R{rotation}.json"),
            ]
            before = int(
                read(fit_root(rotation) / "CHECKPOINT_RECEIPT.json").get("completed_updates", 0)
            )
            with (directory / f"SCIENTIFIC_R{rotation}.log").open("a") as log:
                child = subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            children[rotation] = child
            (directory / "SCIENTIFIC_PID").write_text(str(child.pid) + "\n")
            return child.pid, before

        status("WAIT_DUAL_RESULT")
        while not (OPS / "DUAL_6MIN/RESULT.json").exists():
            check()
            time.sleep(2)
        pauses = wait_paused((4, 3))
        starts = {rotation: start(rotation) for rotation in (4, 3, 2)}
        status("TRIPLE_WARMUP", starts=starts, dual_durable_updates=pauses)
        while True:
            check()
            ready = True
            for rotation, (pid, before) in starts.items():
                if not live(pid):
                    raise RuntimeError(f"Triple worker {rotation}/{pid} exited during warmup")
                progress = read(fit_root(rotation) / "PROGRESS.json")
                ready &= (
                    progress.get("pid") == pid
                    and progress.get("completed_updates", 0) >= before + 16
                )
            if ready:
                queried = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                observed = {int(value) for value in queried.stdout.split()}
                if observed == {value[0] for value in starts.values()}:
                    break
            time.sleep(2)
        command = [PYTHON, str(sampler), "--output", str(OPS / "TRIPLE_6MIN"), "--seconds", "360"]
        for rotation in (4, 3, 2):
            command += ["--fit-dir", str(fit_root(rotation)), "--stop-run", str(run_root(rotation))]
        status("MEASURING_TRIPLE", starts=starts)
        with (OPS / "TRIPLE_6MIN.log").open("w") as log:
            subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=450
            )
        final = wait_paused((4, 3, 2))
        status(
            "TRIPLE_COMPLETE_ALL_DURABLE",
            completed_updates=final,
            selection_pending_evidence_review=True,
        )


if __name__ == "__main__":
    main()
