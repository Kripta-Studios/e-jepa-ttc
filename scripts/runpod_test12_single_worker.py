"""One public H5 download and one CPU parity probe under explicit resource guards."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil
from resource_reader import memory_snapshot

ROOT = Path("/workspace/ejepa_test12_probe")
DEADLINE = 1793145600.0
RAW_BYTES = 12915647068
QUOTA = 600_000_000_000
RESERVE = 10_000_000_000 + 12 * 1024**3


def publish(state: dict) -> None:
    """Write durable status without interfering with training journals."""
    state.update(timestamp=time.time(), pid=os.getpid(), optimizer_updates=0)
    temporary = ROOT / "STATUS.pending.json"
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, ROOT / "STATUS.json")
    print(json.dumps(state), flush=True)


def guard() -> None:
    """Honor authorization deadline and reserve memory for existing workers."""
    if time.time() >= DEADLINE or (ROOT / "STOP_REQUEST").exists():
        raise RuntimeError("Deadline or manual stop")
    available = psutil.virtual_memory().available
    headroom, _ = memory_snapshot(
        Path("/sys/fs/cgroup"), Path("/proc/self/cgroup").read_text(), 2**63 - 1
    )
    available = min(available, headroom)
    if available < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB available RAM; training keeps priority")


def run_child(command: list[str], stage: str, env: dict[str, str]) -> None:
    """Stop only this worker's child if a guard fails."""
    guard()
    child = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        while child.poll() is None:
            guard()
            publish({"status": stage, "child_pid": child.pid})
            time.sleep(10)
        if child.returncode:
            raise RuntimeError(f"{stage} exited {child.returncode}")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def main() -> None:
    """Download exactly the selected H5, then run one SHA-bound tensor check."""
    import fcntl

    with (ROOT / "WORKER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            guard()
            snapshot = json.loads(
                Path("/workspace/ejepa_event_runs/EVENT_VOLUME_SNAPSHOT.json").read_text()
            )
            if not 0 <= time.time() - float(snapshot["observed_timestamp"]) <= 120:
                raise RuntimeError("Volume quota snapshot is stale")
            if int(snapshot["quota_bytes"]) != QUOTA:
                raise RuntimeError("Volume quota snapshot belongs to another allocation")
            used = int(snapshot["used_bytes"])
            if used + RAW_BYTES + RESERVE > QUOTA:
                raise RuntimeError(f"Persistent quota reserve would be breached: {used}")
            env = dict(
                os.environ,
                HF_HUB_DISABLE_XET="1",
                HF_HUB_ENABLE_HF_TRANSFER="0",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
            )
            plan = json.loads((ROOT / "PLAN.json").read_text())
            run_child(
                ["/opt/ejepa_env/bin/hf", *plan["download_argv"][1:]], "DOWNLOADING_ONE_H5", env
            )
            argv = json.loads((ROOT / "PROBE_ARGV.json").read_text())
            run_child([*argv, "--execute"], "VERIFYING_AND_PREPROCESSING_ONE_QUERY", env)
            publish({"status": "COMPLETE_EXACT_SINGLE_QUERY_PARITY"})
        except Exception as error:
            publish({"status": "STOPPED_REQUIRES_REVIEW", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
