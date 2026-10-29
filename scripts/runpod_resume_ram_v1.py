"""Resume the measured three-worker configuration after durable benchmark stops."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime

from runpod_concurrency_sequence_v1 import OPS, PYTHON, durable, fit_root, live, read, run_root


def main() -> None:
    import fcntl

    with (OPS / "CONCURRENCY_SEQUENCE.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = read(OPS / "CONCURRENCY_SEQUENCE_STATUS.json")
        if status.get("phase") != "TRIPLE_COMPLETE_ALL_DURABLE":
            raise ValueError("Triple experiment has not durably completed")
        deadline = datetime.fromisoformat(
            read(OPS / "USER_BUDGET_24USD.json")["execution_deadline_utc"]
        )
        if datetime.now(UTC) >= deadline or (OPS / "STOP_CONCURRENCY_SEQUENCE").exists():
            raise ValueError("Authorization stopped or expired")
        for rotation in (4, 3, 2):
            folder = fit_root(rotation)
            if live(read(folder / "PROGRESS.json").get("pid")):
                raise ValueError(f"Worker already alive: {rotation}")
            if not durable(
                read(folder / "CHECKPOINT_RECEIPT.json"), read(folder / "UPDATE_JOURNAL.json")
            ):
                raise ValueError(f"Not durably paused: {rotation}")
            marker = run_root(rotation) / "STOP_REQUEST"
            if not marker.read_text().startswith(
                "User authorized durable pause for actual cgroup RAM policy"
            ):
                raise ValueError(f"Non-benchmark stop: {rotation}")
        env = dict(os.environ)
        repo = "/workspace/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement"
        env.update(PYTHONPATH=f"{repo}/src:{repo}", PYTHONDONTWRITEBYTECODE="1")
        starts = {}
        for rotation in (4, 3, 2):
            root = run_root(rotation)
            marker = root / "STOP_REQUEST"
            os.replace(marker, marker.with_name(f"STOP_AFTER_RAM_POLICY_{time.time_ns()}.txt"))
            directory = OPS if rotation == 4 else OPS / f"r{rotation}"
            entrypoint = OPS / "runpod_ram_policy_overlay_v1.py"
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
            with (directory / f"SCIENTIFIC_R{rotation}.log").open("a") as log:
                child = subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            (directory / "SCIENTIFIC_PID").write_text(str(child.pid) + "\n")
            starts[str(rotation)] = {
                "pid": child.pid,
                "durable_start": read(fit_root(rotation) / "CHECKPOINT_RECEIPT.json")[
                    "completed_updates"
                ],
            }
        result = {
            "status": "RAM_POLICY_WORKERS_STARTED",
            "utc": datetime.now(UTC).isoformat(),
            "starts": starts,
            "scientific_recipe_changed": False,
            "billing_stop_implied": False,
        }
        (OPS / "RAM_POLICY_RESTART.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
