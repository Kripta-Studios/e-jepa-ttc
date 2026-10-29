"""Measure six admitted workers after warmup without stopping any training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

OPS = Path("/workspace/ejepa_operations")
PIDS = {"A5_4": 24558, "A5_3": 24559, "A5_2": 24560, "C2F_2": 27445, "C2F_3": 27482, "C2F_4": 27519}


def main() -> None:
    sampler = OPS / "runpod_measure_window.py"
    expected = "8bb610974fdd72e6a3f84f45a7abdb986518eab5cdae63c6cb3948f1a5078af4"
    if hashlib.sha256(sampler.read_bytes()).hexdigest() != expected:
        raise ValueError("Sampler source differs from baseline")
    deadline = datetime.fromisoformat(
        json.loads((OPS / "USER_BUDGET_24USD.json").read_text())["execution_deadline_utc"]
    )
    status = OPS / "SIX_WINDOW_STATUS.json"
    fits = {}
    for name, _pid in PIDS.items():
        kind, rotation = name.split("_")
        run = f"r{rotation}_{'rgb' if kind == 'A5' else 'c2f'}_v3"
        fits[name] = Path(f"/workspace/ejepa_technical/{run}/rotations/{rotation}/fits/R_{kind}")
    until = time.monotonic() + 1800
    while True:
        if time.monotonic() > until or datetime.now(UTC) >= deadline:
            raise TimeoutError("Six-worker warmup did not finish within its bound")
        if (OPS / "STOP_SIX_WINDOW").exists():
            return
        ready = True
        observed = {}
        for name, pid in PIDS.items():
            if not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError(f"Expected worker exited: {name}/{pid}")
            path = fits[name] / "PROGRESS.json"
            value = json.loads(path.read_text()) if path.exists() else {}
            observed[name] = value.get("completed_updates", 0)
            ready &= value.get("pid") == pid and observed[name] >= 16
        if ready:
            gpu_pids = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                text=True,
            )
            ready = {int(value) for value in gpu_pids.split()} == set(PIDS.values())
        status.write_text(
            json.dumps(
                {
                    "phase": "WARMUP",
                    "utc": datetime.now(UTC).isoformat(),
                    "pids": PIDS,
                    "updates": observed,
                }
            )
            + "\n"
        )
        if ready:
            break
        time.sleep(3)
    output = OPS / "SIX_6MIN"
    command = [
        "/opt/ejepa_env/bin/python",
        str(sampler),
        "--output",
        str(output),
        "--seconds",
        "360",
    ]
    for folder in fits.values():
        command += ["--fit-dir", str(folder)]
    status.write_text(
        json.dumps(
            {
                "phase": "MEASURING_SIX_NO_STOP",
                "utc": datetime.now(UTC).isoformat(),
                "pids": PIDS,
                "stop_training_at_end": False,
            }
        )
        + "\n"
    )
    subprocess.run(command, check=True, timeout=450)
    status.write_text(
        json.dumps(
            {
                "phase": "SIX_MEASUREMENT_COMPLETE_TRAINING_CONTINUES",
                "utc": datetime.now(UTC).isoformat(),
                "pids": PIDS,
                "stop_training_at_end": False,
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
