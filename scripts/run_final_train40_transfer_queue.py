"""Run the two frozen local inference routes sequentially, retaining logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "QUERY_MANIFEST.json"
    routes = [
        (
            "garl",
            [
                "scripts/final_train40_transfer_garl.py",
                "--manifest",
                str(manifest),
                "--registration",
                str(output / "COHORT_REGISTRATION.json"),
                "--output",
                str(output / "garl"),
                "--device",
                "cuda",
            ],
        ),
        (
            "final_train40",
            [
                "scripts/final_train40_transfer.py",
                "--candidate",
                str(args.candidate.resolve()),
                "--manifest",
                str(manifest),
                "--output",
                str(output / "final_train40"),
                "--device",
                "cuda",
            ],
        ),
    ]
    state = {
        "schema": "local_frozen_transfer_queue_v1",
        "started": time.time(),
        "optimizer_updates": 0,
        "external_submission": False,
        "routes": {},
    }

    def save() -> None:
        path = output / "QUEUE_STATUS.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root)])
    for route, command in routes:
        record = {"status": "RUNNING", "started": time.time(), "command": command}
        state["routes"][route] = record
        state["status"] = "RUNNING"
        with (output / f"{route}_run.log").open("a", encoding="utf-8") as log:
            log.write(f"\nQUEUE_START {time.time()}\n")
            log.flush()
            process = subprocess.Popen(
                [sys.executable, "-u", *command],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            record["pid"] = process.pid
            save()
            code = process.wait()
        record.update(
            status="COMPLETE" if code == 0 else "FAILED", ended=time.time(), exit_code=code
        )
        if code:
            state["status"] = "FAILED"
            save()
            raise SystemExit(code)
        save()
    state.update(status="INFERENCE_COMPLETE_SCORING_PENDING", ended=time.time())
    save()


if __name__ == "__main__":
    main()
