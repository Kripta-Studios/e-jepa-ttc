"""Continue from trained endpoints to TRAIN predictions and a verified essential delivery."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from operational.efficient_context.common import ROOT, Lease
from operational.train40_system.contracts import read, verify_sources
from operational.train40_system.controller import active
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json


def run(output: Path, raw_root: Path, code_root: Path) -> None:
    """Keep training priority and preserve failed independent postprocessing for retry."""
    verify_sources(read(output / "DELIVERY_FREEZE.json"))
    deadline = datetime.fromisoformat(read(output / "AUTHORIZATION.json")["deadline_utc"])
    retry_after = 0.0
    while datetime.now(UTC) < deadline:
        if (output / "BUNDLE_VERIFICATION.json").exists():
            if read(output / "BUNDLE_VERIFICATION.json")["status"] == "PASSED":
                return
        complete = output / "TRAINING_QUEUE_COMPLETION.json"
        ready = complete.exists() and read(complete)["status"] == "COMPLETE"
        if not ready:
            atomic_json(
                output / "POST_QUEUE_PROGRESS.json",
                {
                    "status": "WAITING_FOR_TRAINED_ENDPOINTS",
                    "checked_utc": datetime.now(UTC).isoformat(),
                    "no_new_scientific_arms": True,
                    "optimizer_updates": 0,
                },
            )
        elif time.monotonic() >= retry_after and not any(
            active("operational.train40_system." + marker)
            for marker in (
                "engine",
                "heads",
                "history_features",
                "teacher",
                "garl_predictions",
                "training_diagnostics",
                "delivery --",
            )
        ):
            if not (output / "TRAIN_FIT_DIAGNOSTICS.json").exists():
                module, arguments, tag = "training_diagnostics", [], "own_predictions"
            elif not (output / "PUBLIC_GARL_PREDICTION_MANIFEST.json").exists():
                module, arguments, tag = (
                    "garl_predictions",
                    ["--raw-root", str(raw_root), "--code-root", str(code_root)],
                    "garl",
                )
            else:
                module, arguments, tag = "delivery", ["--bundle"], "bundle"
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHONUTF8": "1",
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "OPENBLAS_NUM_THREADS": "4",
                    "NUMEXPR_NUM_THREADS": "4",
                }
            )
            command = [
                sys.executable,
                "-m",
                "operational.train40_system." + module,
                "--output",
                str(output),
                *arguments,
            ]
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            with (output / "post_queue" / f"{tag}_{stamp}.log").open("ab") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                while process.poll() is None:
                    atomic_json(
                        output / "POST_QUEUE_PROGRESS.json",
                        {
                            "status": "RUNNING",
                            "task": tag,
                            "pid": process.pid,
                            "command": command,
                            "optimizer_updates": 0,
                            "checked_utc": datetime.now(UTC).isoformat(),
                        },
                    )
                    time.sleep(60)
            atomic_json(
                output / "post_queue" / f"{tag}_{stamp}.json",
                {
                    "exit_code": process.returncode,
                    "command": command,
                    "scientific_negative": False,
                    "work_preserved": True,
                    "optimizer_updates": 0,
                },
            )
            if process.returncode:
                retry_after = time.monotonic() + 1800
            elif tag == "garl" and not (output / "PUBLIC_GARL_PREDICTION_MANIFEST.json").exists():
                retry_after = time.monotonic() + 1800
        time.sleep(60)
    atomic_json(
        output / "POST_QUEUE_PROGRESS.json",
        {
            "status": "DEADLINE_STOP_PRESERVED",
            "scientific_negative": False,
            "exact_dependencies_remain_in_receipts": True,
            "optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()
    (args.output / "post_queue").mkdir(exist_ok=True)
    with Lease((args.output / "post_queue").resolve()):
        run(args.output.resolve(), args.raw_root.resolve(), args.code_root.resolve())
