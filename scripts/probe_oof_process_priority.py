"""Measure reversible Windows process scheduling changes on a live OOF producer.

No model, data, checkpoint, global power setting, or scientific source is modified.
The original priority is restored even when a measurement fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def progress(path: Path, pid: int) -> dict:
    """Read a current progress record and reject a changed producer."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["pid"] != pid:
        raise RuntimeError("Producer changed during the measurement")
    return value


def main() -> None:
    """Run an ABBA scheduling probe, retaining every observation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=90)
    args = parser.parse_args()
    first = json.loads(args.progress.read_text(encoding="utf-8"))
    process = psutil.Process(first["pid"])
    if "operational.scientific_oof_isolated" not in process.cmdline():
        raise RuntimeError("Expected the isolated OOF producer")
    original = process.nice()
    rows, cases = [], []
    report = {"scientific_code_changed": False, "cases": cases, "samples": rows}
    try:
        for name, priority in (
            ("baseline_A", original),
            ("above_normal_A", psutil.ABOVE_NORMAL_PRIORITY_CLASS),
            ("above_normal_B", psutil.ABOVE_NORMAL_PRIORITY_CLASS),
            ("baseline_B", original),
        ):
            process.nice(priority)
            time.sleep(3)
            start = progress(args.progress, process.pid)
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                sample = progress(args.progress, process.pid)
                gpu = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,temperature.gpu,power.draw,"
                        "clocks.sm,clocks.mem,clocks_event_reasons.active",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                ).stdout.strip()
                rows.append(
                    {
                        "case": name,
                        "time": time.time(),
                        "progress": sample,
                        "gpu": gpu,
                        "priority": int(process.nice()),
                    }
                )
                time.sleep(5)
            end = progress(args.progress, process.pid)
            updates = end["completed_updates"] - start["completed_updates"]
            elapsed = end["seconds_this_call"] - start["seconds_this_call"]
            if updates <= 0 or elapsed <= 0:
                raise RuntimeError("No measurable producer progress")
            cases.append(
                {
                    "name": name,
                    "updates": updates,
                    "seconds": elapsed,
                    "updates_per_minute": updates * 60 / elapsed,
                    "start": start,
                    "end": end,
                }
            )
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(cases[-1]), flush=True)
    finally:
        if process.is_running():
            process.nice(original)
            report["priority_restored"] = process.nice() == original
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
