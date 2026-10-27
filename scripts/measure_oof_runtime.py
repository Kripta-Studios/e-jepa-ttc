"""Read-only interval measurements of a live OOF producer and its GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> None:
    """Save progress deltas, wall-clock duration and raw hardware observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=300)
    args = parser.parse_args()
    samples = []
    start = time.monotonic()
    while True:
        value = json.loads(args.progress.read_text(encoding="utf-8"))
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw,"
                "clocks.sm,clocks.mem,clocks_event_reasons.active",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        samples.append(
            dict(unix=time.time(), elapsed=time.monotonic() - start, progress=value, gpu=gpu)
        )
        first = samples[0]["progress"]
        if value["pid"] != first["pid"]:
            raise RuntimeError("Producer restarted; interval is not comparable")
        elapsed = value["seconds_this_call"] - first["seconds_this_call"]
        updates = value["completed_updates"] - first["completed_updates"]
        report = dict(
            samples=samples,
            updates=updates,
            seconds=elapsed,
            updates_per_minute=updates * 60 / elapsed if elapsed > 0 else None,
        )
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if time.monotonic() - start >= args.seconds:
            print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
