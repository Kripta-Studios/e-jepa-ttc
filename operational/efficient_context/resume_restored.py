"""Execute the authorized scientific queue as its pinned TRAIN restoration becomes ready."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .common import ROOT, Campaign, read
from .restore_data import atomic_json, plan


def missing(c: Campaign, rows: list[dict]) -> list[str]:
    """Accept only fully SHA-verified receipts whose present file still matches its stat."""
    absent = []
    for row in rows:
        receipt = c.out / "data_recovery/files" / (row["sequence_id"] + ".json")
        raw = c.raw / row["sequence_id"] / "events.h5"
        if not receipt.exists() or not raw.is_file():
            absent.append(str(raw))
            continue
        value = read(receipt)
        stat = raw.stat()
        if not (
            value.get("status") == "VERIFIED"
            and value["sha256"] == row["sha256"]
            and value["bytes"] == stat.st_size == row["bytes"]
            and value["mtime_ns"] == stat.st_mtime_ns
        ):
            absent.append(str(raw))
    return absent


def wait_sources(c: Campaign, rows: list[dict], stage: str) -> bool:
    """Wait in short polls; a stopped restoration is an exact dependency, never a result."""
    import psutil

    previous = None
    while True:
        absent = missing(c, rows)
        atomic_json(
            c.out / "data_recovery/FOLLOWUP_PROGRESS.json",
            {
                "status": "READY" if not absent else "WAITING_RESTORATION",
                "stage": stage,
                "missing": absent,
                "observed_utc": datetime.now(UTC).isoformat(),
                "optimizer_updates_from_waiting": 0,
            },
        )
        if not absent:
            return True
        if len(absent) != previous:
            print("WAIT_RESTORATION", stage, len(absent), flush=True)
            previous = len(absent)
        live = False
        for process in psutil.process_iter(["cmdline"]):
            line = " ".join(process.info["cmdline"] or [])
            if "operational.efficient_context.restore_data restore" in line:
                try:
                    live |= Path(process.cwd()).resolve() == ROOT
                except psutil.Error:
                    pass
        if not live:
            atomic_json(
                c.out / "data_recovery/FOLLOWUP_PROGRESS.json",
                {
                    "status": "BLOCKED_DEPENDENCY",
                    "stage": stage,
                    "missing": absent,
                    "dependency": "primary restoration stopped before full SHA verification",
                    "scientific_negative": False,
                    "resume": "python -m operational.efficient_context.restore_data restore",
                },
            )
            return False
        time.sleep(30)


def execute(c: Campaign, stage: str) -> int:
    """One foreground scientific queue; output and return codes are durable."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log = c.out / "data_recovery" / f"{timestamp}_FOLLOWUP_{stage}.txt"
    command = [
        sys.executable,
        "-m",
        "operational.efficient_context.queue",
        "all",
        "--protocol",
        str(c.config_path),
        "--resume",
    ]
    print("FOLLOWUP_EXECUTE", stage, flush=True)
    with log.open("xb") as stream:
        run = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    atomic_json(
        c.out / "data_recovery" / f"FOLLOWUP_{stage}_RECEIPT.json",
        {
            "command": command,
            "returncode": run.returncode,
            "log": str(log),
            "end_utc": datetime.now(UTC).isoformat(),
            "scientific_negative_for_dependency": False,
        },
    )
    print("FOLLOWUP_FINISHED", stage, run.returncode, flush=True)
    return run.returncode


def main() -> int:
    """Advance E1 after its23 sources and E3 after all31, with no new experiment arms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "4"
    c = Campaign(args.protocol)
    c.freeze()
    rows = plan(c)["files"]
    if not wait_sources(c, [r for r in rows if r["profile_required"]], "E1"):
        return 3
    execute(c, "E1")
    if not wait_sources(c, rows, "E3"):
        return 3
    return execute(c, "E3")


if __name__ == "__main__":
    raise SystemExit(main())
