"""Close only the registered passive collector and bind its final durable samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil


def close(output: Path, *, stop: bool) -> dict[str, Any]:
    """Require explicit --stop for signalling; never target a scientific worker."""
    output = output.resolve(strict=True)
    status = json.loads((output / "COLLECTOR_STATUS.json").read_text())
    samples = Path(status["sample_file"]).resolve(strict=True)
    if samples.parent != output or not samples.name.endswith("_samples.jsonl"):
        raise ValueError("Collector sample path escaped diagnostics root")
    prefix = samples.name.removesuffix("_samples.jsonl")
    metadata = json.loads((output / f"{prefix}_metadata.json").read_text())
    if metadata["pid"] != status["pid"]:
        raise ValueError("Collector metadata and status PID differ")
    try:
        process = psutil.Process(status["pid"])
    except psutil.NoSuchProcess:
        process = None
    if process and process.status() != psutil.STATUS_ZOMBIE:
        expected = "/workspace/ejepa_final_operations/runpod_final_diagnostics_telemetry.py"
        started = datetime.fromisoformat(metadata["started_utc"]).timestamp()
        if expected not in process.cmdline() or not -1 <= started - process.create_time() < 30:
            raise ValueError("Collector PID was reused or its launch identity differs")
        if not stop:
            raise RuntimeError("Collector still runs; explicit --stop required")
        process.send_signal(signal.SIGTERM)
        until = time.monotonic() + 20
        while time.monotonic() < until:
            try:
                if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(0.2)
        else:
            raise TimeoutError("Collector did not close; not publishing preservation completion")
    closure_path = output / f"{prefix}_closure.json"
    closure = json.loads(closure_path.read_text())
    if closure.get("status") not in ("SIGNALLED", "DEADLINE"):
        raise ValueError("Collector lacks a clean durable closure")
    files = []
    for path in sorted(output.glob("*_samples.jsonl")):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest})
    with samples.open() as stream:
        last = None
        for line in stream:
            last = json.loads(line)
    receipt = {
        "schema": "ejepa_passive_diagnostics_closed_v1",
        "status": "STOPPED",
        "captured_utc": datetime.now(UTC).isoformat(),
        "collector_pid": status["pid"],
        "collector_started_utc": metadata["started_utc"],
        "closure": closure,
        "last_sample_utc": last["utc"] if last else None,
        "sample_files": files,
        "scientific_processes_signalled": 0,
        "optimizer_updates": 0,
        "collector_script_sha256": metadata.get("script_sha256"),
    }
    temporary = output / "COLLECTOR_STOPPED.pending"
    with temporary.open("w") as stream:
        json.dump(receipt, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output / "COLLECTOR_STOPPED.json")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    print(json.dumps(close(args.output, stop=args.stop)))


if __name__ == "__main__":
    main()
