"""Lightweight owner that starts one admitted EVENT queue when E_A5 completes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic(path: Path, value: dict[str, object]) -> None:
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-entrypoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--exclusion", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--raw-coverage", type=Path, required=True)
    parser.add_argument("--supplemental-admission", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    endpoint = args.run / "rotations" / str(args.rotation) / "fits/E_A5/ENDPOINT_ADMISSION.json"
    receipt_path = args.run / f"EVENT_QUEUE_OWNER_R{args.rotation}.json"
    while not endpoint.is_file():
        receipt: dict[str, object] = {
            "schema": "ejepa_event_queue_lightweight_owner_v1",
            "status": "WAITING_DEPENDENCY",
            "stage": "WAIT_E_A5",
            "rotation": args.rotation,
            "pid": os.getpid(),
            "updated_utc": datetime.now(UTC).isoformat(),
            "heavy_scientific_runtime_loaded": False,
        }
        receipt["receipt_sha256"] = _canonical(receipt)
        _atomic(receipt_path, receipt)
        time.sleep(args.poll_seconds)
    argv = [
        sys.executable,
        str(args.queue_entrypoint.resolve(strict=True)),
        "--bundle",
        str(args.bundle),
        "--run",
        str(args.run),
        "--rotation",
        str(args.rotation),
        "--assignment",
        str(args.assignment),
        "--budget",
        str(args.budget),
        "--exclusion",
        str(args.exclusion),
        "--raw-manifest",
        str(args.raw_manifest),
        "--raw-coverage",
        str(args.raw_coverage),
        "--supplemental-admission",
        str(args.supplemental_admission),
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
