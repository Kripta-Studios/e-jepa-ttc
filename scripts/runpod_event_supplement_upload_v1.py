"""Wait for producer staging, then transfer and admit the small downstream supplement."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ARCHIVE_SHA = "0c26ca271a484beb892524d175643720e43c8c8e45fbaaf79abfbf9cec7422c0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    ssh = [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-i", str(args.key), "-p", str(args.port), args.host,
    ]
    while True:
        try:
            ready = subprocess.run(
                ssh + ["test -f /opt/ejepa_event_bundle.SEAL.json"],
                check=False,
                timeout=30,
            )
            if ready.returncode == 0:
                break
        except (OSError, subprocess.TimeoutExpired):
            pass
        time.sleep(30)
    remote_pending = "/workspace/ejepa_event_operations/EVENT_DOWNSTREAM_SUPPLEMENT.tar.partial"
    subprocess.run(
        [
            "scp", "-P", str(args.port), "-i", str(args.key),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            str(args.archive.resolve(strict=True)), f"{args.host}:{remote_pending}",
        ],
        check=True,
        timeout=600,
    )
    command = [
        "set", "-eu;",
        "printf", f"'%s  %s\\n' {ARCHIVE_SHA} {remote_pending}", "|", "sha256sum", "-c", "-;",
        "mkdir", "-p", "/workspace/ejepa_event_operations/supplement;",
        "/opt/ejepa_env/bin/python",
        "/workspace/ejepa_event_operations/event_downstream_runtime_v4/scripts/runpod_event_supplement_v1.py",
        "--archive", remote_pending,
        "--root", "/opt/ejepa_event_supplement",
        "--admission",
        "/workspace/ejepa_event_operations/supplement/EVENT_DOWNSTREAM_SUPPLEMENTAL_ADMISSION.json",
    ]
    result = subprocess.run(
        ssh + [" ".join(command)], capture_output=True, text=True, check=True, timeout=600
    )
    value = {
        "schema": "ejepa_event_supplement_upload_v1",
        "status": "VERIFIED_REMOTE",
        "completed_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": ARCHIVE_SHA,
        "remote_output": result.stdout,
    }
    args.status.parent.mkdir(parents=True, exist_ok=True)
    pending = args.status.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, args.status)


if __name__ == "__main__":
    main()
