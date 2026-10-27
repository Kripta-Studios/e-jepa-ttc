"""Wait for the existing transfer and run exactly one zero-update technical preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_MANIFEST = "8326ce1d265430369815d520680fe5f9fd70b29e49501e303a8979e1bf86b298"


def process_start(pid: int) -> str | None:
    """Identify the existing tar process without trusting a reused PID."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main() -> int:
    """Leave an auditable status and never authorize or invoke optimizer work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar-pid", type=int, required=True)
    parser.add_argument("--tar-start-ticks", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    root = Path("/workspace/ejepa_operations")
    status = root / "ZERO_UPDATE_PREFLIGHT_STATUS.json"
    deadline = time.monotonic() + args.timeout_seconds

    def emit(state: str, **values: object) -> None:
        record = {
            "schema": "ejepa_zero_update_preflight_orchestration_v3",
            "status": state,
            "utc": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "optimizer_updates_authorized": 0,
            **values,
        }
        pending = status.with_suffix(".pending")
        pending.write_text(json.dumps(record, indent=2) + "\n")
        os.replace(pending, status)

    try:
        while process_start(args.tar_pid) == args.tar_start_ticks:
            if time.monotonic() >= deadline:
                raise TimeoutError("Existing tar transfer did not finish within the bounded wait")
            emit("WAITING_EXISTING_TRANSFER", tar_pid=args.tar_pid)
            time.sleep(15)
        bundle = Path("/workspace/ejepa_bundle")
        observed = hashlib.sha256((bundle / "BUNDLE_MANIFEST.json").read_bytes()).hexdigest()
        if observed != EXPECTED_MANIFEST:
            raise ValueError("Transferred manifest differs from the frozen local tar manifest")
        repo = bundle / "workspace/e-jepa-ttc-v18-joint-measurement"
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{repo / 'src'}:{repo}"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        command = [
            "/workspace/ejepa_env/bin/python",
            str(root / "runpod_technical_overlay_v3.py"),
            "preflight",
            "--bundle",
            str(bundle),
            "--run",
            "/workspace/ejepa_technical/r4_rgb_v3",
            "--rotation",
            "4",
            "--fit",
            "R_A5",
            "--budget-bytes",
            "46000000000",
        ]
        emit("RUNNING_ZERO_UPDATE_PREFLIGHT", command=command)
        with (root / "ZERO_UPDATE_PREFLIGHT.log").open("w") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False
            )
        if result.returncode:
            raise RuntimeError(
                f"Preflight exit {result.returncode}; inspect ZERO_UPDATE_PREFLIGHT.log"
            )
        report = json.loads(
            Path("/workspace/ejepa_technical/r4_rgb_v3/PREFLIGHT_R4_R_A5.json").read_text()
        )
        if report.get("status") != "PASS" or report.get("optimizer_updates") != 0:
            raise ValueError("Preflight did not confirm PASS with zero optimizer updates")
        emit(
            "PASS",
            migration_binding_sha256=report["migration_binding_sha256"],
            input_parity=report["input_parity"],
            scientific_training_started=False,
        )
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        emit("FAILED", error=str(error), scientific_training_started=False)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
