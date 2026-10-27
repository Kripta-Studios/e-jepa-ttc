"""Transfer verified RGB return files to Pod2 without extraction."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operational.runpod_migration.pod2_rgb_handoff import (
    LocalPendingError,
    Pod2Handoff,
    atomic_json,
)
from operational.runpod_migration.rgb_return_coordinator import SingleInstanceLock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Pod2 handoff config must be a JSON object")
    lock = Path(str(config["local"]["lock_path"]))
    with SingleInstanceLock(lock):
        worker = Pod2Handoff(config)
        attempts = int(config.get("max_retries", 5))
        if attempts < 1:
            raise ValueError("Pod2 handoff max_retries must be positive")
        attempt = 0
        while True:
            try:
                value = worker.run()
                break
            except LocalPendingError as error:
                atomic_json(
                    Path(str(config["local"]["status_path"])),
                    {
                        "schema": "ejepa_rgb_pod2_handoff_status_v1",
                        "stage": "WAITING_LOCAL_VERIFICATION",
                        "reason": str(error),
                        "updated_utc": datetime.now(UTC).isoformat(),
                        "remote_deletion_performed": False,
                    },
                )
                time.sleep(60)
            except ConnectionError as error:
                attempt += 1
                atomic_json(
                    Path(str(config["local"]["status_path"])),
                    {
                        "schema": "ejepa_rgb_pod2_handoff_status_v1",
                        "stage": "RETRYABLE_CONNECTION_ERROR",
                        "attempt": attempt,
                        "error": str(error),
                        "updated_utc": datetime.now(UTC).isoformat(),
                        "remote_deletion_performed": False,
                    },
                )
                if attempt >= attempts:
                    raise
                time.sleep(float(config.get("retry_seconds", 10)))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
