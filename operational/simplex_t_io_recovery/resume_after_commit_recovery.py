"""Admit the pinned remaining queue only after the diagnosed commit shortage clears."""

from __future__ import annotations

import json
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path


def check_startup(metrics: dict[str, int]) -> None:
    """Require startup slack for historical CPU Torch imports and cached fits."""
    if metrics["available"] < 3 * 1024**3 or metrics["commit_headroom"] < 4 * 1024**3:
        raise InterruptedError(
            "Diagnosed recovery requires available RAM >=3GiB and Windows commit "
            "headroom >=4GiB before importing Torch; engine limits remain unchanged: "
            + json.dumps(metrics)
        )


def main() -> None:
    """Keep source pins, six recipes, original window and completed work intact."""
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from operational.simplex_t_closure.runtime import atomic_json, digest
    from operational.simplex_t_h16_replication.common import memory, record

    night = root / "artifacts/simplex_t/nocturnal_20261003"
    window = record(night / "WINDOW_AUTHORIZATION.json")
    if datetime.now(UTC) >= datetime.fromisoformat(window["deadline_utc"]):
        raise InterruptedError("Original ten-hour window expired; new explicit window required")
    metrics = memory()
    check_startup(metrics)
    atomic_json(
        night / "COMMIT_RECOVERY_PREFLIGHT.json",
        dict(
            resources=metrics,
            startup_commit_minimum_bytes=4 * 1024**3,
            source_sha256=digest(Path(__file__)),
            reason="Repeated post-import commit pauses; no scientific recipe change",
            optimizer_updates_for_preflight=0,
            original_deadline_utc=window["deadline_utc"],
        ),
    )
    runpy.run_module("operational.simplex_t_io_recovery.resume_remaining", run_name="__main__")


if __name__ == "__main__":
    main()
