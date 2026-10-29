"""Close operational failures without interpreting them as scientific negatives."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import compute_file_hash


def failure_category(error: BaseException) -> str:
    """Conservative categories: a failed integrity check never enables fallback."""
    if isinstance(error, TimeoutError):
        return "RESOURCE_BLOCKED"
    if isinstance(error, FileNotFoundError):
        return "INPUT_MISSING"
    if isinstance(error, FloatingPointError):
        return "NUMERICAL_FAILURE"
    if isinstance(error, (ValueError, KeyError, TypeError)):
        return "INTEGRITY_BLOCKED"
    if isinstance(error, KeyboardInterrupt):
        return "INTERRUPTED"
    return "TECHNICAL_FAILURE"


def close_campaign_failure(root: Path, error: BaseException) -> dict[str, Any]:
    """Persist an inventory of actual partial outputs while the writer lock is held."""
    ledger = root / "RUN_LEDGER.jsonl"
    phase: object = "before_first_ledger_record"
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            try:
                phase = json.loads(line).get("state", phase)
            except (ValueError, AttributeError):
                phase = "ledger_incomplete"
    inventory = []
    for pattern in ("**/checkpoint*.pt*", "**/*manifest.json", "**/*_oof.csv"):
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                inventory.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": compute_file_hash(str(path)),
                        "accepted_as_valid": False,
                    }
                )
    result = {
        "status": failure_category(error),
        "failure_phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "scientific_negative": False,
        "fallback_authorized_by_failure": False,
        "automatic_next_action": "none",
        "observed_artifacts": inventory,
        "time_ns": time.time_ns(),
    }
    destination = root / "CAMPAIGN_RESULT.json"
    if destination.exists():
        destination.replace(root / f"PRIOR_CAMPAIGN_RESULT_{time.time_ns()}.json")
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return result
