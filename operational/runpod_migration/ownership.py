"""Durable future-fit exclusion using the frozen runner's actual stop contract."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .inventory import digest, read


def verify_claim(run: Path, receipt_path: Path) -> dict[str, Any]:
    """Reject removed markers, changed authority and conflicting local updates."""
    receipt = read(receipt_path)
    if (
        receipt.get("schema") != "ejepa_cloud_local_exclusion_v2"
        or receipt.get("status") != "EXCLUDED_FROM_LOCAL_EXECUTION"
        or receipt.get("tasks") != [{"rotation": 4, "fit": "R_A5"}]
        or receipt.get("scientific_config_sha256") != digest(run / "SCIENTIFIC_CONFIG.json")
    ):
        raise ValueError("Cloud exclusion receipt is not bound to this scientific task")
    marker = run / "rotations/4/fits/R_A5/STOP_REQUEST"
    if str(marker.resolve()) != receipt.get("stop_marker") or digest(marker) != receipt.get(
        "stop_marker_sha256"
    ):
        raise ValueError("Effective local stop marker is missing or changed")
    directory = marker.parent
    for name in ("CHECKPOINT_RECEIPT.json", "UPDATE_JOURNAL.json", "PROGRESS.json"):
        path = directory / name
        if path.exists():
            value = read(path)
            if value.get("completed_updates", 0) != 0 or value.get("pending_update_upper", 0) != 0:
                raise ValueError("Local fit has updates and cannot be assigned to cloud")
    return receipt


def reserve(run: Path, receipt_path: Path) -> dict[str, Any]:
    """Reserve only future R4/R_A5; no active stage, queue code or recipe is changed."""
    from operational.final_refit.continuation_queue import queue_lock
    from operational.rgb_port.accounting import atomic_write_json

    run = run.resolve(strict=True)
    directory = run / "rotations/4/fits/R_A5"
    directory.mkdir(parents=True, exist_ok=True)
    with queue_lock(directory / "ISOLATED_PRODUCER.lock"):
        if receipt_path.exists():
            return verify_claim(run, receipt_path)
        if any(
            (directory / name).exists()
            for name in (
                "STOP_REQUEST",
                "CHECKPOINT_RECEIPT.json",
                "CHECKPOINT_POINTER.json",
                "UPDATE_JOURNAL.json",
                "PROGRESS.json",
                "ENDPOINT_ADMISSION.json",
            )
        ):
            raise ValueError("Future fit is not untouched; refusing ambiguous ownership")
        marker = directory / "STOP_REQUEST"
        marker_value = {
            "schema": "ejepa_future_fit_cloud_stop_v1",
            "rotation": 4,
            "fit": "R_A5",
            "reason": "CLOUD_OWNERSHIP_WAIT_FOR_VERIFIED_RETURN",
            "scientific_config_sha256": digest(run / "SCIENTIFIC_CONFIG.json"),
            "created_utc": datetime.now(UTC).isoformat(),
            "remove_automatically": False,
        }
        # Publish effective exclusion before its receipt. A crash in between
        # leaves a blocked local fit, never an unprotected cloud ownership claim.
        atomic_write_json(marker, marker_value)
        receipt = {
            "schema": "ejepa_cloud_local_exclusion_v2",
            "status": "EXCLUDED_FROM_LOCAL_EXECUTION",
            "tasks": [{"rotation": 4, "fit": "R_A5"}],
            "scientific_config_sha256": marker_value["scientific_config_sha256"],
            "stop_marker": str(marker.resolve()),
            "stop_marker_sha256": digest(marker),
            "mechanism": "frozen scientific producer checks fit STOP_REQUEST before each update",
            "active_fit_interrupted": False,
            "queue_sources_modified": False,
            "original_scientific_sources_modified": False,
            "ownership_implementation_sha256": digest(Path(__file__)),
        }
        atomic_write_json(receipt_path, receipt)
        return verify_claim(run, receipt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reserve", "verify"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    from operational.final_refit.scientific_authority import load_config

    load_config(args.run)
    result = (
        reserve(args.run, args.receipt)
        if args.command == "reserve"
        else verify_claim(args.run, args.receipt)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
