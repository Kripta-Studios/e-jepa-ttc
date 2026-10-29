"""Reserve RGB downstream work through the frozen pipeline's producer dependency gate."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .inventory import digest, read


def verify_claim(run: Path, receipt_path: Path, rotation: int) -> dict[str, Any]:
    """Reject a released producer barrier or any existing local RGB head work."""
    receipt = read(receipt_path)
    if (
        receipt.get("schema") != "ejepa_cloud_rgb_dependency_exclusion_v1"
        or receipt.get("status") != "EXCLUDED_FROM_LOCAL_EXECUTION"
        or receipt.get("tasks") != [{"rotation": rotation, "fit": "RGB_DOWNSTREAM"}]
        or receipt.get("scientific_config_sha256") != digest(run / "SCIENTIFIC_CONFIG.json")
    ):
        raise ValueError("RGB downstream ownership is not bound to this task")
    for fit in ("R_A5", "R_C2F"):
        directory = run / f"rotations/{rotation}/fits/{fit}"
        marker = directory / "STOP_REQUEST"
        if digest(marker) != receipt["producer_stop_sha256"][fit]:
            raise ValueError("Effective producer dependency barrier changed")
        if (directory / "ENDPOINT_ADMISSION.json").exists():
            raise ValueError("Producer was published before downstream ownership handoff")
    for name in ("PAIR_R", "R_CTX"):
        root = run / f"rotations/{rotation}"
        directory = root / "fits" / name
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Local RGB downstream work already exists")
        if (root / "caches" / f"{name}.npz").exists():
            raise ValueError("Local RGB downstream cache already exists")
    return receipt


def reserve(run: Path, receipt_path: Path, rotation: int) -> dict[str, Any]:
    """Preserve real producer STOPs; never pretend that head-specific STOPs exist."""
    from operational.final_refit.continuation_queue import queue_lock
    from operational.final_refit.scientific_authority import load_config
    from operational.rgb_port.accounting import atomic_write_json

    if rotation not in {2, 3, 4}:
        raise ValueError("Only assigned rotations may reserve RGB downstream work")
    run = run.resolve(strict=True)
    load_config(run)
    with queue_lock(run / "CLOUD_RGB_DOWNSTREAM_OWNERSHIP.lock"):
        if receipt_path.exists():
            return verify_claim(run, receipt_path, rotation)
        value = {
            "schema": "ejepa_cloud_rgb_dependency_exclusion_v1",
            "status": "EXCLUDED_FROM_LOCAL_EXECUTION",
            "tasks": [{"rotation": rotation, "fit": "RGB_DOWNSTREAM"}],
            "scientific_config_sha256": digest(run / "SCIENTIFIC_CONFIG.json"),
            "producer_stop_sha256": {
                fit: digest(run / f"rotations/{rotation}/fits/{fit}/STOP_REQUEST")
                for fit in ("R_A5", "R_C2F")
            },
            "mechanism": (
                "frozen pipeline completes all four producers before any PAIR or CTX stage; "
                "both RGB producers remain effectively stopped locally"
            ),
            "producer_publication_forbidden_until_downstream_return_verified": True,
            "head_stop_files_claimed": False,
            "active_event_fit_interrupted": False,
            "implementation_sha256": digest(Path(__file__)),
            "created_utc": datetime.now(UTC).isoformat(),
        }
        # Validate the complete intended receipt before publishing it.
        temporary = receipt_path.with_suffix(".pending.json")
        atomic_write_json(temporary, value)
        try:
            verify_claim(run, temporary, rotation)
            atomic_write_json(receipt_path, value)
        finally:
            temporary.unlink(missing_ok=True)
        return verify_claim(run, receipt_path, rotation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    args = parser.parse_args()
    print(json.dumps(reserve(args.run, args.receipt, args.rotation), indent=2))
