"""Check a completed Linux probe on Windows without resuming or changing its state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from operational.runpod_migration.inventory import digest
from operational.runpod_migration.probe import STATE_FIELDS, state_digest


def verify(root: Path, rotation: int) -> dict:
    """Verify both checkpoint bytes and every recorded numerical state digest."""
    import torch

    proof_path = root / "technical_resume_probe/R_A5/RESULT.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if (
        proof.get("status") != "PASS"
        or proof.get("rotation") != rotation
        or proof.get("technical_optimizer_updates") != 8
        or proof.get("scientific_optimizer_updates") != 0
    ):
        raise ValueError("Cloud proof is not a completed technical eight-update run")
    result = {
        "status": "PASS",
        "cloud_result_status": "PASS",
        "rotation": rotation,
        "resume_proof_sha256": digest(proof_path),
        "cross_hardware_numerical_continuity_claimed": False,
        "cases": {},
    }
    for case in ("uninterrupted", "restored"):
        checkpoint = (
            root
            / f"technical_resume_probe/R_A5/{case}/rotations/{rotation}/fits/R_A5"
            / "checkpoint_last.pt"
        )
        expected = proof["cases"][case]
        if digest(checkpoint) != expected["checkpoint_sha256"]:
            raise ValueError("Windows copy differs from the Linux checkpoint")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            saved["completed_updates"] != 4
            or saved["identity"]["technical"] is not True
            or saved["identity"]["rotation"] != rotation
        ):
            raise ValueError("Technical checkpoint identity/cursor differs")
        state = {key: state_digest(saved[key]) for key in STATE_FIELDS}
        if state != expected["state"]:
            raise ValueError("Windows-loaded numerical state differs from Linux proof")
        result["cases"][case] = {
            "checkpoint": str(checkpoint),
            "sha256": expected["checkpoint_sha256"],
            "torch_load_windows_cpu": "PASS",
            "all_numerical_state_hashes": "MATCH_REMOTE",
            "completed_updates": 4,
        }
    (root / "WINDOWS_CHECKPOINT_COMPATIBILITY.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.rotation), indent=2))
