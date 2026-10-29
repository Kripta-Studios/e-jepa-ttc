"""Prepare one newly assigned C2F fit and its eight-update technical resume proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    rotation = parser.parse_args().rotation
    ops = Path("/workspace/ejepa_operations")
    output = ops / f"r{rotation}_c2f"
    output.mkdir(exist_ok=True)
    run = Path(f"/workspace/ejepa_technical/r{rotation}_c2f_v3")
    budget = json.loads((ops / "USER_BUDGET_24USD.json").read_text())
    if datetime.now(UTC) >= datetime.fromisoformat(budget["execution_deadline_utc"]):
        raise ValueError("Total authorization expired")
    env = dict(os.environ)
    repo = "/workspace/ejepa_bundle/workspace/e-jepa-ttc-v18-joint-measurement"
    env.update(PYTHONPATH=f"{repo}/src:{repo}", PYTHONDONTWRITEBYTECODE="1")
    base = ["/opt/ejepa_env/bin/python", "-u", str(ops / "runpod_technical_overlay_v3.py")]
    flags = [
        "--bundle",
        "/workspace/ejepa_bundle",
        "--run",
        str(run),
        "--rotation",
        str(rotation),
        "--fit",
        "R_C2F",
        "--budget-bytes",
        "60999999488",
    ]
    with (output / "TECHNICAL_PREFLIGHT.log").open("w") as log:
        subprocess.run(
            base + ["preflight", *flags],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )
    report = json.loads((run / f"PREFLIGHT_R{rotation}_R_C2F.json").read_text())
    if report.get("status") != "PASS" or report.get("optimizer_updates") != 0:
        raise ValueError("Zero-update preflight failed")
    assignment = {
        "schema": "ejepa_cloud_assignment_v1",
        "execution_authorized": True,
        "tasks": [{"rotation": rotation, "fit": "R_C2F"}],
        "migration_binding_sha256": report["migration_binding_sha256"],
        "maximum_compute_usd": budget["maximum_total_usd"],
        "execution_deadline_utc": budget["execution_deadline_utc"],
        "technical_update_cap": 8,
        "scientific_update_cap": 0,
        "shared_total_budget_not_additive": True,
    }
    assignment_path = output / "TECHNICAL_ASSIGNMENT.json"
    assignment_path.write_text(json.dumps(assignment, indent=2) + "\n")
    with (output / "TECHNICAL_PROBE.log").open("w") as log:
        subprocess.run(
            base + ["resume-probe", *flags, "--assignment", str(assignment_path)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )
    proof_path = run / "technical_resume_probe/R_C2F/RESULT.json"
    proof = json.loads(proof_path.read_text())
    if proof.get("status") != "PASS":
        raise ValueError("Resume probe did not pass")
    result = {
        "status": "TECHNICAL_READY",
        "rotation": rotation,
        "fit": "R_C2F",
        "proof_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        "migration_binding_sha256": report["migration_binding_sha256"],
        "scientific_optimizer_updates": 0,
        "utc": datetime.now(UTC).isoformat(),
    }
    (output / "TECHNICAL_STATUS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
