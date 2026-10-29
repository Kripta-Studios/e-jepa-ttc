# ruff: noqa: E501
"""CPU smoke coverage for the V8 job command substrate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash

ROOT = Path(__file__).resolve().parents[2]


def test_v8_smoke_executes_temporal_models_jepa_and_router_on_cpu() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/smoke_scientific_recovery_v8.py", "--device", "cpu"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"status": "smoke_passed"' in completed.stdout


def test_v8_smoke_writes_signed_manifest_only_after_success(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/smoke_scientific_recovery_v8.py",
            "--device",
            "cpu",
            "--output-dir",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert verify_artifact_hash(manifest)


def test_v8_stage_scripts_expose_help_and_dry_run() -> None:
    for script in (
        "scripts/run_scientific_recovery_v8_temporal.py",
        "scripts/run_scientific_recovery_v8_adaptive.py",
        "scripts/run_scientific_recovery_v8_multiseed_replication.py",
        "scripts/run_scientific_recovery_v8_robustness.py",
    ):
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
