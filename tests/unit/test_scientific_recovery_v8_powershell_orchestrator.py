"""Contracts for the fail-closed V8 PowerShell orchestration entrypoint."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_scientific_recovery_v8.ps1"
SIGNER = ROOT / "scripts" / "sign_scientific_recovery_v8_status.py"
EXPORT = ROOT / "scripts" / "export_scientific_recovery_v8_onnx.py"
PACKAGE = ROOT / "scripts" / "package_scientific_recovery_v8_evidence.py"


def test_orchestrator_declares_complete_stage_dag_and_sealed_monitoring() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    for stage in (
        "preflight",
        "autopsy",
        "router",
        "temporal",
        "adaptive",
        "jepa",
        "multiseed_replication",
        "robustness",
        "export",
        "package",
        "screen",
        "all",
    ):
        assert f"'{stage}'" in source
    assert "MaxParallel = 2" in source
    assert "EnableThreeWayConcurrency" in source
    assert "three-way V8 scheduling denied" in source
    assert "MonitorIntervalMinutes = 30" in source
    assert "artifacts/scientific_recovery_v8/monitor/TRAINING_STATUS.md" not in source
    assert "monitor/TRAINING_STATUS.md" in source
    assert "PublishTrainingStatus" in source
    assert "private test, EvTTC test and CodaBench remain sealed" in source
    assert "Start-Process" in source
    assert "-WindowStyle Hidden" in source
    assert "C1/adaptive is closed" in source
    assert "blocked_gate" in source
    assert "run_scientific_recovery_v8_nested_router.py" in source
    assert "run_scientific_recovery_v8_jepa_attribution.py" in source
    assert "Invoke-Seed7Aggregate" in source
    assert "Resolve-SignedCandidate" in source
    assert "Resolve-WinnerContract" in source
    assert "cache_manifest binding required for JEPA attribution" in source
    assert "torch.cuda.mem_get_info" in source


def test_live_monitor_does_not_write_the_tracked_status_document() -> None:
    tracked_status = ROOT / "docs" / "SCIENTIFIC_RECOVERY_V8_TRAINING_STATUS.md"
    template = tracked_status.read_text(encoding="utf-8")
    assert "live, signed monitor snapshot" in template
    assert "artifacts/scientific_recovery_v8/monitor/TRAINING_STATUS.md" in template
    source = SCRIPT.read_text(encoding="utf-8")
    monitor_body = source.split("function Write-MonitorStatus", maxsplit=1)[1].split(
        "function Publish-MonitorStatus", maxsplit=1
    )[0]
    assert "PublishedStatusMarkdown" not in monitor_body


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell 7 is unavailable")
def test_orchestrator_powershell_parser_and_preflight_dry_run() -> None:
    command = [
        "pwsh",
        "-NoProfile",
        "-File",
        str(SCRIPT),
        "-Stage",
        "preflight",
        "-DryRun",
        "-NoBackgroundMonitor",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "DRY RUN: uv" in completed.stdout


def test_status_signer_writes_a_verifiable_atomic_artifact(tmp_path: Path) -> None:
    payload = tmp_path / "unsigned.json"
    output = tmp_path / "state.json"
    payload.write_text(
        json.dumps(
            {
                "artifact_type": "scientific_recovery_v8_training_monitor_state_v1",
                "sealed_evaluation": True,
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(SIGNER), "--input", str(payload), "--output", str(output)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    state = json.loads(output.read_text(encoding="utf-8"))
    assert verify_artifact_hash(state)
    assert state["sealed_evaluation"] is True


@pytest.mark.parametrize("script", (EXPORT, PACKAGE))
def test_delivery_stage_scripts_are_directly_invocable(script: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
