from __future__ import annotations

from pathlib import Path


def test_orchestrator_has_gate_branch_and_always_runs_x3() -> None:
    repo = Path(__file__).resolve().parents[2]
    text = (repo / "scripts/run_scientific_recovery_v9_stage61_stage62.ps1").read_text(
        encoding="utf-8"
    )
    assert "if (-not $stage61.gate_passed)" in text
    assert "audit_scientific_recovery_v9_x3_feasibility.py" in text
    assert "run_scientific_recovery_v9_stage62.py" in text
    assert "push" not in text.lower()
