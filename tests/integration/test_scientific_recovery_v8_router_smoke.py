"""CPU-only integration contract for the prospective V8 nested router stage."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash

ROOT = Path(__file__).resolve().parents[2]


def test_nested_router_cli_has_a_fixture_smoke_that_cannot_be_mistaken_for_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "router_fixture"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_scientific_recovery_v8_nested_router.py",
            "--fixture-smoke",
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
    assert (output / "router_fold_fixture.json").is_file()
    predictions = pd.read_csv(output / "outer_dev_predictions.csv")
    assert predictions["support_ms"].gt(0.0).all()
    assert "fixture" in completed.stdout.lower()


def test_nested_router_cli_plans_real_execution_without_manual_artifact_paths() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_scientific_recovery_v8_nested_router.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "planned"


def test_router_stage_runner_plans_all_nested_expert_jobs_without_manual_paths() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_scientific_recovery_v8_nested_router.py",
            "--results-root",
            "artifacts/scientific_recovery_v8",
            "--device",
            "cpu",
            "--max-parallel",
            "2",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["status"] == "planned"
    assert len(plan["expert_jobs"]) == 24
    assert {job["role"] for job in plan["expert_jobs"]} == {"inner_oof", "outer_dev"}
    assert plan["sealed_evaluation"] == "closed"


def test_router_aggregate_refuses_fixture_evidence_in_real_mode(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_scientific_recovery_v8_nested_router.py",
            "--fixture-smoke",
            "--output-dir",
            str(fixture),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    result = fixture / "router_fold_fixture.json"
    aggregate = subprocess.run(
        [
            sys.executable,
            "scripts/aggregate_scientific_recovery_v8_router.py",
            "--fold-artifact",
            str(result),
            "--output-dir",
            str(tmp_path / "aggregate"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert aggregate.returncode == 2
    assert "fixture" in aggregate.stderr.lower()
    assert verify_artifact_hash(json.loads(result.read_text(encoding="utf-8")))
