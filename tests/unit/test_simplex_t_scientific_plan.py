"""The generated scientific tail retains controls, late gates and delivery verification."""

import copy
import runpy
from pathlib import Path

import pytest


def test_plan_binds_all_phases_and_resource_report_root():
    work = Path(__file__).resolve().parents[2]
    build = runpy.run_path(str(work / "scripts/prepare_simplex_t_scientific_plan.py"))["build_plan"]
    prefix = dict(schema="simplex_t_orchestration_plan_v1", steps=[dict(id="prerequisites")])
    saved = copy.deepcopy(prefix)
    root = work / "artifacts/simplex_t/scientific_fixture"
    plan = build(
        work=work,
        prefix=prefix,
        freeze_launch=root / "freeze_launch.json",
        freeze_hash="a" * 64,
        campaign_root=root,
        run_root=root / "logs",
        reconciliation=root / "accounting.json",
        reconciliation_hash="b" * 64,
        analysis_commit="c" * 40,
        other_reserved=0,
        own_reserved=8_388_608,
    )
    assert prefix == saved
    assert [step["id"] for step in plan["steps"]] == [
        "prerequisites",
        "scientific_freeze",
        "T2",
        "T4",
        "T3",
        "T5",
        "T6",
    ]
    assert plan["required_run_root"] == str(root / "logs")
    for step in plan["steps"][1:]:
        for mode in ("run", "resume", "verify"):
            command = step[mode]
            assert len(command["sha256"]) == 64
            assert all(isinstance(value, str) for value in command["arguments"])
            assert ("--verify-only" in command["arguments"]) == (mode == "verify")
            if step["id"] != "scientific_freeze":
                assert ("--resume" in command["arguments"]) == (mode == "resume")
            assert command.get("attempt_report", False) == (
                step["id"] in {"T2", "T3", "T4", "T5"} and mode != "verify"
            )
    assert plan["steps"][-1]["verify"]["script"] == "scripts/run_simplex_t_materialized_delivery.py"
    with pytest.raises(ValueError, match="already contains"):
        build(
            work=work,
            prefix=plan,
            freeze_launch=root / "freeze_launch.json",
            freeze_hash="a" * 64,
            campaign_root=root,
            run_root=root / "logs",
            reconciliation=root / "accounting.json",
            reconciliation_hash="b" * 64,
            analysis_commit="c" * 40,
            other_reserved=0,
            own_reserved=8_388_608,
        )
