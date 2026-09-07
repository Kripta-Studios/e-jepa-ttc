"""Synthetic evidence fixtures, without pytest subprocesses or optimizer work."""

import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.unit_qa_evidence import UNIT_QA_FILES, verify_companion_unit_qa


def fixture(work: Path, failure: str) -> tuple[Path, dict]:
    root = work / "artifacts/simplex_t/T0/unit"
    root.mkdir(parents=True)
    runner = work / "scripts/run_simplex_t_full_unit_qa.py"
    test = work / "tests/unit/test_simplex_fixture.py"
    for path in (runner, test):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic\n", encoding="utf-8")
    node = "tests/unit/test_simplex_fixture.py::test_fixture"
    reports = [
        {"nodeid": node, "when": phase, "outcome": "passed", "longrepr": None}
        for phase in ("setup", "call", "teardown")
    ]
    if failure == "missing_call":
        reports.pop(1)
    if failure == "skip":
        reports[1]["outcome"] = "skipped"
    if failure == "repeat":
        reports.extend(reports.copy())
    (root / "reports.jsonl").write_text(
        "\n".join(json.dumps(row) for row in reports), encoding="utf-8"
    )
    (root / "QA.xml").write_text("<testsuites/>", encoding="utf-8")
    counters = {"attempted_optimizer_updates": 20, "completed_optimizer_updates": 20}
    contract = {
        "device": "cpu",
        "threads": 4,
        "interop_threads": 2,
        "cuda_visible_devices": "-1",
        "reserved_optimizer_updates": 20,
        "operation_id": "fixture",
        "test_files": {str(test.relative_to(work)): sha256(test)},
        "code_files": {str(runner.relative_to(work)): sha256(runner)},
        "runner_sha256": sha256(runner),
    }
    result = {
        "status": "UNIT_QA_COMPLETED_NOT_SCIENTIFIC_ADMISSION",
        "exit_code": 0,
        "scientific_updates": 0,
        "failed_nodeids": [],
        "collected": 1,
        "reports_sha256": sha256(root / "reports.jsonl"),
        **counters,
    }
    if failure == "boolean":
        result["exit_code"] = False
    if failure == "updates":
        counters["completed_optimizer_updates"] = 19
    if failure == "source":
        runner.write_text("# changed\n", encoding="utf-8")
    if failure == "inventory":
        (test.parent / "test_simplex_new.py").write_text("# new\n", encoding="utf-8")
    for name, value in (
        ("CONTRACT.json", contract),
        ("RESULT.json", result),
        ("COLLECTED.json", {"nodeids": [node, node] if failure == "duplicate" else [node]}),
        ("UPDATE_PROGRESS.json", counters),
    ):
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    ledger = work / "artifacts/simplex_t/TECHNICAL_BUDGET.json"
    ledger.write_text(
        json.dumps(
            {
                "schema": "simplex_t_technical_budget_v1",
                "reservations": {} if failure == "budget" else {"fixture": 20},
            }
        ),
        encoding="utf-8",
    )
    options = dict(
        pins={name: sha256(root / name) for name in UNIT_QA_FILES},
        technical_ledger_sha256=sha256(ledger),
        resource_ok=lambda: True,
    )
    if failure == "bytes":
        (root / "QA.xml").write_text("changed", encoding="utf-8")
    return root, options


@pytest.mark.parametrize(
    "failure",
    [
        "",
        "missing_call",
        "skip",
        "repeat",
        "boolean",
        "updates",
        "source",
        "inventory",
        "duplicate",
        "budget",
        "bytes",
    ],
)
def test_saved_unit_qa(tmp_path: Path, failure: str):
    root, options = fixture(tmp_path, failure)
    if failure:
        with pytest.raises(ValueError):
            verify_companion_unit_qa(tmp_path, root, **options)
    else:
        result = verify_companion_unit_qa(tmp_path, root, **options)
        assert result["passed"] == 1
        assert result["optimizer_updates_executed"] == 0
        assert result["scientific_admission"] is False
