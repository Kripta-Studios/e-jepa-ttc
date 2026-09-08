"""Bind the complete companion unit suite to its raw reports and current sources."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .lifecycle import TECHNICAL_UPDATE_CAP

UNIT_QA_FILES = frozenset(
    {
        "CONTRACT.json",
        "COLLECTED.json",
        "RESULT.json",
        "UPDATE_PROGRESS.json",
        "reports.jsonl",
        "QA.xml",
    }
)


def verify_companion_unit_qa(
    work: Path,
    root: Path,
    *,
    pins: dict[str, str],
    technical_ledger_sha256: str,
    resource_ok: Callable[[], bool],
) -> dict:
    """Reject stale, partial, skipped or edited evidence without running tests.

    This proves only the selected companion unit suite. It does not replace
    baseline-versus-new repository QA, real TRAIN/GPU replay, or source loading.
    Source changes require fresh evidence; a historical PASS is not promoted.
    """
    work, root = work.resolve(strict=True), root.resolve(strict=True)
    if not root.is_relative_to(work / "artifacts/simplex_t/T0") or set(pins) != UNIT_QA_FILES:
        raise ValueError("complete companion T0 unit evidence required")

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: saved unit QA verification")

    boundary()
    for name, digest in pins.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or path.stat().st_size > 33_554_432:
            raise ValueError("unit QA evidence path or size invalid")
        if sha256(path) != digest:
            raise ValueError("unit QA evidence bytes changed")

    def read(name: str) -> dict:
        return json.loads((root / name).read_text(encoding="utf-8"))

    contract, result = read("CONTRACT.json"), read("RESULT.json")
    counters = read("UPDATE_PROGRESS.json")
    if (
        contract.get("device") != "cpu"
        or type(contract.get("threads")) is not int
        or contract["threads"] != 4
        or type(contract.get("interop_threads")) is not int
        or contract["interop_threads"] != 2
        or contract.get("cuda_visible_devices") != "-1"
        or type(contract.get("reserved_optimizer_updates")) is not int
        or contract["reserved_optimizer_updates"] != 20
    ):
        raise ValueError("unit QA execution recipe differs")
    expected = {"attempted_optimizer_updates": 20, "completed_optimizer_updates": 20}
    if (
        counters != expected
        or any(type(counters.get(key)) is not int for key in expected)
        or any(
            type(result.get(key)) is not int or result[key] != value
            for key, value in expected.items()
        )
        or type(result.get("exit_code")) is not int
        or result["exit_code"] != 0
        or result.get("status") != "UNIT_QA_COMPLETED_NOT_SCIENTIFIC_ADMISSION"
        or type(result.get("scientific_updates")) is not int
        or result["scientific_updates"] != 0
        or result.get("failed_nodeids") != []
        or result.get("reports_sha256") != pins["reports.jsonl"]
    ):
        raise ValueError("unit QA completion or counted work differs")
    ledger_path = work / "artifacts/simplex_t/TECHNICAL_BUDGET.json"
    if ledger_path.stat().st_size > 1_048_576 or sha256(ledger_path) != technical_ledger_sha256:
        raise ValueError("technical ledger changed")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    reservations = ledger.get("reservations", {})
    if (
        ledger.get("schema") != "simplex_t_technical_budget_v1"
        or any(type(value) is not int or value < 1 for value in reservations.values())
        or sum(reservations.values()) > TECHNICAL_UPDATE_CAP
        or reservations.get(contract.get("operation_id")) != 20
    ):
        raise ValueError("unit QA operation missing from technical budget")
    nodeids = read("COLLECTED.json")["nodeids"]
    if (
        not isinstance(nodeids, list)
        or not nodeids
        or any(not isinstance(node, str) for node in nodeids)
        or len(set(nodeids)) != len(nodeids)
        or type(result.get("collected")) is not int
        or result["collected"] != len(nodeids)
    ):
        raise ValueError("unit QA collection incomplete or duplicated")
    reports: dict[str, list[dict]] = {node: [] for node in nodeids}
    for line in (root / "reports.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("nodeid") not in reports:
            raise ValueError("unit report not present in collection")
        reports[row["nodeid"]].append(row)
    for rows in reports.values():
        if [row.get("when") for row in rows] != ["setup", "call", "teardown"] or any(
            row.get("outcome") != "passed" or row.get("longrepr") is not None for row in rows
        ):
            raise ValueError("unit QA has missing, failed, skipped or repeated test phases")

    def source_pins(field: str, actual_paths: set[Path]) -> set[str]:
        declared = contract[field]
        normalized: dict[Path, str] = {}
        for name, digest in declared.items():
            relative = Path(name)
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                raise ValueError("unit QA source path escapes worktree")
            path = (work / relative).resolve(strict=True)
            if not path.is_relative_to(work) or path in normalized:
                raise ValueError("unit QA source paths escape or alias")
            normalized[path] = digest
        if set(normalized) != actual_paths:
            raise ValueError("unit QA does not cover current complete source/test inventory")
        for path, digest in normalized.items():
            boundary()
            if sha256(path) != digest:
                raise ValueError(f"unit QA source changed: {path.relative_to(work)}")
        return {path.relative_to(work).as_posix() for path in normalized}

    tests = source_pins(
        "test_files", {p.resolve() for p in (work / "tests/unit").glob("test_simplex*.py")}
    )
    represented = {Path(node.split("::", 1)[0]).as_posix() for node in nodeids}
    if represented != tests:
        raise ValueError("unit QA collection omits a selected test file")
    source_pins(
        "code_files",
        {p.resolve() for folder in ("src", "scripts") for p in (work / folder).rglob("*.py")},
    )
    if contract.get("runner_sha256") != sha256(work / "scripts/run_simplex_t_full_unit_qa.py"):
        raise ValueError("unit QA runner changed")
    boundary()
    return {
        "status": "SAVED_COMPANION_UNIT_QA_CURRENT_SOURCES_VERIFIED",
        "passed": len(nodeids),
        "test_files": len(tests),
        "verified_saved_technical_updates": 20,
        "optimizer_updates_executed": 0,
        "scientific_admission": False,
        "pins": pins,
    }
