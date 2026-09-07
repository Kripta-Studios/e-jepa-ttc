"""Reconstruct static failure comparisons from pinned diagnostics, not PASS flags."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


def verify_ruff_comparison(
    work: Path,
    root: Path,
    *,
    pins: dict[str, str],
    resource_ok: Callable[[], bool],
) -> dict:
    """Verify current whole-repository Ruff coverage and preserve baseline IDs.

    This is static QA only. It neither runs pytest nor establishes replay parity.
    Input report hashes must be independently pinned by scientific admission.
    """
    work, root = work.resolve(strict=True), root.resolve(strict=True)
    names = {"COMPARISON.json", "BASELINE_RUFF.json", "CURRENT_RUFF.json"}
    if set(pins) != names or not root.is_relative_to(work / "artifacts/simplex_t/T0"):
        raise ValueError("complete companion static comparison evidence required")

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: static QA evidence")

    records = {}
    for name, digest in pins.items():
        boundary()
        path = (root / name).resolve(strict=True)
        if (
            not path.is_relative_to(root)
            or path.stat().st_size > 16_777_216
            or sha256(path) != digest
        ):
            raise ValueError("static QA report bytes or path changed")
        records[name] = json.loads(path.read_text(encoding="utf-8"))
    report = records["COMPARISON.json"]
    if (
        report.get("baseline_commit") != "57b39cb2b9a5ec8378755f8f350634c007aa822e"
        or report.get("status") != "STATIC_FAILURE_IDS_COMPARED_NOT_FULL_QA_OR_SCIENTIFIC_ADMISSION"
        or report.get("position_sensitive_ids") is not True
        or type(report.get("optimizer_updates")) is not int
        or report["optimizer_updates"] != 0
        or type(report.get("tests_executed")) is not int
        or report["tests_executed"] != 0
        or sha256(work / "pyproject.toml") != report.get("common_rule_config_sha256")
    ):
        raise ValueError("static comparison contract or common rules changed")
    actual = {
        path.relative_to(work).as_posix(): path
        for folder in ("src", "scripts", "tests")
        for path in (work / folder).rglob("*.py")
    }
    declared = report["current_source_sha256"]
    if set(actual) != set(declared):
        raise ValueError("static QA inventory does not cover current repository")
    for name, path in actual.items():
        boundary()
        if sha256(path) != declared[name]:
            raise ValueError(f"static QA source changed: {name}")
    baseline = (root / "baseline_source").resolve(strict=True)
    baseline_paths = set()
    for pin in report["baseline_source_files"]:
        relative = Path(pin["path"])
        path = (baseline / relative).resolve(strict=True)
        if relative.is_absolute() or not path.is_relative_to(baseline) or path in baseline_paths:
            raise ValueError("baseline source path escapes or aliases")
        boundary()
        if sha256(path) != pin["sha256"]:
            raise ValueError("preserved baseline source changed")
        baseline_paths.add(path)

    def ids(name: str, source_root: Path) -> set[str]:
        result = set()
        for item in records[f"{name.upper()}_RUFF.json"]["diagnostics"]:
            path = Path(item["filename"]).resolve(strict=True)
            if not path.is_relative_to(source_root):
                raise ValueError("static diagnostic outside audited source")
            location = item["location"]
            key = (
                f"{path.relative_to(source_root).as_posix()}:{location['row']}:"
                f"{location['column']}:{item['code']}:{item['message']}"
            )
            if key in result:
                raise ValueError("duplicate static failure ID")
            result.add(key)
        run = report["runs"][name]
        if run != {"failures": len(result), "exit_code": 1 if result else 0}:
            raise ValueError("static run counts differ from raw diagnostics")
        return result

    before, after = ids("baseline", baseline), ids("current", work)
    for key, expected in (
        ("persistent_failure_ids", before & after),
        ("new_or_relocated_failure_ids", after - before),
        ("resolved_or_relocated_failure_ids", before - after),
    ):
        if report[key] != sorted(expected):
            raise ValueError("static failure comparison differs from raw diagnostics")
    if after - before:
        raise ValueError("new or relocated static failures require explicit review")
    boundary()
    return {
        "status": "CURRENT_RUFF_COMPARISON_REVERIFIED",
        "baseline_failures": len(before),
        "current_failures": len(after),
        "persistent_failures": len(before & after),
        "new_failures": 0,
        "source_files": len(actual),
        "optimizer_updates_executed": 0,
        "scientific_admission": False,
        "pins": pins,
    }
