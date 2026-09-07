"""Reconstruct static failure comparisons from pinned diagnostics, not PASS flags."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256


def _read_report(work: Path, path: Path, digest: str) -> dict:
    path = path.resolve(strict=True)
    if (
        not path.is_relative_to(work / "artifacts/simplex_t/T0")
        or path.stat().st_size > 16_777_216
        or sha256(path) != digest
    ):
        raise ValueError("static report path, size or bytes changed")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sources(
    work: Path,
    declared: dict[str, str],
    actual: set[Path],
    resource_ok: Callable[[], bool],
) -> None:
    resolved = {}
    for name, digest in declared.items():
        relative = Path(name)
        path = (work / relative).resolve(strict=True)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(work)
            or path in resolved
        ):
            raise ValueError("static source path escapes or aliases")
        resolved[path] = digest
    if set(resolved) != actual:
        raise ValueError("static report does not cover its complete current source scope")
    for path, digest in resolved.items():
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: static source verification")
        if sha256(path) != digest:
            raise ValueError("static report source bytes changed")


def verify_companion_types(
    work: Path,
    report: Path,
    *,
    report_sha256: str,
    resource_ok: Callable[[], bool],
) -> dict:
    """Recheck complete companion package/script Pyright evidence, not all repo types."""
    work = work.resolve(strict=True)
    record = _read_report(work, report, report_sha256)
    actual = {p.resolve() for p in (work / "src/e_jepa_ttc/simplex_t").rglob("*.py")} | {
        p.resolve() for p in (work / "scripts").glob("*simplex*.py")
    }
    summary = record["result"]["summary"]
    if (
        record.get("status") != "STATIC_TYPES_ONLY_NOT_SCIENTIFIC_ADMISSION"
        or type(record.get("exit_code")) is not int
        or record["exit_code"] != 0
        or type(record.get("optimizer_updates")) is not int
        or record["optimizer_updates"] != 0
        or record["result"]["generalDiagnostics"] != []
        or not actual
        or type(summary.get("filesAnalyzed")) is not int
        or summary["filesAnalyzed"] != len(actual)
        or any(
            type(summary.get(key)) is not int or summary[key] != 0
            for key in ("errorCount", "warningCount", "informationCount")
        )
    ):
        raise ValueError("complete clean companion type evidence required")
    _verify_sources(work, record["source_sha256"], actual, resource_ok)
    return {
        "status": "COMPANION_TYPES_CURRENT_SOURCES_VERIFIED",
        "files": len(actual),
        "optimizer_updates_executed": 0,
        "scientific_admission": False,
    }


def verify_powershell_syntax(
    work: Path,
    report: Path,
    *,
    report_sha256: str,
    resource_ok: Callable[[], bool],
) -> dict:
    """Verify current PowerShell parse coverage and preserved baseline parse results."""
    work = work.resolve(strict=True)
    record = _read_report(work, report, report_sha256)
    if (
        record.get("status") != "POWERSHELL_AST_ONLY_NOT_SCRIPT_EXECUTION"
        or type(record.get("optimizer_updates")) is not int
        or record["optimizer_updates"] != 0
        or len(record["runs"]) != 2
    ):
        raise ValueError("current and baseline PowerShell parsing evidence required")
    roots = [Path(run["root"]).resolve(strict=True) for run in record["runs"]]
    if len(set(roots)) != 2 or roots.count(work) != 1:
        raise ValueError("ambiguous PowerShell current/baseline roots")
    baseline = next(root for root in roots if root != work)
    if not baseline.is_relative_to(work / "artifacts/simplex_t/T0"):
        raise ValueError("PowerShell baseline outside preserved T0 evidence")
    counts = {}
    for root, run in zip(roots, record["runs"], strict=True):
        rows = run["files"]
        if not rows or any(row["errors"] != [] for row in rows):
            raise ValueError("PowerShell syntax failures require review")
        declared = {row["path"]: row["sha256"] for row in rows}
        if len(declared) != len(rows):
            raise ValueError("duplicate PowerShell source record")
        actual = {
            p.resolve()
            for folder in ("src", "scripts")
            for p in (root / folder).rglob("*")
            if p.is_file() and p.suffix.lower() in {".ps1", ".psm1"}
        }
        _verify_sources(root, declared, actual, resource_ok)
        counts["current" if root == work else "baseline"] = len(actual)
    return {
        "status": "POWERSHELL_SYNTAX_CURRENT_SOURCES_VERIFIED",
        "files": counts,
        "optimizer_updates_executed": 0,
        "scientific_admission": False,
    }


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
