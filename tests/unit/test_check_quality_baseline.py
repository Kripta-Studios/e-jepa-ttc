"""Hermetic checks for the V8 Ruff baseline quality gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import check_quality_baseline as quality


def test_normalize_ruff_diagnostics_is_path_stable_and_counted(tmp_path: Path) -> None:
    source = tmp_path / "nested" / "example.py"
    source.parent.mkdir()
    source.write_text("x = 1\n", encoding="utf-8")

    violations = quality.normalize_ruff_diagnostics(
        [
            {"filename": str(source), "code": "F401", "message": "unused import"},
            {"filename": "nested/example.py", "code": "F401", "message": "unused import"},
            {"filename": "nested/example.py", "code": "E501", "message": "line too long"},
        ],
        tmp_path,
    )

    assert violations == [
        quality.Violation("nested/example.py", "E501", "line too long", 1),
        quality.Violation("nested/example.py", "F401", "unused import", 2),
    ]


def test_new_violations_allows_historical_decreases() -> None:
    baseline = [quality.Violation("src/example.py", "F401", "unused import", 3)]
    current = [
        quality.Violation("src/example.py", "F401", "unused import", 2),
        quality.Violation("src/new.py", "E501", "line too long", 1),
    ]

    assert quality._new_violations(current, baseline) == [
        quality.Violation("src/new.py", "E501", "line too long", 1)
    ]


def test_load_baseline_requires_a_valid_signature(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    payload = quality.sign_artifact(
        {
            "schema_version": quality.SCHEMA_VERSION,
            "generated_against_git_commit": "abc123",
            "ruff_version": "ruff 0.1.0",
            "normalized_violations": [],
        }
    )
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    assert quality.load_baseline(baseline_path) == []

    payload["ruff_version"] = "tampered"
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(quality.QualityGateError, match="artifact_sha256"):
        quality.load_baseline(baseline_path)


def test_changed_python_files_includes_diff_and_untracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("x = 1\n", encoding="utf-8")

    def fake_git_lines(root: Path, arguments: list[str], purpose: str) -> list[str]:  # noqa: ARG001
        if arguments[0] == "diff":
            return ["tracked.py", "deleted.py", "README.md"]
        return ["new.py", "scratch.txt"]

    monkeypatch.setattr(quality, "_git_lines", fake_git_lines)

    assert quality.changed_python_files(tmp_path, "base") == [Path("new.py"), Path("tracked.py")]


def test_changed_python_files_allows_an_empty_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quality, "_git_lines", lambda *_: [])

    assert quality.changed_python_files(tmp_path, "base") == []


def test_repository_root_reports_missing_git_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_git = subprocess.CompletedProcess(
        args=["git", "rev-parse", "--show-toplevel"], returncode=128, stderr="not a git repository"
    )
    monkeypatch.setattr(quality, "_run", lambda *_: missing_git)

    with pytest.raises(quality.QualityGateError, match="Cannot locate repository root"):
        quality.repository_root(tmp_path)


def test_run_gate_rejects_unformatted_changed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_path = tmp_path / "baseline.json"
    payload = quality.sign_artifact(
        {
            "schema_version": quality.SCHEMA_VERSION,
            "generated_against_git_commit": "abc123",
            "ruff_version": "ruff 0.1.0",
            "normalized_violations": [],
        }
    )
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(quality, "repository_root", lambda _: tmp_path)
    monkeypatch.setattr(quality, "ruff_diagnostics", lambda _: [])
    monkeypatch.setattr(quality, "changed_python_files", lambda _, __: [Path("new.py")])
    monkeypatch.setattr(quality, "_changed_files_clean", lambda _, __: (False, "needs formatting"))

    assert quality.run_gate(tmp_path, baseline_path, "base", write_baseline=False) == 1
