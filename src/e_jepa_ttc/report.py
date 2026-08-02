"""Build a provenance-first report from regenerable result artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash

REPORTABLE_SUFFIXES = {".json", ".jsonl", ".csv", ".parquet"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_paths(artifacts_root: Path, excluded_root: Path) -> Iterator[Path]:
    """Yield reportable result files while excluding the report being rebuilt."""

    for path in sorted(artifacts_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in REPORTABLE_SUFFIXES:
            continue
        if path.resolve().is_relative_to(excluded_root):
            continue
        yield path


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _status(payload: dict[str, Any]) -> str:
    value = payload.get("status")
    if isinstance(value, str):
        return value.strip().lower()
    if payload.get("failure") is True or "failure" in str(payload.get("artifact_type", "")).lower():
        return "failed"
    return "unknown"


def _artifact_row(path: Path, root: Path) -> dict[str, Any]:
    payload = _load_json(path) if path.suffix.lower() == ".json" else None
    row: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "format": path.suffix.lower().lstrip("."),
        "artifact_type": "unparsed" if payload is None else payload.get("artifact_type", "unknown"),
        "status": "unknown" if payload is None else _status(payload),
        "experiment_id": None,
        "run_name": None,
        "git_commit": None,
        "config_hash": None,
        "seed": None,
        "provenance_present": False,
        "declared_artifact_sha256": None,
        "declared_artifact_sha256_valid": None,
        "metric_keys": [],
    }
    if payload is not None:
        declared_hash = payload.get("artifact_sha256")
        row.update(
            {
                "experiment_id": payload.get("experiment_id"),
                "run_name": payload.get("run_name"),
                "git_commit": payload.get("git_commit"),
                "config_hash": payload.get("config_hash"),
                "seed": payload.get("seed"),
                "provenance_present": isinstance(payload.get("provenance"), dict),
                "declared_artifact_sha256": declared_hash,
                "declared_artifact_sha256_valid": (
                    verify_artifact_hash(payload) if isinstance(declared_hash, str) else None
                ),
                "metric_keys": sorted(
                    key
                    for key, value in payload.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ),
            }
        )
    return row


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_report(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Index regenerable result artifacts below ``artifacts`` and write a report bundle."""

    artifacts_root = repo_root / "artifacts"
    if not artifacts_root.is_dir():
        raise FileNotFoundError(f"Artifact root does not exist: {artifacts_root}")
    rows = [
        _artifact_row(path, repo_root)
        for path in _artifact_paths(artifacts_root, output_dir.resolve())
    ]
    rows.sort(key=lambda row: row["path"])
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    payload = {
        "artifact_type": "regenerable_report_manifest_v1",
        "git_commit_at_build": _git_commit(repo_root),
        "artifact_root": "artifacts",
        "artifact_count": len(rows),
        "reportable_formats": sorted(REPORTABLE_SUFFIXES),
        "status_counts": status_counts,
        "declared_hash_counts": {
            "present": sum(row["declared_artifact_sha256"] is not None for row in rows),
            "valid": sum(row["declared_artifact_sha256_valid"] is True for row in rows),
            "invalid": sum(row["declared_artifact_sha256_valid"] is False for row in rows),
        },
        "artifacts": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Regenerable experiment report",
        "",
        "This file is generated from result artifacts; no metric is entered by hand.",
        "",
        f"- Git commit at build: `{payload['git_commit_at_build'] or 'unavailable'}`",
        f"- Result artifacts indexed: `{len(rows)}`",
        f"- Status counts: `{json.dumps(status_counts, sort_keys=True)}`",
        "",
        "| Artifact | Type | Status | SHA-256 | Declared hash | Scalar metrics |\n"
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        metrics = ", ".join(str(value) for value in row["metric_keys"]) or "—"
        lines.append(
            f"| `{row['path']}` | `{row['artifact_type']}` | `{row['status']}` "
            f"| `{row['sha256']}` | `{row['declared_artifact_sha256_valid']}` | {metrics} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


__all__ = ["build_report"]
