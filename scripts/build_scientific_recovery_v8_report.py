#!/usr/bin/env python
"""Build a fail-closed Scientific Recovery V8 evidence report.

The builder reads only local, signed JSON artifacts and CSV files whose digest a
signed JSON artifact declares.  It never runs an experiment or opens a sealed
split.  Missing or invalid evidence leaves the corresponding phase pending or
blocked instead of manufacturing a result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402

REPORT_VERSION = "scientific_recovery_v8_report_v1"
SEALED_EVALUATIONS = ("public validation", "private test", "EvTTC test", "CodaBench")
EXPECTED_PHASES = (
    ("P0", "integrity and frozen protocol", "pending"),
    ("A", "autopsy without new training", "pending"),
    ("R", "prospective nested router", "pending"),
    ("B1", "TIMEVOL20-3", "pending"),
    ("B2", "EXP6-3", "pending"),
    ("B3", "PAIR20-2", "blocked: requires the B1 screen gate"),
    ("C1", "GATED-EXP6-3", "blocked: the mechanism gate remains closed"),
    ("D0-D4", "JEPA attribution", "pending"),
    ("E", "one confirmation candidate, robustness, export and package", "blocked"),
)


@dataclass(frozen=True)
class EvidenceFile:
    """A file whose content has passed the V8 report's provenance checks."""

    path: Path
    relative_path: str
    sha256: str
    kind: str
    artifact_type: str | None
    status: str | None
    payload: dict[str, Any] | None


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot parse JSON evidence {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must contain an object: {path}")
    return payload


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


def _v8_candidate(path: Path) -> bool:
    return "scientific_recovery_v8" in path.as_posix().lower()


def _evidence_roots(repo_root: Path) -> tuple[Path, ...]:
    """Return the local roots allowed as V8 report evidence."""

    return (
        repo_root / "artifacts" / "scientific_recovery_v8",
        repo_root / "artifacts" / "runs",
    )


def _iter_v8_json(repo_root: Path, output_dir: Path) -> Iterable[Path]:
    for root in _evidence_roots(repo_root):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            if path.resolve().is_relative_to(output_dir.resolve()):
                continue
            if root.name == "runs" and not _v8_candidate(path):
                continue
            yield path


def _normalise_relative(repo_root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(repo_root.resolve())
        except ValueError:
            return None
    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return resolved


def _declared_file_hashes(payload: Mapping[str, Any], repo_root: Path) -> dict[Path, str]:
    """Find path/digest pairs in a signed artifact, including nested source maps."""

    found: dict[Path, str] = {}

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            path_value = value.get("path")
            digest_value = value.get("sha256", value.get("file_sha256"))
            path = _normalise_relative(repo_root, path_value)
            if path is not None and isinstance(digest_value, str) and len(digest_value) == 64:
                found[path] = digest_value.lower()
            for key, nested in value.items():
                if key.endswith("_path") and isinstance(nested, str):
                    prefix = key[: -len("_path")]
                    digest = value.get(f"{prefix}_sha256")
                    path = _normalise_relative(repo_root, nested)
                    if path is not None and isinstance(digest, str) and len(digest) == 64:
                        found[path] = digest.lower()
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return found


def _validate_csv_schema(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return [f"CSV cannot be read: {error}"]
    if not header or any(not item.strip() for item in header):
        return ["CSV requires a non-empty header"]
    if len(set(header)) != len(header):
        return ["CSV header contains duplicate columns"]
    return []


def _phase_for(payload: Mapping[str, Any], path: Path) -> str | None:
    text = " ".join(
        str(value)
        for value in (
            payload.get("arm"),
            payload.get("phase"),
            payload.get("run_name"),
            payload.get("experiment_id"),
            payload.get("artifact_type"),
            path.name,
        )
        if value is not None
    ).lower()
    if "router" in text:
        return "R"
    if "timevol" in text:
        return "B1"
    if "exp6" in text and "gated" not in text:
        return "B2"
    if "pair20" in text:
        return "B3"
    if "gated" in text:
        return "C1"
    if "jepa" in text or any(f"d{item}" in text for item in range(5)):
        return "D0-D4"
    if "autopsy" in text or "replay" in text:
        return "A"
    if "freeze" in text or "integrity" in text or "quality" in text:
        return "P0"
    if "robust" in text or "onnx" in text or "confirm" in text:
        return "E"
    return None


def _status(payload: Mapping[str, Any]) -> str:
    value = payload.get("status")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    if payload.get("failure") is True:
        return "failed"
    return "unknown"


def _validate_json_schema(payload: Mapping[str, Any]) -> list[str]:
    """Check the minimum schema needed to label a signed result artifact."""

    errors: list[str] = []
    if not isinstance(payload.get("artifact_type"), str) or not payload["artifact_type"].strip():
        errors.append("JSON artifact requires a non-empty artifact_type")
    if not isinstance(payload.get("status"), str) or not payload["status"].strip():
        errors.append("JSON result artifact requires a non-empty status")
    return errors


def _run_row(evidence: EvidenceFile, phase: str | None) -> dict[str, Any]:
    assert evidence.payload is not None
    payload = evidence.payload
    return {
        "phase": phase or "unclassified",
        "path": evidence.relative_path,
        "artifact_type": payload.get("artifact_type", "unknown"),
        "status": _status(payload),
        "run_name": payload.get("run_name"),
        "experiment_id": payload.get("experiment_id"),
        "seed": payload.get("seed"),
        "git_commit": payload.get("git_commit"),
        "config_hash": payload.get("config_hash", payload.get("config_sha256")),
        "artifact_sha256": payload.get("artifact_sha256"),
        "file_sha256": evidence.sha256,
        "metrics": {
            key: value
            for key, value in payload.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
    }


def collect_evidence(
    repo_root: Path, output_dir: Path
) -> tuple[list[EvidenceFile], list[dict[str, str]]]:
    """Collect signed V8 JSON and signed CSV evidence, retaining validation errors."""

    signed_json: list[EvidenceFile] = []
    errors: list[dict[str, str]] = []
    declared_files: dict[Path, str] = {}
    for path in _iter_v8_json(repo_root, output_dir):
        try:
            payload = _load_json(path)
        except ValueError as error:
            errors.append({"path": path.relative_to(repo_root).as_posix(), "error": str(error)})
            continue
        if not verify_artifact_hash(payload):
            errors.append(
                {
                    "path": path.relative_to(repo_root).as_posix(),
                    "error": "JSON artifact signature is missing or invalid",
                }
            )
            continue
        schema_errors = _validate_json_schema(payload)
        if schema_errors:
            errors.extend(
                {"path": path.relative_to(repo_root).as_posix(), "error": error}
                for error in schema_errors
            )
            continue
        evidence = EvidenceFile(
            path=path,
            relative_path=path.relative_to(repo_root).as_posix(),
            sha256=sha256_file(path),
            kind="json",
            artifact_type=(
                str(payload.get("artifact_type")) if payload.get("artifact_type") else None
            ),
            status=_status(payload),
            payload=payload,
        )
        signed_json.append(evidence)
        declared_files.update(_declared_file_hashes(payload, repo_root))

    csv_evidence: list[EvidenceFile] = []
    for path, expected_digest in sorted(
        declared_files.items(), key=lambda item: item[0].as_posix()
    ):
        if path.suffix.lower() != ".csv" or not _v8_candidate(path):
            continue
        relative = path.relative_to(repo_root).as_posix() if path.exists() else path.as_posix()
        if not path.is_file():
            errors.append({"path": relative, "error": "signed CSV source is missing"})
            continue
        observed = sha256_file(path)
        if observed != expected_digest:
            errors.append(
                {"path": relative, "error": "signed CSV SHA-256 differs from declaration"}
            )
            continue
        schema_errors = _validate_csv_schema(path)
        if schema_errors:
            errors.extend({"path": relative, "error": error} for error in schema_errors)
            continue
        csv_evidence.append(
            EvidenceFile(
                path=path,
                relative_path=relative,
                sha256=observed,
                kind="csv",
                artifact_type="signed_csv_source",
                status="validated_source",
                payload=None,
            )
        )
    return signed_json + csv_evidence, errors


def build_report(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Write a signed report bundle from V8 evidence, or a blocked bundle when absent."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    protocol_path = repo_root / "configs" / "protocol" / "scientific_recovery_v8_temporal.json"
    protocol_error: str | None = None
    protocol: dict[str, Any] | None = None
    if protocol_path.is_file():
        try:
            candidate = _load_json(protocol_path)
            if not verify_artifact_hash(candidate):
                protocol_error = "V8 protocol signature is missing or invalid"
            else:
                protocol = candidate
        except ValueError as error:
            protocol_error = str(error)
    else:
        protocol_error = "frozen V8 protocol is absent"

    evidence, errors = collect_evidence(repo_root, output_dir)
    if protocol_error is not None:
        errors.append(
            {"path": protocol_path.relative_to(repo_root).as_posix(), "error": protocol_error}
        )
    run_rows = [
        _run_row(item, _phase_for(item.payload or {}, item.path))
        for item in evidence
        if item.payload
    ]
    phase_rows: list[dict[str, str]] = []
    for code, description, fallback in EXPECTED_PHASES:
        observed = [row["status"] for row in run_rows if row["phase"] == code]
        if any("fail" in status or "negative" in status for status in observed):
            status = "failed_or_negative"
        elif any("complete" in status or "pass" in status for status in observed):
            status = "evidence_present_unreviewed"
        else:
            status = fallback
        phase_rows.append({"phase": code, "description": description, "status": status})

    report: dict[str, Any] = {
        "artifact_type": REPORT_VERSION,
        "schema_version": REPORT_VERSION,
        "status": "blocked" if errors or not run_rows else "evidence_indexed_not_claim_ready",
        "git_commit_at_build": _git_commit(repo_root),
        "protocol": (
            {
                "path": protocol_path.relative_to(repo_root).as_posix(),
                "file_sha256": sha256_file(protocol_path),
                "artifact_sha256": protocol.get("artifact_sha256"),
            }
            if protocol is not None
            else None
        ),
        "sealed_evaluations": {name: "sealed" for name in SEALED_EVALUATIONS},
        "sota_claim_allowed": False,
        "claim_policy": "The V8 report does not make SOTA claims.",
        "evidence_counts": {
            "validated_json": sum(item.kind == "json" for item in evidence),
            "validated_csv": sum(item.kind == "csv" for item in evidence),
            "runs": len(run_rows),
            "validation_errors": len(errors),
        },
        "phases": phase_rows,
        "runs": run_rows,
        "validation_errors": errors,
        "missing_evidence": [
            "No V8 result is claim-ready until signed aggregate and prediction evidence exists."
        ]
        if not run_rows
        else [],
    }
    sign_artifact(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scientific_recovery_v8_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    csv_path = output_dir / "scientific_recovery_v8_runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "phase",
                "path",
                "artifact_type",
                "status",
                "run_name",
                "experiment_id",
                "seed",
                "git_commit",
                "config_hash",
                "artifact_sha256",
                "file_sha256",
            ),
        )
        writer.writeheader()
        for row in run_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    _write_markdown(report, output_dir / "scientific_recovery_v8_report.md")
    return report


def _write_markdown(report: Mapping[str, Any], output_path: Path) -> None:
    lines = [
        "# Scientific Recovery V8 evidence report",
        "",
        f"Build status: `{report['status']}`.",
        "",
        "The builder accepts signed local artifacts and signed CSV sources only. "
        "It does not run experiments or open sealed evaluations.",
        "",
        "## Phase ledger",
        "",
        "| Phase | Scope | Status |",
        "|---|---|---|",
    ]
    for phase in report["phases"]:
        lines.append(f"| {phase['phase']} | {phase['description']} | `{phase['status']}` |")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"Validated JSON artifacts: `{report['evidence_counts']['validated_json']}`.",
            f"Validated CSV sources: `{report['evidence_counts']['validated_csv']}`.",
            f"Runs indexed: `{report['evidence_counts']['runs']}`.",
            "",
            "| Phase | Artifact | Status | Seed |",
            "|---|---|---|---|",
        ]
    )
    for row in report["runs"]:
        lines.append(f"| {row['phase']} | `{row['path']}` | `{row['status']}` | `{row['seed']}` |")
    if report["validation_errors"]:
        lines.extend(["", "## Blockers", ""])
        for error in report["validation_errors"]:
            lines.append(f"- `{error['path']}`: {error['error']}")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "No V8 metric or comparative conclusion appears until signed evidence passes the "
            "frozen protocol checks. This report prohibits SOTA claims.",
            "",
            "Public validation, private test, EvTTC test and CodaBench remain sealed.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/scientific_recovery_v8/report"),
    )
    args = parser.parse_args()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else args.repo_root / args.output_dir
    )
    try:
        report = build_report(args.repo_root, output_dir)
    except Exception as error:
        parser.exit(2, f"V8 report build failed: {type(error).__name__}: {error}\n")
    print(json.dumps({"status": report["status"], "artifact_sha256": report["artifact_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
