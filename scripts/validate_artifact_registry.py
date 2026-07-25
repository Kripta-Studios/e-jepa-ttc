"""Validate the append-only recovery artifact registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ALLOWED_STATUSES = {
    "invalid_pre_fix",
    "reused_test_diagnostic",
    "smoke_only",
    "valid_post_fix",
}
REQUIRED_FIELDS = {
    "schema_version",
    "record_type",
    "run_id",
    "project",
    "stage",
    "status",
    "run_status",
    "validity_status",
    "claim_level",
    "created_at",
    "started_at",
    "completed_at",
    "git_commit",
    "dirty_worktree",
    "command",
    "config_path",
    "config_hash",
    "dataset_path",
    "dataset_manifest_path",
    "dataset_manifest_hash",
    "feature_schema_version",
    "input_view",
    "split_protocol",
    "split_path",
    "split_hash",
    "test_open_count",
    "pretrain_seed",
    "downstream_seed",
    "evaluation_seed",
    "data_seed",
    "model_seed",
    "requested_backbone",
    "actual_backbone",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_role",
    "checkpoint_selected_by",
    "best_checkpoint",
    "last_checkpoint",
    "selection_criterion",
    "best_checkpoint_path",
    "last_checkpoint_path",
    "metrics_path",
    "metrics_sha256",
    "predictions_path",
    "predictions_sha256",
    "hardware",
    "elapsed_seconds",
    "artifact_exists",
    "notes",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
PATH_HASH_FIELDS = (
    ("config_path", "config_hash"),
    ("dataset_manifest_path", "dataset_manifest_hash"),
    ("split_path", "split_hash"),
    ("checkpoint_path", "checkpoint_sha256"),
    ("metrics_path", "metrics_sha256"),
    ("predictions_path", "predictions_sha256"),
)
EXISTENCE_ONLY_FIELDS = (
    "best_checkpoint",
    "last_checkpoint",
    "best_checkpoint_path",
    "last_checkpoint_path",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_artifact_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    resolved = (path if path.is_absolute() else repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repository: {raw_path}") from exc
    return resolved


def _validate_physical_artifacts(
    record: dict[str, Any],
    *,
    line_number: int,
    repo_root: Path,
    digest_cache: dict[Path, str],
) -> list[str]:
    prefix = f"line {line_number} ({record.get('run_id', 'unknown')})"
    errors: list[str] = []
    asserted_to_exist = record.get("artifact_exists") is True
    for path_field, hash_field in PATH_HASH_FIELDS:
        raw_path = record.get(path_field)
        declared_hash = record.get(hash_field)
        if raw_path in (None, ""):
            if declared_hash is not None:
                errors.append(f"{prefix}: {hash_field} exists without {path_field}")
            continue
        if not isinstance(raw_path, str):
            errors.append(f"{prefix}: {path_field} must be a string or null")
            continue
        try:
            path = _resolve_artifact_path(repo_root, raw_path)
        except ValueError as exc:
            errors.append(f"{prefix}: {exc}")
            continue
        if not path.is_file():
            if asserted_to_exist or declared_hash is not None:
                errors.append(f"{prefix}: referenced file is missing: {raw_path}")
            continue
        if declared_hash is None:
            errors.append(f"{prefix}: existing {path_field} lacks {hash_field}")
            continue
        actual_hash = digest_cache.get(path)
        if actual_hash is None:
            actual_hash = _sha256(path)
            digest_cache[path] = actual_hash
        if actual_hash != declared_hash:
            errors.append(
                f"{prefix}: {hash_field} mismatch for {raw_path}: "
                f"declared={declared_hash}, actual={actual_hash}"
            )
    if asserted_to_exist:
        for path_field in EXISTENCE_ONLY_FIELDS:
            raw_path = record.get(path_field)
            if raw_path in (None, ""):
                continue
            if not isinstance(raw_path, str):
                errors.append(f"{prefix}: {path_field} must be a string or null")
                continue
            try:
                path = _resolve_artifact_path(repo_root, raw_path)
            except ValueError as exc:
                errors.append(f"{prefix}: {exc}")
                continue
            if not path.is_file():
                errors.append(f"{prefix}: referenced file is missing: {raw_path}")
    return errors


def _validate_record(record: dict[str, Any], *, line_number: int) -> list[str]:
    prefix = f"line {line_number} ({record.get('run_id', 'unknown')})"
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        errors.append(f"{prefix}: missing fields: {', '.join(missing)}")
    status = record.get("status")
    if status not in ALLOWED_STATUSES:
        errors.append(f"{prefix}: invalid status {status!r}")
    if record.get("validity_status") != status:
        errors.append(f"{prefix}: status must equal validity_status")
    if record.get("run_status") not in {"smoke", "diagnostic", "official_candidate"}:
        errors.append(f"{prefix}: invalid run_status {record.get('run_status')!r}")
    if record.get("schema_version") != 1:
        errors.append(f"{prefix}: schema_version must be 1")
    for field in (
        "config_hash",
        "dataset_manifest_hash",
        "split_hash",
        "checkpoint_sha256",
        "metrics_sha256",
        "predictions_sha256",
    ):
        value = record.get(field)
        if value is not None and (not isinstance(value, str) or not SHA256.fullmatch(value)):
            errors.append(f"{prefix}: {field} must be null or lowercase SHA-256")
    commit = record.get("git_commit")
    if commit is not None and (not isinstance(commit, str) or not GIT_SHA.fullmatch(commit)):
        errors.append(f"{prefix}: git_commit must be null or a full lowercase commit SHA")
    if status == "valid_post_fix":
        if record.get("dirty_worktree") is not False:
            errors.append(f"{prefix}: valid_post_fix requires dirty_worktree=false")
        for field in (
            "git_commit",
            "command",
            "config_hash",
            "dataset_manifest_hash",
            "split_hash",
            "metrics_path",
            "metrics_sha256",
            "started_at",
            "completed_at",
            "hardware",
            "data_seed",
            "model_seed",
            "requested_backbone",
            "actual_backbone",
            "best_checkpoint",
            "last_checkpoint",
            "selection_criterion",
        ):
            if record.get(field) in (None, "", {}):
                errors.append(f"{prefix}: valid_post_fix requires {field}")
    if status == "reused_test_diagnostic" and record.get("claim_level") != "diagnostic":
        errors.append(f"{prefix}: reused_test_diagnostic requires claim_level=diagnostic")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", nargs="?", type=Path, default=Path("artifacts/registry.jsonl"))
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip filesystem existence and SHA-256 verification.",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON audit report path.")
    args = parser.parse_args()
    errors: list[str] = []
    seen: set[str] = set()
    count = 0
    digest_cache: dict[Path, str] = {}
    repo_root = args.registry.resolve().parent.parent
    with args.registry.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            count += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {line_number}: expected a JSON object")
                continue
            run_id = record.get("run_id")
            if run_id in seen:
                errors.append(f"line {line_number}: duplicate run_id {run_id!r}")
            elif isinstance(run_id, str):
                seen.add(run_id)
            errors.extend(_validate_record(record, line_number=line_number))
            if not args.metadata_only:
                errors.extend(
                    _validate_physical_artifacts(
                        record,
                        line_number=line_number,
                        repo_root=repo_root,
                        digest_cache=digest_cache,
                    )
                )
    report = {
        "registry": args.registry.as_posix(),
        "record_count": count,
        "physical_verification": not args.metadata_only,
        "unique_files_hashed": len(digest_cache),
        "error_count": len(errors),
        "status": "passed" if not errors else "failed",
        "errors": errors,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {count} records from {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
