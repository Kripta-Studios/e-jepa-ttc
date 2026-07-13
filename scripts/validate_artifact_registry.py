"""Validate the append-only recovery artifact registry."""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    errors: list[str] = []
    seen: set[str] = set()
    count = 0
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
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {count} records from {args.registry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
