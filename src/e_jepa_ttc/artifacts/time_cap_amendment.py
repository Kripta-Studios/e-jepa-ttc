"""Explicit, QA-bound wall-time-only amendment to an immutable scientific lock."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash

AMENDMENT_NAME = "WALL_TIME_AMENDMENT.json"
ALLOWED_FILES = frozenset(
    {
        "src/e_jepa_ttc/artifacts/time_cap_amendment.py",
        "src/e_jepa_ttc/artifacts/training_authorization.py",
        "src/e_jepa_ttc/training/campaign_budget.py",
        "scripts/run_scientific_recovery_v9_stage63_65.py",
        "scripts/run_scientific_recovery_v9_stage64.py",
        "scripts/audit_scientific_recovery_v9_stage63.py",
        "scripts/authorize_stage63_65_time_amendment.py",
        "tests/test_stage63_time_amendment.py",
        "docs/STAGE63_65_TIME_CAP_AMENDMENT.md",
    }
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError(f"time amendment signature mismatch: {path}")
    return value


def validate_time_amendment(root: Path, repo: Path) -> dict[str, Any] | None:
    """Accept only the reviewed execution commit and byte-bound passing QA."""
    path = root / AMENDMENT_NAME
    if not path.exists():
        return None
    value = _read(path)
    lock = _read(root / "TRAINING_LOCK.json")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    if (
        dirty
        or value.get("execution_commit") != head
        or value.get("original_training_commit") != lock["training_commit"]
        or value.get("training_lock_sha256") != compute_file_hash(str(root / "TRAINING_LOCK.json"))
        or value.get("policy") != "disable_wall_time_caps_only"
        or value.get("scope") != ["stage63", "stage64_seed7", "stage64_replications", "stage65"]
        or value.get("fixed_endpoint_updates") != 3000
        or value.get("resource_margins_unchanged") is not True
        or not value.get("user_request")
    ):
        raise ValueError("time amendment execution/lock/scope identity mismatch")
    base = str(lock["training_commit"])
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", base, head, "--"], cwd=repo, text=True
        ).splitlines()
    )
    diff = subprocess.check_output(["git", "diff", "--binary", base, head, "--"], cwd=repo)
    if (
        not changed
        or not changed <= ALLOWED_FILES
        or value.get("changed_files") != sorted(changed)
        or value.get("implementation_diff_sha256") != hashlib.sha256(diff).hexdigest()
    ):
        raise ValueError("time amendment exceeds reviewed code scope")
    qa_path = root / "qa/TIME_AMENDMENT_QA.json"
    if compute_file_hash(str(qa_path)) != value.get("qa_sha256"):
        raise ValueError("time amendment QA bytes changed")
    qa = _read(qa_path)
    if (
        qa.get("status") != "passed"
        or qa.get("execution_commit") != head
        or qa.get("training_lock_sha256") != value["training_lock_sha256"]
        or not qa.get("evidence")
    ):
        raise ValueError("time amendment lacks matching passing QA")
    for record in qa["evidence"].values():
        evidence = Path(record["path"])
        if compute_file_hash(str(evidence)) != record["sha256"]:
            raise ValueError("time amendment QA evidence changed")
    return value


def training_identity_commit(root: Path, repo: Path) -> str:
    """Retain the original scientific identity under the separate execution amendment."""
    value = validate_time_amendment(root, repo)
    if value is not None:
        return str(value["original_training_commit"])
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
