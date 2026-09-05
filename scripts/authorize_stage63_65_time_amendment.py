"""Test and freeze the explicitly requested wall-time-only execution amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from e_jepa_ttc.artifacts.campaign_session import append_transition  # noqa: E402
from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact  # noqa: E402
from e_jepa_ttc.artifacts.time_cap_amendment import (  # noqa: E402
    ALLOWED_FILES,
    AMENDMENT_NAME,
    validate_time_amendment,
)
from e_jepa_ttc.artifacts.training_authorization import read_signed  # noqa: E402
from e_jepa_ttc.training.raw_time_residual import load_frozen_raw_endpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--user-request", required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()

    def git(*parts: str) -> str:
        return subprocess.check_output(["git", *parts], cwd=REPO, text=True).strip()

    head = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("amendment QA requires a clean execution commit")
    if (root / AMENDMENT_NAME).exists():
        raise FileExistsError("preserve the existing amendment")
    lock_path = root / "TRAINING_LOCK.json"
    lock = read_signed(lock_path)
    lock_sha = compute_file_hash(str(lock_path))
    base = lock["training_commit"]
    changed = sorted(git("diff", "--name-only", base, head, "--").splitlines())
    if not changed or not set(changed) <= ALLOWED_FILES:
        raise ValueError("execution diff exceeds wall-time-only review scope")
    # No retrospective score-conditioned amendment is accepted by this issuer.
    if list((root / "stage64").glob("seed*/STAGE64_RESULT.json")) or list(
        (root / "stage64").glob("seed*/*_oof.csv")
    ):
        raise ValueError("this amendment issuer requires no observed outer scores")
    qa_dir = root / "qa" / f"time_amendment_{head[:12]}"
    qa_dir.mkdir(exist_ok=False)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    commands = {
        "targeted": [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "tests/test_stage63_time_amendment.py",
            "tests/test_stage63_training_authorization.py",
            "tests/test_stage63_campaign_budget.py",
            "tests/test_raw_time_residual.py",
            "--junitxml",
            str(qa_dir / "targeted.xml"),
        ],
        "ruff": [
            str(REPO / ".venv/Scripts/ruff.exe"),
            "check",
            *[name for name in changed if name.endswith(".py")],
        ],
        "format": [
            str(REPO / ".venv/Scripts/ruff.exe"),
            "format",
            "--check",
            *[name for name in changed if name.endswith(".py")],
        ],
        "types": [
            str(REPO / ".venv/Scripts/pyright.exe"),
            "--project",
            "pyright-stage63-65.json",
            "scripts/authorize_stage63_65_time_amendment.py",
            "src/e_jepa_ttc/artifacts/time_cap_amendment.py",
        ],
        "diff_check": ["git", "diff", "--check", base, head],
    }
    evidence = {}
    for name, command in commands.items():
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        log = qa_dir / f"{name}.log"
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        evidence[name] = {
            "path": str(log),
            "sha256": compute_file_hash(str(log)),
            "command": command,
            "exit_code": result.returncode,
        }
        print(f"{name}: {result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"amendment QA failed: {log}")
    anchors = []
    for path in sorted((root / "stage64").glob("seed*/outer*/S64-*/checkpoint_last.pt")):
        if (path.parent / "frozen_manifest.json").is_file():
            _, checkpoint, _ = load_frozen_raw_endpoint(path.parent, device=torch.device("cpu"))
        else:
            if compute_file_hash(str(path)) != path.with_suffix(".pt.sha256").read_text().strip():
                raise ValueError("partial checkpoint receipt mismatch")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if not (
                checkpoint["completed_updates"]
                == checkpoint["next_schedule_index"]
                == len(checkpoint["loss_history"])
            ):
                raise ValueError("partial checkpoint schedule mismatch")
        if checkpoint["identity"]["training_lock_sha256"] != lock_sha:
            raise ValueError("checkpoint belongs to another scientific lock")
        anchors.append(
            {
                "path": str(path),
                "sha256": compute_file_hash(str(path)),
                "updates": checkpoint["completed_updates"],
            }
        )
    if git("rev-parse", "HEAD") != head or git("status", "--porcelain"):
        raise ValueError("code changed during amendment QA")
    if compute_file_hash(str(lock_path)) != lock_sha:
        raise ValueError("original lock changed during amendment QA")
    qa_path = root / "qa/TIME_AMENDMENT_QA.json"
    qa = sign_artifact(
        {
            "status": "passed",
            "execution_commit": head,
            "training_lock_sha256": lock_sha,
            "evidence": evidence,
        }
    )
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    diff = subprocess.check_output(["git", "diff", "--binary", base, head, "--"], cwd=REPO)
    value = sign_artifact(
        {
            "artifact_type": "stage63_65_wall_time_amendment_v1",
            "execution_commit": head,
            "original_training_commit": base,
            "training_lock_sha256": lock_sha,
            "policy": "disable_wall_time_caps_only",
            "scope": ["stage63", "stage64_seed7", "stage64_replications", "stage65"],
            "fixed_endpoint_updates": 3000,
            "resource_margins_unchanged": True,
            "user_request": args.user_request,
            "changed_files": changed,
            "implementation_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "qa_sha256": compute_file_hash(str(qa_path)),
            "checkpoint_anchors": anchors,
            "outer_scores_observed": False,
        }
    )
    (root / AMENDMENT_NAME).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_time_amendment(root, REPO)
    append_transition(
        root / "RUN_LEDGER.jsonl",
        "user_time_amendment_frozen",
        execution_commit=head,
        amendment_sha256=compute_file_hash(str(root / AMENDMENT_NAME)),
        fixed_updates=3000,
        original_training_lock_sha256=lock_sha,
    )
    print(json.dumps({"status": "TIME_CAPS_DISABLED", "execution_commit": head}))


if __name__ == "__main__":
    main()
