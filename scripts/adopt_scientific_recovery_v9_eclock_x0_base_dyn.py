#!/usr/bin/env python
"""Adopt verified BASE/DYN folds after a reader-only infrastructure fix."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.hashing import (
    compute_file_hash,
    sign_artifact,
    verify_artifact_hash,
)
from e_jepa_ttc.training.collision_clock_eap import require_frozen_checkpoint

ARMS = ("X0-BASE-U", "X0-DYN-U")
CONFIGS = {
    "X0-BASE-U": "configs/experiment/scientific_recovery_v9_eclock/x0_base_u.yaml",
    "X0-DYN-U": "configs/experiment/scientific_recovery_v9_eclock/x0_dyn_u.yaml",
}
ALLOWED_CHANGED_PATHS = {
    "scripts/adopt_scientific_recovery_v9_eclock_x0_base_dyn.py",
    "scripts/analyze_scientific_recovery_v9_eclock_x0.py",
    "scripts/package_scientific_recovery_v9_eclock_x0_results.py",
    "scripts/run_scientific_recovery_v9_eclock_x0_full_campaign.ps1",
    "src/e_jepa_ttc/evaluation/collision_clock_aggregate.py",
    "src/e_jepa_ttc/evaluation/collision_clock_cross_arm.py",
    "src/e_jepa_ttc/evaluation/collision_clock_protocol.py",
    "src/e_jepa_ttc/evaluation/collision_clock_runner.py",
    "tests/unit/test_collision_clock_aggregate_io.py",
}
TRAINING_SURFACE = (
    "configs/experiment/scientific_recovery_v9_eclock",
    "configs/protocol",
    "configs/schema",
    "src/e_jepa_ttc/data/collision_clock_cache.py",
    "src/e_jepa_ttc/losses/collision_clock.py",
    "src/e_jepa_ttc/models/collision_clock_features.py",
    "src/e_jepa_ttc/models/collision_clock_math.py",
    "src/e_jepa_ttc/models/collision_clock_motion.py",
    "src/e_jepa_ttc/models/collision_clock_ttc.py",
    "src/e_jepa_ttc/training/collision_clock_eap.py",
    "src/e_jepa_ttc/evaluation/collision_clock_config.py",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_signed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError(f"signed artifact verification failed: {path}")
    return value


def _validate_fold(
    fold_root: Path,
    *,
    arm: str,
    fold: int,
    source_commit: str,
    config_sha256: str,
) -> dict[str, Any]:
    summary_path = fold_root / "fold_summary.json"
    oof_path = fold_root / "oof_predictions.csv"
    summary = _load_signed(summary_path)
    required = {
        "arm_id": arm,
        "outer_fold": fold,
        "seed": 7,
        "git_commit": source_commit,
        "config_sha256": config_sha256,
        "checkpoint_policy": "last_update_fixed_budget",
        "status": "completed_after_frozen_checkpoint",
        "updates_completed": 6840,
        "outer_dev_evaluations": 1,
        "outer_dev_used_during_training": False,
        "outer_dev_used_for_selection": False,
        "external_official_a5": False,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise ValueError(f"{arm} fold {fold} {key} differs from reusable contract")
    if not oof_path.is_file():
        raise ValueError(f"{arm} fold {fold} OOF is missing")
    if (
        oof_path.stat().st_size != int(summary["oof_bytes"])
        or compute_file_hash(str(oof_path)) != summary["oof_file_sha256"]
    ):
        raise ValueError(f"{arm} fold {fold} OOF physical identity mismatch")
    checkpoint = Path(str(summary["checkpoint_path"]))
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != int(summary["checkpoint_bytes"])
        or compute_file_hash(str(checkpoint)) != summary["checkpoint_file_sha256"]
    ):
        raise ValueError(f"{arm} fold {fold} checkpoint physical identity mismatch")
    frozen = require_frozen_checkpoint(checkpoint)
    if frozen.get("artifact_sha256") != summary["checkpoint_manifest_sha256"]:
        raise ValueError(f"{arm} fold {fold} checkpoint manifest identity mismatch")
    frame = pd.read_csv(oof_path, float_precision="round_trip")
    if len(frame) != int(summary["row_count"]):
        raise ValueError(f"{arm} fold {fold} OOF row count mismatch")
    if frame["sample_token"].astype(str).duplicated().any():
        raise ValueError(f"{arm} fold {fold} OOF tokens are duplicated")
    categorical = {
        "arm_id": arm,
        "seed": 7,
        "outer_fold": fold,
        "checkpoint_sha256": summary["checkpoint_file_sha256"],
        "config_sha256": config_sha256,
        "protocol_sha256": summary["protocol_sha256"],
    }
    for column, expected in categorical.items():
        if set(frame[column].tolist()) != {expected}:
            raise ValueError(f"{arm} fold {fold} OOF {column} identity mismatch")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{arm} fold {fold} OOF contains non-finite values")
    return {
        "outer_fold": fold,
        "row_count": len(frame),
        "fold_summary_sha256": summary["artifact_sha256"],
        "fold_summary_file_sha256": compute_file_hash(str(summary_path)),
        "oof_file_sha256": summary["oof_file_sha256"],
        "oof_bytes": summary["oof_bytes"],
        "checkpoint_file_sha256": summary["checkpoint_file_sha256"],
        "checkpoint_bytes": summary["checkpoint_bytes"],
        "checkpoint_manifest_sha256": summary["checkpoint_manifest_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--destination-campaign", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--current-commit", required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = args.source_campaign.resolve()
    destination = args.destination_campaign.resolve()
    observed_commit = _git(repo, "rev-parse", "HEAD")
    if observed_commit != args.current_commit:
        raise ValueError("current commit does not match checked-out HEAD")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("reuse adoption is forbidden from a dirty tracked worktree")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            args.source_commit,
            observed_commit,
        ],
        check=True,
    )
    changed = set(
        filter(
            None,
            _git(
                repo, "diff", "--name-only", f"{args.source_commit}..{observed_commit}"
            ).splitlines(),
        )
    )
    unexpected = sorted(changed - ALLOWED_CHANGED_PATHS)
    if unexpected:
        raise ValueError(
            f"post-source changes escape the reader/provenance allowlist: {unexpected}"
        )
    sensitive = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--quiet",
            f"{args.source_commit}..{observed_commit}",
            "--",
            *TRAINING_SURFACE,
        ]
    )
    if sensitive.returncode != 0:
        raise ValueError("training/config/cache/model surface changed after source commit")
    if not source.is_dir() or not destination.is_dir() or source == destination:
        raise ValueError(
            "source and destination campaign roots must be distinct existing directories"
        )
    failure = json.loads((source / "failure/fatal.json").read_text(encoding="utf-8"))
    comparison_log = source / "master_logs/compare-DYN-vs-BASE.log"
    if (
        failure.get("git_commit") != args.source_commit
        or failure.get("message") != "compare-DYN-vs-BASE failed with exit code 1"
        or "self-consistent row hashes do not match the canonical protocol"
        not in comparison_log.read_text(encoding="utf-8")
    ):
        raise ValueError("source campaign did not fail at the audited reader-only boundary")
    arm_records: dict[str, Any] = {}
    all_tokens: dict[str, set[str]] = {}
    for arm in ARMS:
        source_arm = source / arm
        destination_arm = destination / arm
        if destination_arm.exists():
            raise FileExistsError(f"destination arm already exists: {destination_arm}")
        manifest = _load_signed(source_arm / "campaign_manifest.json")
        config_path = repo / CONFIGS[arm]
        config_sha256 = compute_file_hash(str(config_path))
        if (
            manifest.get("arm_id") != arm
            or manifest.get("git_commit") != args.source_commit
            or manifest.get("seed") != 7
            or manifest.get("folds") != [0, 1, 2]
            or manifest.get("config_sha256") != config_sha256
        ):
            raise ValueError(f"{arm} source campaign manifest identity mismatch")
        folds = [
            _validate_fold(
                source_arm / f"fold-{fold}",
                arm=arm,
                fold=fold,
                source_commit=args.source_commit,
                config_sha256=config_sha256,
            )
            for fold in (0, 1, 2)
        ]
        tokens: set[str] = set()
        for fold in (0, 1, 2):
            frame = pd.read_csv(
                source_arm / f"fold-{fold}/oof_predictions.csv",
                usecols=["sample_token"],
            )
            fold_tokens = set(frame["sample_token"].astype(str))
            if tokens & fold_tokens:
                raise ValueError(f"{arm} tokens overlap across folds")
            tokens |= fold_tokens
        if len(tokens) != 8192:
            raise ValueError(f"{arm} does not have complete 8192-row OOF coverage")
        all_tokens[arm] = tokens
        shutil.copytree(
            source_arm, destination_arm, ignore=shutil.ignore_patterns("aggregate.json")
        )
        arm_records[arm] = {
            "source_campaign_manifest_sha256": manifest["artifact_sha256"],
            "config_path": CONFIGS[arm],
            "config_file_sha256": config_sha256,
            "folds": folds,
        }
    if all_tokens[ARMS[0]] != all_tokens[ARMS[1]]:
        raise ValueError("reused BASE/DYN token universes differ")
    diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary", f"{args.source_commit}..{observed_commit}"]
    )
    payload = sign_artifact(
        {
            "artifact_type": "eclock_x0_cross_commit_reuse_provenance_v1",
            "authorization": (
                "Explicit user override: reuse verified dc17643 BASE/DYN and train PAIR-U "
                "new on the current HEAD."
            ),
            "source_campaign": str(source),
            "destination_campaign": str(destination),
            "source_training_commit": args.source_commit,
            "current_finalization_commit": observed_commit,
            "reused_arms": list(ARMS),
            "new_training_arms": ["X0-PAIR-U"],
            "official_external_replay_arms": ["X0-A5-REPLAY"],
            "training_surface_git_diff_empty": True,
            "training_surface_pathspecs": list(TRAINING_SURFACE),
            "post_source_changed_files": sorted(changed),
            "post_source_diff_sha256": _sha(diff),
            "source_failure_boundary": {
                "message": failure["message"],
                "comparison_log_sha256": compute_file_hash(str(comparison_log)),
                "classification": "reader_only_post_training_failure",
            },
            "arms": arm_records,
            "token_universe_count": 8192,
            "source_artifacts_copied_byte_for_byte": True,
            "aggregates_and_comparisons_must_be_recomputed": True,
        }
    )
    output = destination / "provenance_exception.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
