"""Common fail-closed authorization for every scientific training entrypoint."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash
from e_jepa_ttc.artifacts.time_cap_amendment import training_identity_commit


def read_signed(path: Path) -> dict[str, Any]:
    """Read only an identity-verified artifact, never approve by file existence."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not verify_artifact_hash(value):
        raise ValueError(f"authorization artifact signature mismatch: {path}")
    return value


def bind_output_files(root: Path, paths: list[Path]) -> dict[str, Any]:
    """Bind complete result outputs for read-only endpoint resume."""
    return {
        str(path.resolve().relative_to(root.resolve())): {
            "bytes": path.stat().st_size,
            "sha256": compute_file_hash(str(path)),
        }
        for path in paths
    }


def verify_output_bindings(root: Path, value: dict[str, Any]) -> None:
    """A completed result is reusable only while every bound output remains exact."""
    bindings = value.get("output_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("completed result lacks output identities")
    for name, item in bindings.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root.resolve()):
            raise ValueError("completed result output path escapes attempt")
        if path.stat().st_size != item["bytes"] or compute_file_hash(str(path)) != item["sha256"]:
            raise ValueError(f"completed result output identity mismatch: {name}")


def validate_branch(root: Path, *, stage: int, seed: int, lock_sha256: str) -> None:
    """Enforce branch eligibility separately from bytes/code/invocation identity."""
    stage63 = read_signed(root / "stage63/X3_FEASIBILITY_V2.json")
    if stage63.get("integrity_passed") is not True:
        raise ValueError("scientific continuation requires Stage63 integrity")
    if stage == 64:
        if stage63.get("training_ready") is not True or seed not in (7, 13, 23):
            raise ValueError("Stage64 branch is not authorized")
        if seed == 7:
            return
    elif stage == 65:
        if any((root / "stage64" / f"seed{value}").exists() for value in (13, 23)):
            raise ValueError("Stage65 rescue after a raw replica attempt is forbidden")
        if stage63.get("decision") in ("RAW_DATA_UNAVAILABLE", "RAW_SUPPORT_INSUFFICIENT"):
            return
    else:
        raise ValueError("unknown scientific training stage")
    seed7 = read_signed(root / "stage64/seed7/STAGE64_RESULT.json")
    if seed7.get("training_lock_sha256") != lock_sha256:
        raise ValueError("seed7 authorization belongs to another training lock")
    if stage == 64:
        if (
            seed7.get("status") != "RAW_ALL_GATES_PASSED"
            or seed7.get("gates", {}).get("all_passed") is not True
            or seed7.get("all_endpoints_frozen_before_evaluation") is not True
        ):
            raise ValueError("replicas require all seed7 utility and system gates")
        for replica in (13, 23):
            prior_path = root / f"stage64/seed{replica}/STAGE64_RESULT.json"
            if prior_path.is_file():
                prior = read_signed(prior_path)
                if (
                    prior.get("training_lock_sha256") != lock_sha256
                    or prior.get("status") != "RAW_ALL_GATES_PASSED"
                    or prior.get("gates", {}).get("all_passed") is not True
                ):
                    raise ValueError("a failed or foreign raw replica terminates the campaign")
    elif seed7.get("status") != "RAW_GATES_FAILED":
        raise ValueError(
            "Stage65 requires an eligible raw scientific failure, not a technical error"
        )


def validate_training_authorization(
    root: Path,
    repo: Path,
    *,
    stage: int,
    seed: int = 7,
    invocation: dict[str, Any],
) -> dict[str, Any]:
    """Bind authorization to the clean code commit, complete inputs and QA receipt."""
    if (repo / "CAMPAIGN_INTEGRITY_HOLD.json").exists():
        raise ValueError("scientific training is under an integrity hold")
    lock_path = root / "TRAINING_LOCK.json"
    lock = read_signed(lock_path)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()
    scientific_commit = training_identity_commit(root, repo)
    if dirty or lock.get("training_commit") != scientific_commit:
        raise ValueError("training authorization requires the exact clean training commit")
    if lock.get("authorization_version") != "complete_identity_v2":
        raise ValueError("old training locks do not authorize the corrected campaign")
    if (
        lock.get("python") != sys.version
        or lock.get("python_executable") != sys.executable
        or lock.get("torch") != torch.__version__
        or lock.get("cuda_runtime") != torch.version.cuda
    ):
        raise ValueError("runtime differs from the QA and training lock environment")
    for name, value in invocation.items():
        if lock["invocation"].get(name) != value:
            raise ValueError(f"training invocation differs from lock: {name}")
    bindings = lock.get("input_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("training lock lacks input bindings")
    if stage == 65 and "stage65_prerequisites" not in bindings:
        raise ValueError("Stage65 sources were not frozen in the training lock")
    if stage == 64:
        smoke_path = root / "qa/STAGE64_REAL_TRAIN_ONLY_SMOKE.json"
        smoke = read_signed(smoke_path)
        if (
            compute_file_hash(str(smoke_path)) != lock.get("smoke_sha256")
            or smoke.get("status") != "passed"
            or smoke.get("identity", {}).get("training_commit") != scientific_commit
            or smoke["identity"].get("microbatch") != lock["invocation"]["microbatch"]
            or smoke["identity"].get("device") != lock["invocation"]["device"]
        ):
            raise ValueError("train-only smoke is not bound to the current training authorization")
    for name, binding in bindings.items():
        path = Path(binding["path"])
        if (
            path.stat().st_size != binding["bytes"]
            or compute_file_hash(str(path)) != binding["sha256"]
        ):
            raise ValueError(f"training input changed after freeze: {name}")
    qa = read_signed(root / "qa/QA_MANIFEST.json")
    if (
        qa.get("training_commit") != scientific_commit
        or qa.get("status") != "passed"
        or qa.get("input_bindings") != bindings
        or compute_file_hash(str(root / "qa/QA_MANIFEST.json")) != lock.get("qa_sha256")
    ):
        raise ValueError("QA receipt does not authorize the current code and inputs")
    evidence = qa.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("QA receipt lacks independently verifiable command evidence")
    for name, item in evidence.items():
        if compute_file_hash(str(Path(item["path"]))) != item["sha256"]:
            raise ValueError(f"QA evidence changed after acceptance: {name}")
    validate_branch(root, stage=stage, seed=seed, lock_sha256=compute_file_hash(str(lock_path)))
    return lock
