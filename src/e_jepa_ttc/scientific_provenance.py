"""Fail-closed Git and environment provenance for scientific V8 jobs."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_SCIENTIFIC_ENV = (
    "DINO_NUM_CHUNKS",
    "DINO_CHUNK_INDEX",
    "DINO_START_ROW",
    "DINO_END_ROW",
    "DINO_ALLOW_PARTIAL_CACHE",
    "ALLOW_DIRTY_MATERIALIZE",
    "ALLOW_DIRTY",
    "ALLOW_PARTIAL",
)

# Files whose content can change A5/C2F/Garl autopsy replay identity.
# A HEAD advance that does not touch these paths may reuse a clean producer.
AUTOPSY_REPLAY_IDENTITY_PATHS = (
    "scripts/replay_scientific_recovery_v8_mechanisms.py",
    "scripts/materialize_scientific_recovery_v8_autopsy_inputs.py",
    "scripts/materialize_scientific_recovery_v8_garl_autopsy_comparator.py",
    "src/e_jepa_ttc/models/causal_scale_ttc.py",
    "src/e_jepa_ttc/models/garl_ttc_replica.py",
    "src/e_jepa_ttc/evaluation/scientific_recovery_v8.py",
    "configs/protocol/scientific_recovery_v8_temporal.json",
    "configs/experiment/scientific_recovery_v8_fold_chain/frozen_manifest.json",
)


class ScientificProvenanceError(RuntimeError):
    """Raised when scientific execution cannot bind a clean, observed identity."""


def observe_git_identity(root: Path | None = None) -> dict[str, Any]:
    """Return the observed Git commit and dirty flag. Never invent cleanliness."""

    cwd = Path(ROOT if root is None else root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ScientificProvenanceError(f"Git identity cannot be observed: {error}") from error
    dirty = bool(porcelain.strip())
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "git_status_porcelain": porcelain,
    }


def serialize_git_identity(observed: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize observed Git identity without substituting expected cleanliness."""

    commit = observed.get("git_commit")
    dirty = observed.get("git_dirty")
    if not isinstance(commit, str) or not commit:
        raise ScientificProvenanceError("git_commit must be the observed HEAD")
    if not isinstance(dirty, bool):
        raise ScientificProvenanceError("git_dirty must be the observed boolean")
    if dirty is False and str(observed.get("git_status_porcelain", "")).strip():
        raise ScientificProvenanceError(
            "git_dirty=false is forbidden when Git porcelain is non-empty"
        )
    return {"git_commit": commit, "git_dirty": dirty}


def require_clean_scientific_worktree(root: Path | None = None) -> dict[str, Any]:
    """Abort scientific execution when the worktree is dirty."""

    identity = observe_git_identity(root)
    serialized = serialize_git_identity(identity)
    if serialized["git_dirty"] is not False:
        raise ScientificProvenanceError(
            f"scientific execution requires a clean Git worktree; commit={serialized['git_commit']}"
        )
    return serialized


def refuse_scientific_bypass_env(environ: Mapping[str, str] | None = None) -> None:
    """Fail if a scientific bypass environment variable is set."""

    source = os.environ if environ is None else environ
    present = [name for name in FORBIDDEN_SCIENTIFIC_ENV if str(source.get(name, "")).strip()]
    if present:
        raise ScientificProvenanceError(
            "scientific execution forbids bypass environment variables: " + ", ".join(present)
        )


def repo_relative_posix(path: Path, *, root: Path | None = None) -> str:
    """Return a POSIX path relative to the repository after resolving both sides.

    Windows ``Path.relative_to`` is fatal when one path is relative and the other
    is absolute.  Callers that record artifact paths from a relative ``--output-dir``
    must resolve first.
    """

    base = Path(ROOT if root is None else root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError as error:
        raise ScientificProvenanceError(f"path escapes repository: {path}") from error


def autopsy_replay_identity_paths_changed(
    producer_commit: str,
    head_commit: str,
    *,
    root: Path | None = None,
) -> bool:
    """Return True when git history between two commits touches replay identity files.

    Comparison failure is fatal: a producer that cannot be compared to HEAD cannot
    be reused.
    """

    cwd = Path(ROOT if root is None else root)
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", producer_commit, head_commit],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ScientificProvenanceError(
            f"cannot compare autopsy producer {producer_commit} to HEAD {head_commit}: {error}"
        ) from error
    changed = [
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    identity = set(AUTOPSY_REPLAY_IDENTITY_PATHS)
    return any(path in identity for path in changed)


def assert_autopsy_replay_producer_reusable(
    payload: Mapping[str, Any],
    *,
    expected_commit: str,
    source: str = "replay manifest",
) -> None:
    """Refuse autopsy replay reuse unless the producer is clean and identity-compatible.

    Signed CSV hashes prove a completed replay, not that the replay was produced
    by the current implementation.  Missing git_commit or a dirty producer are
    fatal.  A commit mismatch is fatal only when git history between the producer
    and HEAD touches files that can change replay identity.
    """

    if payload.get("status") != "completed_replay_without_optimizer_steps":
        raise ScientificProvenanceError(f"{source} is incomplete")
    commit = payload.get("git_commit")
    dirty = payload.get("git_dirty")
    if not isinstance(commit, str) or not commit:
        raise ScientificProvenanceError(
            f"{source} has no git_commit; existence-only reuse is forbidden "
            "after implementation repair"
        )
    if dirty is not False:
        raise ScientificProvenanceError(f"{source} was not produced from a clean worktree")
    if not isinstance(expected_commit, str) or not expected_commit:
        raise ScientificProvenanceError("implementation HEAD git_commit is missing")
    if commit == expected_commit:
        return
    try:
        identity_changed = autopsy_replay_identity_paths_changed(commit, expected_commit)
    except ScientificProvenanceError as error:
        raise ScientificProvenanceError(
            f"{source} git_commit {commit} differs from implementation HEAD "
            f"{expected_commit}; {error}"
        ) from error
    if identity_changed:
        raise ScientificProvenanceError(
            f"{source} git_commit {commit} differs from implementation HEAD {expected_commit}"
        )


def assert_router_expert_reusable(payload: Mapping[str, Any]) -> None:
    """Reject dirty, unsigned, fixture, or failed-integrity router experts."""

    if payload.get("status") in {"failed_integrity", "failed_gate"}:
        raise ScientificProvenanceError("router expert is classified failed_integrity")
    if payload.get("fixture") is True:
        raise ScientificProvenanceError("fixture router expert cannot be aggregated")
    if payload.get("git_dirty") is not False:
        raise ScientificProvenanceError("router expert was produced from a dirty worktree")
    if payload.get("integrity_status") == "failed_integrity":
        raise ScientificProvenanceError("router expert integrity_status is failed_integrity")
