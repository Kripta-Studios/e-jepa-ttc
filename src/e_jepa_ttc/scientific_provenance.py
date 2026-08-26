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


def assert_autopsy_replay_producer_reusable(
    payload: Mapping[str, Any],
    *,
    expected_commit: str,
    source: str = "replay manifest",
) -> None:
    """Refuse autopsy replay reuse unless producer identity matches this HEAD.

    Signed CSV hashes prove a completed replay, not that the replay was produced
    by the current implementation.  Missing git_commit, a dirty producer, or a
    commit mismatch are fatal.
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
    if commit != expected_commit:
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
