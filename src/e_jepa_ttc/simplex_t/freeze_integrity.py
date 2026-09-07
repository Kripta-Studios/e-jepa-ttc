"""Read-only integrity checks for scientific freeze inputs.

Byte integrity is necessary, not authority: the caller must independently verify
roles, transitive producer exclusions, QA, resource admission and practical gates.
This module neither publishes a freeze nor authorizes an optimizer update.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from e_jepa_ttc.artifacts.hashing import compute_file_hash

from .phase_manifest import fit_key
from .registry import registered_graph

REQUIRED_CATEGORIES = frozenset(
    {"code", "config", "roles", "time", "producers", "normalizers", "schemas", "qa"}
)


def _digest(value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA256")


@dataclass(frozen=True)
class FrozenFile:
    """A named file under an explicitly permitted read root, with exact bytes."""

    category: str
    root: str
    relative_path: str
    sha256: str


def verify_code_commit(files: list[FrozenFile], roots: dict[str, Path], commit: str) -> None:
    """Bind pinned companion code bytes to an existing Git commit, read-only.

    Every Python/PowerShell source under src/scripts must be pinned, including
    newly added files; untracked executable code cannot be silently omitted.
    Unrelated untracked documents do not invalidate this scoped check.
    Code must be inside the explicit companion ``work`` root; imported producer
    artifacts have separate producer pins, not a claim of belonging to this commit.
    No checkout, index update, filter, or repository configuration is performed.
    """
    if len(commit) != 40 or set(commit) - set("0123456789abcdef") or "work" not in roots:
        raise ValueError("full source commit and companion work root required")
    work = roots["work"].resolve(strict=True)

    def git(*args: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(work), *args], capture_output=True, check=False, timeout=30
        )
        if result.returncode:
            raise ValueError("cannot verify scientific code against declared Git commit")
        return result.stdout

    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != work:
        raise ValueError("companion work root must be the Git worktree root")
    if git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() != commit:
        raise ValueError("declared source object is not an exact commit")
    committed_paths = {
        item.decode("utf-8")
        for item in git("ls-tree", "-r", "-z", "--name-only", commit, "--", "src", "scripts").split(
            b"\0"
        )
        if item
    }
    local_paths = {
        path.relative_to(work).as_posix()
        for folder in ("src", "scripts")
        if (work / folder).exists()
        for path in (work / folder).rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".ps1", ".psm1"}
    }
    required_code = {
        relative
        for relative in committed_paths | local_paths
        if Path(relative).suffix.lower() in {".py", ".ps1", ".psm1"}
    }
    code = [pin for pin in files if pin.category == "code"]
    if not code:
        raise ValueError("scientific freeze requires committed code pins")
    declared_code = {Path(pin.relative_path).as_posix() for pin in code if pin.root == "work"}
    missing = sorted(required_code - declared_code)
    if missing:
        raise ValueError(f"scientific freeze omits executable source inventory: {missing[:8]}")
    for pin in code:
        relative = Path(pin.relative_path)
        if pin.root != "work" or relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("companion code pin must remain under work")
        path = (work / relative).resolve(strict=True)
        if not path.is_relative_to(work) or path.stat().st_size > 16_777_216:
            raise ValueError("code pin escapes work or exceeds source file bound")
        blob = git("cat-file", "blob", f"{commit}:{relative.as_posix()}")
        if hashlib.sha256(blob).hexdigest() != pin.sha256:
            raise ValueError(f"code pin differs from declared commit: {pin.relative_path}")
        if compute_file_hash(str(path)) != pin.sha256:
            raise ValueError(f"working source differs from committed code pin: {pin.relative_path}")


def verify_files(files: list[FrozenFile], roots: dict[str, Path]) -> None:
    """Reject incomplete pins, aliases and root escapes before hashing any file.

    Roots are supplied by the audited local configuration, not by the freeze
    document. Resolving symlinks prevents a relative path escaping its root.
    Hashing streams files; it does not deserialize checkpoints or dataset labels.
    """
    if {item.category for item in files} != REQUIRED_CATEGORIES:
        raise ValueError("freeze categories are missing or unregistered")
    resolved_roots = {key: path.resolve(strict=True) for key, path in roots.items()}
    paths: list[Path] = []
    for item in files:
        _digest(item.sha256)
        relative = Path(item.relative_path)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("freeze path must be relative without traversal")
        if item.root not in resolved_roots:
            raise ValueError("freeze references an unapproved root")
        root = resolved_roots[item.root]
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("freeze file outside approved root or not a file")
        if path in paths:
            raise ValueError("freeze file aliases another pin")
        paths.append(path)
    for item, path in zip(files, paths, strict=True):
        if compute_file_hash(str(path)) != item.sha256:
            raise ValueError(f"frozen {item.category} bytes changed: {item.relative_path}")


def verify_stage_sources(
    *,
    stage: str,
    availability: dict[str, bool],
    frozen: dict[str, dict[str, str]],
    observed: dict[str, dict[str, str]],
) -> None:
    """Bind both TRAIN normalization and unchanged OLD evaluation identities.

    Exact stage membership is mandatory: neither omitted controls nor an extra
    renamed fit is accepted. Availability must already have separate evidence;
    matching booleans here does not establish that scientific gate evidence.
    """
    if set(availability) != {
        "d1",
        "density",
        "t3",
        "latent",
        "replicate_scalar",
        "replicate_latent",
    } or any(type(value) is not bool for value in availability.values()):
        raise ValueError("all availability fields require resolved booleans")
    expected = {fit_key(spec) for spec in registered_graph(**availability) if spec.stage == stage}
    if not expected or set(frozen) != expected or set(observed) != expected:
        raise ValueError("source set differs from complete registered stage")
    for key in sorted(expected):
        for sources in (frozen[key], observed[key]):
            if set(sources) != {"inner_oof", "outer_dev"}:
                raise ValueError("both TRAIN and OLD_DEV identities required")
            for digest in sources.values():
                _digest(digest)
        if frozen[key] != observed[key]:
            raise ValueError(f"source or TRAIN-fitted normalization changed: {key}")
