"""Bind diagnostic execution to audited candidate bytes and source snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .common import file_sha


def bind_candidate(binding: dict[str, Any], protocol: dict[str, Any]) -> None:
    """Reject a candidate different from the protocol's audited component graph."""
    expected = protocol.get("candidate_binding")
    if not expected or binding.get("components") != expected:
        raise ValueError("Diagnostic candidate does not match protocol lineage binding")


def snapshot_sources(root: Path, names: Sequence[str] | None = None) -> dict[str, str]:
    """Hash local Python sources for the whole scientific graph plus its orchestration."""
    if names is None:
        files = [*root.glob("src/e_jepa_ttc/**/*.py"), *root.glob("operational/**/*.py")]
        files += [
            root / "scripts/audit_evttc_transfer.py",
            root / "scripts/final_train40_transfer.py",
        ]
        names = sorted({p.relative_to(root).as_posix() for p in files})
    return {name: file_sha(root / name) for name in names}


def assert_sources_unchanged(root: Path, snapshot: dict[str, str]) -> None:
    """Fail if preparation, geometry, runtime or other bound source changed during a run."""
    if snapshot != snapshot_sources(root, list(snapshot)):
        raise ValueError("Source changed during diagnosis; retain traces as an incomplete attempt")
