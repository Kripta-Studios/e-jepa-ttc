from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def read_artifact(relative_path: str) -> dict[str, Any]:
    """Read optional, locally generated integration evidence.

    Run/checkpoint trees are deliberately ignored by Git.  A clean clone must
    therefore skip evidence-only assertions instead of failing because another
    machine's generated artifact is absent.
    """
    path = ROOT / relative_path
    if not path.is_file():
        pytest.skip(f"Optional local evidence artifact is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


def artifact_path(relative_path: str) -> Path:
    """Resolve an optional local evidence path or skip its contract test."""
    path = ROOT / relative_path
    if not path.exists():
        pytest.skip(f"Optional local evidence artifact is absent: {path}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
