"""Portable receipts and strict structured input helpers."""

# ruff: noqa: ANN401 -- arbitrary structured documents are validated by each audit boundary

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def file_sha(path: Path) -> str:
    """Hash bytes without loading the file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value: Any) -> str:
    """Hash a finite JSON value, independently of formatting."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_document(path: Path) -> Any:
    """Read a JSON or YAML configuration."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_receipt(path: Path, value: Any) -> None:
    """Write a new receipt; never silently replace earlier evidence."""
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Refusing to overwrite different evidence: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
