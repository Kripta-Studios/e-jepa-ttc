"""Bind every cache component to its bytes and on-disk tensor layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash


def component_record(path: Path) -> dict[str, Any]:
    """Inspect a component without materializing large NumPy arrays."""
    result: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": compute_file_hash(str(path)),
    }
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        result.update(dtype=str(array.dtype), shape=list(array.shape))
    return result


def verify_components(root: Path, records: dict[str, Any], required: set[str]) -> None:
    """Reject missing, substituted, escaped or layout-altered cache components."""
    if set(records) != required:
        raise ValueError("cache component inventory differs from its contract")
    resolved = root.resolve(strict=True)
    for name, expected in records.items():
        path = (root / name).resolve(strict=True)
        if path.parent != resolved or expected.get("path") != name:
            raise ValueError("cache component path escapes its root or identity")
        if component_record(path) != expected:
            raise ValueError(f"cache component bytes/layout mismatch: {name}")
