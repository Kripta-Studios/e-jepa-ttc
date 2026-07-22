"""Size and optional SHA-256 verification for selective dataset manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def verify_file_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    sha256: bool = False,
) -> dict[str, Any]:
    """Verify exact paths without interpreting missing large files as partial data."""

    base = Path(root).resolve()
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        msg = "Dataset integrity manifest must contain a non-empty files object."
        raise ValueError(msg)
    rows: list[dict[str, Any]] = []
    expected_total = 0
    present_total = 0
    for relative, metadata in sorted(files.items()):
        path = (base / relative).resolve()
        if base != path and base not in path.parents:
            msg = f"Manifest path escapes dataset root: {relative!r}."
            raise ValueError(msg)
        expected_size = int(metadata["size"])
        expected_total += expected_size
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else 0
        present_total += actual_size
        size_matches = exists and actual_size == expected_size
        digest: str | None = None
        hash_matches: bool | None = None
        expected_hash = metadata.get("lfs_sha256")
        if sha256 and size_matches and expected_hash is not None:
            digest = _sha256(path)
            hash_matches = digest == expected_hash
        rows.append(
            {
                "path": relative,
                "exists": exists,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "size_matches": size_matches,
                "sha256": digest,
                "sha256_matches": hash_matches,
            }
        )
    valid = all(row["size_matches"] and row["sha256_matches"] is not False for row in rows)
    return {
        "format": payload.get("format"),
        "repository": payload.get("repository"),
        "root": base.as_posix(),
        "file_count": len(rows),
        "present_file_count": sum(bool(row["exists"]) for row in rows),
        "valid_file_count": sum(
            bool(row["size_matches"] and row["sha256_matches"] is not False) for row in rows
        ),
        "expected_total_bytes": expected_total,
        "present_total_bytes": present_total,
        "sha256_checked": sha256,
        "valid": valid,
        "files": rows,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["verify_file_manifest"]
