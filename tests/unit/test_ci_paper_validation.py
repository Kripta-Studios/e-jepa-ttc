"""Focused checks for the committed manuscript source-package gate."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.ci_paper_validation import (
    SOURCE_MANIFEST_NAME,
    SOURCE_ZIP_NAME,
    SOURCE_ZIP_PREFIX,
    _safe_manifest_path,
    verify_source_package,
)


def make_package(tmp_path: Path, *, archived_data: bytes = b"published bytes") -> Path:
    paper = tmp_path / "paper"
    source = paper / "data" / "result.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"published bytes")
    manifest = {
        "schema": "test_source_manifest_v1",
        "files": [
            {
                "path": "data/result.json",
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    (paper / SOURCE_MANIFEST_NAME).write_bytes(manifest_bytes)
    with zipfile.ZipFile(paper / SOURCE_ZIP_NAME, "w") as archive:
        archive.writestr(f"{SOURCE_ZIP_PREFIX}/data/result.json", archived_data)
        archive.writestr(f"{SOURCE_ZIP_PREFIX}/{SOURCE_MANIFEST_NAME}", manifest_bytes)
    return paper


def test_source_package_matches_manifest_and_local_bytes(tmp_path: Path) -> None:
    paper = make_package(tmp_path)
    assert verify_source_package(paper) == {"manifest_files": 1, "zip_members": 2}


def test_source_package_rejects_changed_archived_bytes(tmp_path: Path) -> None:
    paper = make_package(tmp_path, archived_data=b"changed")
    with pytest.raises(ValueError, match="Archived source differs"):
        verify_source_package(paper)


def test_source_package_rejects_unexpected_zip_member(tmp_path: Path) -> None:
    paper = make_package(tmp_path)
    with zipfile.ZipFile(paper / SOURCE_ZIP_NAME, "a") as archive:
        archive.writestr(f"{SOURCE_ZIP_PREFIX}/unexpected.txt", b"unexpected")
    with pytest.raises(ValueError, match="unexpected=.*unexpected.txt"):
        verify_source_package(paper)


@pytest.mark.parametrize(
    "path", ["../secret", "/absolute", "C:/secret", "data\\result.json", "data//result"]
)
def test_source_package_rejects_unsafe_or_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError, match="Unsafe or noncanonical"):
        _safe_manifest_path(path)
