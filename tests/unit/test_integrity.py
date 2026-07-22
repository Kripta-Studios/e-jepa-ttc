from __future__ import annotations

import hashlib
import json
from pathlib import Path

from e_jepa_ttc.data.integrity import verify_file_manifest


def test_file_manifest_detects_missing_size_and_hash(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    first = root / "first.bin"
    first.write_bytes(b"verified")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "fixture",
                "repository": "fixture/repository",
                "files": {
                    "first.bin": {
                        "size": first.stat().st_size,
                        "lfs_sha256": hashlib.sha256(b"verified").hexdigest(),
                    },
                    "missing.bin": {"size": 4},
                },
            }
        ),
        encoding="utf-8",
    )

    result = verify_file_manifest(root, manifest, sha256=True)

    assert result["valid"] is False
    assert result["valid_file_count"] == 1
    assert result["present_file_count"] == 1
    assert result["files"][0]["sha256_matches"] is True
