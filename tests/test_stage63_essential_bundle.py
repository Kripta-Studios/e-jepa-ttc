"""Verify actual closed ZIP members, not just the outer archive checksum."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.package_scientific_recovery_v9_stage63_65 import _verify_bundle


@pytest.mark.parametrize("corrupt", [False, True])
def test_closed_zip_checks_member_bytes(tmp_path: Path, corrupt: bool) -> None:
    path = tmp_path / "essential.zip"
    content = b"small router coefficients"
    manifest = {
        "files": [
            {
                "path": "router.npz",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ]
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("router.npz", b"changed coefficients" if corrupt else content)
        archive.writestr("ESSENTIAL_MANIFEST.json", json.dumps(manifest))
    if corrupt:
        with pytest.raises(ValueError, match="member identity"):
            _verify_bundle(path, {"router.npz"})
    else:
        _verify_bundle(path, {"router.npz"})
        with pytest.raises(ValueError, match="contractual"):
            _verify_bundle(path, {"missing-required-fit.npz"})
