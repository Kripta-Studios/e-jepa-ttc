"""A valid CRC must not hide a manifest/hash mismatch in the delivery bundle."""

import hashlib
import json
import zipfile

import pytest

from scripts.verify_v17_bundle import verify


def make_zip(path, *, tamper=False, unsafe=False):
    name = "../outside.md" if unsafe else "source/report.md"
    content = b"scientific evidence\n"
    manifest = {
        "scope": {},
        "files": [
            {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, content + b"changed" if tamper else content)
        archive.writestr("MANIFEST.json", json.dumps(manifest))


def test_bundle_sha_crc_and_exact_manifest_coverage(tmp_path):
    path = tmp_path / "bundle.zip"
    make_zip(path)
    result = verify(path)
    assert result["status"] == "PASS" and result["manifest_files"] == 1 and result["members"] == 2


@pytest.mark.parametrize("case", ["tamper", "unsafe"])
def test_bundle_rejects_content_or_path_tampering(tmp_path, case):
    path = tmp_path / "bundle.zip"
    make_zip(path, tamper=case == "tamper", unsafe=case == "unsafe")
    with pytest.raises(ValueError):
        verify(path)
