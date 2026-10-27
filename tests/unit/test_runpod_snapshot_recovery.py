"""Recovery between pinned directory publication and archive publication."""

import hashlib
import json
import tarfile

import pytest

from scripts.runpod_snapshot_v1 import publish_archive


def test_archive_recovers_from_verified_published_directory(tmp_path):
    target = tmp_path / "pinned"
    target.mkdir()
    payload = b"checkpoint"
    (target / "state.pt").write_bytes(payload)
    manifest = {
        "files": {
            "state.pt": {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        }
    }
    (target / "MANIFEST.json").write_text(json.dumps(manifest))
    archive = tmp_path / "snapshot.tar"
    publish_archive(target, archive)
    with tarfile.open(archive) as stream:
        assert stream.extractfile("state.pt").read() == payload
    (target / "state.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs"):
        publish_archive(target, tmp_path / "another.tar")


def test_archive_rejects_extra_files(tmp_path):
    (tmp_path / "MANIFEST.json").write_text('{"files": {}}')
    (tmp_path / "unexpected").write_text("extra")
    with pytest.raises(ValueError, match="missing or extra"):
        publish_archive(tmp_path, tmp_path / "result.tar")
