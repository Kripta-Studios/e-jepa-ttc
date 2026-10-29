"""Cloud path equivalence must never weaken scientific row or file identity."""

import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

from operational.runpod_migration.consume import CloudMappings
from operational.runpod_migration.return_endpoint import canonical_role_equal, verify_snapshot_files
from operational.runpod_migration.return_rgb_downstream import verify_downstream_authority
from operational.runpod_migration.return_verify import ReturnPathMap


def mapping(tmp_path):
    (tmp_path / "rows.parquet").write_bytes(b"rows")
    (tmp_path / "roles.json").write_bytes(b"roles")
    value = {
        "run_root": "/workspace/cloud",
        "bundle_root": "/workspace/bundle",
        "roots": {str(tmp_path): "original"},
    }
    return ReturnPathMap(value, tmp_path)


def test_only_path_locations_are_normalized(tmp_path):
    mapper = mapping(tmp_path)
    left = {
        "rows_path": "/workspace/cloud/rows.parquet",
        "manifest_path": "/workspace/cloud/roles.json",
        "rows_sha256": "a" * 64,
        "group_ids": ["g1", "g2"],
    }
    right = {
        **left,
        "rows_path": str(tmp_path / "rows.parquet"),
        "manifest_path": str(tmp_path / "roles.json"),
    }
    assert canonical_role_equal(left, right, mapper)
    assert not canonical_role_equal(left, {**right, "rows_sha256": "b" * 64}, mapper)
    assert not canonical_role_equal(left, {**right, "group_ids": ["g2", "g1"]}, mapper)
    assert left["rows_path"].startswith("/workspace/")


@pytest.fixture
def posix_mapping(tmp_path, monkeypatch):
    mapper = ReturnPathMap(
        {
            "run_root": "/workspace/cloud",
            "bundle_root": "/workspace/bundle",
            "roots": {"/tmp/registered-original": "original"},
        },
        tmp_path,
    )
    # Exercise the Linux destination spelling even when tests run on Windows.
    monkeypatch.setattr(mapper, "downloaded", PurePosixPath("/tmp/registered-download"))
    return mapper


@pytest.mark.parametrize("suffix", ["", "/roles.json", "/nested/rows.parquet"])
def test_registered_posix_paths_are_idempotent(posix_mapping, suffix):
    downloaded = "/tmp/registered-download" + suffix
    assert posix_mapping.translate("/workspace/cloud" + suffix) == downloaded
    assert posix_mapping.translate(downloaded) == downloaded
    original = "/tmp/registered-original" + suffix
    assert posix_mapping.translate(original) == original
    assert posix_mapping.translate("/workspace/bundle/original" + suffix) == str(Path(original))


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/unregistered/roles.json",
        "/tmp/registered-download-sibling/roles.json",
        "/tmp/registered-original-sibling/roles.json",
        "/workspace/cloud-sibling/roles.json",
        "/workspace/bundle/original-sibling/roles.json",
    ],
)
def test_posix_local_and_cloud_siblings_remain_unregistered(posix_mapping, path):
    with pytest.raises(ValueError, match="Unregistered"):
        posix_mapping.translate(path)


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/registered-download/../roles.json",
        "/tmp/registered-original/../roles.json",
        "/workspace/cloud/../roles.json",
        "/workspace/bundle/original/../roles.json",
    ],
)
def test_registered_posix_roots_do_not_admit_traversal(posix_mapping, path):
    with pytest.raises(ValueError, match="traversal"):
        posix_mapping.translate(path)


def test_unknown_cloud_outputs_and_traversal_fail_closed(tmp_path):
    mapper = mapping(tmp_path)
    with pytest.raises(ValueError):
        canonical_role_equal({"rows_path": "/workspace/cloud/../rows.parquet"}, {}, mapper)
    with pytest.raises(ValueError, match="Unregistered"):
        CloudMappings([]).translate("/workspace/ejepa_technical/unknown/checkpoint.pt")


def test_snapshot_requires_exact_coverage_and_hashes(tmp_path):
    tmp_path = tmp_path / "snapshot"
    tmp_path.mkdir()
    payload = b"saved"
    (tmp_path / "state.pt").write_bytes(payload)
    manifest = {
        "schema": "ejepa_cloud_checkpoint_snapshot_v1",
        "files": {
            "state.pt": {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        },
    }
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    assert verify_snapshot_files(tmp_path) == manifest
    (tmp_path / "state.pt").write_bytes(b"other")
    with pytest.raises(ValueError, match="changed"):
        verify_snapshot_files(tmp_path)
    (tmp_path / "extra").write_text("not admitted")
    with pytest.raises(ValueError, match="coverage"):
        verify_snapshot_files(tmp_path)


def test_rgb_downstream_requires_exact_archive_bytes_and_task(tmp_path):
    archive = tmp_path / "artifacts/runpod_returns/test/complete"
    root = archive / "roots/r2_downstream"
    root.mkdir(parents=True)
    payload = json.dumps({"task": {"rotation": 2, "fit": "EVENTS"}}).encode()
    target = root / "RGB_DOWNSTREAM_ADMISSION.json"
    target.write_bytes(payload)
    manifest = {
        "schema": "ejepa_cloud_rgb_complete_archive_v1",
        "status": "READY",
        "files": {
            "roots/r2_downstream/RGB_DOWNSTREAM_ADMISSION.json": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        },
    }
    (archive / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="assignment"):
        verify_downstream_authority(root, tmp_path)
    target.write_bytes(payload + b" ")
    with pytest.raises(ValueError, match="changed"):
        verify_downstream_authority(root, tmp_path)
    (root / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="coverage"):
        verify_downstream_authority(root, tmp_path)
