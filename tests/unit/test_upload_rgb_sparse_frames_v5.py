from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import upload_rgb_sparse_frames_v2 as v2
from scripts.upload_rgb_sparse_frames_v5 import (
    digest,
    remote_staged_credit,
    validate_complete_manifest,
)


def _canonical(value: dict[str, object]) -> str:
    encoded = json.dumps(
        {key: item for key, item in value.items() if key != "receipt_sha256"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _complete_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    shard = root / "data" / "train" / "sequence" / "rgb_shards"
    shard.mkdir(parents=True)
    tar = shard / "rgb-00000.tar"
    tar.write_bytes(b"verified sparse tar")
    relative = "data/train/sequence/rgb_shards/rgb-00000.tar"
    record: dict[str, object] = {
        "schema": "ejepa_rgb_sparse_tar_receipt_v1",
        "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
        "relative": relative,
        "sparse_tar_bytes": tar.stat().st_size,
        "sparse_tar_sha256": digest(tar),
        "payload_recompressed": False,
        "decoded_pixel_parity": True,
        "members": [{"member": "rgb/1.png", "roles": ["generation"]}],
    }
    record["receipt_sha256"] = _canonical(record)
    marker = tar.with_name(tar.name + ".json")
    marker.write_text(json.dumps(record), encoding="utf-8")
    manifest = {
        "schema": "ejepa_rgb_frame_supplement_v1",
        "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
        "physical_root": str(root.resolve()),
        "payload_recompressed": False,
        "decoded_pixel_parity": True,
        "optimizer_updates": 0,
        "tar_count": 1,
        "sparse_tar_bytes": tar.stat().st_size,
        "tars": [record],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest_path


def test_complete_manifest_binds_exact_ready_set_and_root(tmp_path: Path) -> None:
    root, manifest = _complete_fixture(tmp_path)
    ready, value = validate_complete_manifest(manifest, digest(manifest), root, v2.load_ready)
    assert [item.relative for item in ready] == [value["tars"][0]["relative"]]


def test_complete_manifest_rejects_different_physical_root(tmp_path: Path) -> None:
    root, manifest = _complete_fixture(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="physical root differs"):
        validate_complete_manifest(manifest, digest(manifest), other, v2.load_ready)


def test_remote_resume_credit_is_bounded_by_expected_files() -> None:
    report = {"credit_bytes": 75, "target_files": 1, "partial_files": 1}
    credit, observed = remote_staged_credit(
        ["ssh"],
        "/workspace/ejepa_rgb_frames",
        [("data/a.tar", 50), ("data/b.tar", 50)],
        runner=lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(report)),
    )
    assert credit == 75
    assert observed == report


def test_remote_resume_credit_rejects_impossible_total() -> None:
    with pytest.raises(ValueError, match="outside pending"):
        remote_staged_credit(
            ["ssh"],
            "/workspace/ejepa_rgb_frames",
            [("data/a.tar", 50)],
            runner=lambda *_args, **_kwargs: SimpleNamespace(
                stdout=json.dumps(
                    {"credit_bytes": 51, "target_files": 0, "partial_files": 1}
                )
            ),
        )


def test_remote_resume_credit_rejects_unsafe_relative() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        remote_staged_credit(
            ["ssh"],
            "/workspace/ejepa_rgb_frames",
            [("../escape", 1)],
            runner=lambda *_args, **_kwargs: None,
        )
