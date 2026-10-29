from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from scripts.preflight_detection_rgb_assets import sha256_file
from scripts.revise_detection_campaign_media import _admit_candidate, _coverage


def frame(token: str, identities: dict[str, list[str]]) -> dict[str, object]:
    return {
        "sample_token": token,
        "sequence_id": "seq",
        "raw_counts": {"bicycle": 1},
        "evaluable_counts": {"bicycle": 1},
        "trainable_counts": {"bicycle": 1},
        "trainable_identities": identities,
        "trainer_overflow": False,
    }


def png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (1280, 720), color=(1, 2, 3)).save(stream, format="PNG")
    return stream.getvalue()


def make_tar(root: Path, raw: bytes) -> tuple[str, str]:
    relative = "data/train/seq/rgb_shards/rgb-00000.tar"
    member = "rgb/frame.png"
    path = root / relative
    path.parent.mkdir(parents=True)
    with tarfile.open(path, "w:") as archive:
        info = tarfile.TarInfo(member)
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    return relative, member


def test_coverage_counts_distinct_sequence_track_identities() -> None:
    result = _coverage(
        [
            frame("a", {"bicycle": ["1"]}),
            frame("b", {"bicycle": ["1", "2"]}),
        ]
    )
    assert result["frame_count"] == 2
    assert result["trainable_annotations_legacy_profile"]["bicycle"] == 2
    assert result["trainable_identity_count"]["bicycle"] == 2


def test_candidate_admission_binds_tar_and_member_bytes(tmp_path: Path) -> None:
    raw = png_bytes()
    relative, member = make_tar(tmp_path, raw)
    tar_path = tmp_path / relative
    expected = {
        relative: {"size_bytes": tar_path.stat().st_size, "sha256": sha256_file(tar_path)}
    }
    evidence = _admit_candidate(
        root=tmp_path.resolve(),
        row={
            "sample_token": "seq:new",
            "sequence_id": "seq",
            "rgb_shard_path": relative,
            "rgb_member_path": member,
        },
        pinned=expected,
        tar_hash_cache={},
    )
    assert evidence["member_size_bytes"] == len(raw)
    assert evidence["member_sha256"] == hashlib.sha256(raw).hexdigest()
    assert evidence["image_wh"] == [1280, 720]


def test_candidate_admission_rejects_truncated_png(tmp_path: Path) -> None:
    raw = png_bytes()[:100]
    relative, member = make_tar(tmp_path, raw)
    tar_path = tmp_path / relative
    expected = {
        relative: {"size_bytes": tar_path.stat().st_size, "sha256": sha256_file(tar_path)}
    }
    with pytest.raises(OSError):
        _admit_candidate(
            root=tmp_path.resolve(),
            row={
                "sample_token": "seq:new",
                "sequence_id": "seq",
                "rgb_shard_path": relative,
                "rgb_member_path": member,
            },
            pinned=expected,
            tar_hash_cache={},
        )
