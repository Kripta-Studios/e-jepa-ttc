from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.upload_rgb_sparse_frames_v3 import RECEIVER, can_schedule, canonical, load_ready


def ready(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    tar = root / "data/train/sequence/rgb_shards/rgb-00000.tar"
    tar.parent.mkdir(parents=True)
    tar.write_bytes(b"exact-tar")
    value = {
        "schema": "ejepa_rgb_sparse_tar_receipt_v1",
        "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
        "relative": "data/train/sequence/rgb_shards/rgb-00000.tar",
        "sparse_tar_bytes": tar.stat().st_size,
        "sparse_tar_sha256": hashlib.sha256(tar.read_bytes()).hexdigest(),
        "payload_recompressed": False,
        "decoded_pixel_parity": True,
        "plan_sha256": "a" * 64,
        "members": [{"roles": ["refiner"], "member": "rgb/a.png"}],
    }
    value["receipt_sha256"] = canonical(value, "receipt_sha256")
    marker = tar.with_name(tar.name + ".json")
    marker.write_text(json.dumps(value), encoding="utf-8")
    return root, marker


def test_verified_ready_binds_tar_and_role(tmp_path: Path) -> None:
    root, marker = ready(tmp_path)
    item = load_ready(marker, root)
    assert item.refiner is True
    assert item.relative.endswith("rgb-00000.tar")
    assert item.size == len(b"exact-tar")
    assert item.plan_sha256 == "a" * 64


@pytest.mark.parametrize("field", ["status", "sparse_tar_sha256", "receipt_sha256"])
def test_changed_or_incomplete_marker_fails_closed(tmp_path: Path, field: str) -> None:
    root, marker = ready(tmp_path)
    value = json.loads(marker.read_text(encoding="utf-8"))
    value[field] = "different"
    marker.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_ready(marker, root)


def test_global_reserve_includes_existing_and_new_inflight() -> None:
    assert can_schedule(40, 10, 15, 14) is True
    assert can_schedule(40, 10, 17, 14) is False


def test_embedded_remote_receiver_is_valid_python() -> None:
    compile(RECEIVER, "<rgb-remote-receiver>", "exec")
