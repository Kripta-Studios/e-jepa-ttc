from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from scripts.runpod_rgb_input_cache_reclaim_v1 import (
    GIB,
    advise_file,
    build_inventory,
    canonical,
    cycle,
    digest,
)


def _seal(value: dict) -> dict:
    value["receipt_sha256"] = canonical(value)
    return value


def _fixture(tmp_path: Path) -> tuple[Path, list[tuple[Path, str]], list[Path]]:
    root = tmp_path / "media"
    manifests: list[tuple[Path, str]] = []
    tars: list[Path] = []
    for index in range(2):
        relative = f"data/train/s{index}/rgb_shards/rgb-00000.tar"
        tar = root / relative
        tar.parent.mkdir(parents=True)
        with tarfile.open(tar, "w") as archive:
            info = tarfile.TarInfo("rgb/a.png")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        record = _seal(
            {
                "schema": "ejepa_rgb_sparse_tar_receipt_v1",
                "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
                "relative": relative,
                "sparse_tar_bytes": tar.stat().st_size,
                "sparse_tar_sha256": digest(tar),
                "members": [{"member": "rgb/a.png"}],
            }
        )
        sidecar = tar.with_name(tar.name + ".json")
        sidecar.write_text(json.dumps(record), encoding="utf-8")
        tar.chmod(0o444)
        sidecar.chmod(0o444)
        manifest = _seal(
            {
                "schema": "ejepa_rgb_frame_supplement_v1",
                "status": "COMPLETE_BYTE_AND_PIXEL_VERIFIED",
                "payload_recompressed": False,
                "decoded_pixel_parity": True,
                "optimizer_updates": 0,
                "tar_count": 1,
                "sparse_tar_bytes": tar.stat().st_size,
                "tars": [record],
            }
        )
        path = tmp_path / f"manifest-{index}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        manifests.append((path, digest(path)))
        tars.append(tar)
    return root, manifests, tars


def test_inventory_binds_manifests_receipts_hashes_and_stats(tmp_path: Path) -> None:
    root, manifests, tars = _fixture(tmp_path)
    inventory, evidence = build_inventory(root, manifests, expected_root=root)
    assert [item.path for item in inventory] == tars
    assert evidence["selected_files"] == 2


def test_cycle_advises_readonly_files_only_below_headroom(tmp_path: Path) -> None:
    root, manifests, tars = _fixture(tmp_path)
    inventory, _ = build_inventory(root, manifests, expected_root=root)
    before_hashes = [digest(path) for path in tars]
    calls: list[tuple[int, int, int]] = []
    observations = iter(
        [
            {"effective_headroom_bytes": 2 * GIB},
            {"effective_headroom_bytes": 9 * GIB},
        ]
    )
    result = cycle(
        inventory,
        observe=lambda: next(observations),
        advise=lambda _fd, offset, length, advice: calls.append((offset, length, advice)),
    )
    assert result["status"] == "ADVISED" and result["advised_files"] == 2
    assert calls == [(0, 0, int(getattr(os, "POSIX_FADV_DONTNEED", 4)))] * 2
    assert [digest(path) for path in tars] == before_hashes


def test_cycle_is_idle_above_headroom(tmp_path: Path) -> None:
    root, manifests, _ = _fixture(tmp_path)
    inventory, _ = build_inventory(root, manifests, expected_root=root)
    calls: list[object] = []
    result = cycle(
        inventory,
        observe=lambda: {"effective_headroom_bytes": 9 * GIB},
        advise=lambda *args: calls.append(args),
    )
    assert result["status"] == "IDLE_HEADROOM_OK" and calls == []


def test_stat_barrier_rejects_changed_tar(tmp_path: Path) -> None:
    root, manifests, tars = _fixture(tmp_path)
    inventory, _ = build_inventory(root, manifests, expected_root=root)
    tars[0].chmod(0o666)
    with pytest.raises(ValueError, match="changed before"):
        advise_file(inventory[0], advise=lambda *_args: None)
