from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.runpod_event_cache_controller_v1 import (
    GIB,
    advise_cycle,
    build_inventory,
    cgroup_observation,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_stage(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    root = tmp_path / "stage"
    producer = root / "drives/E/EJEPA_TRAIN40_CACHE_20261005/producer_inputs/a.npz"
    teacher = root / (
        "workspace/e-jepa-ttc-v12-efficient-context/artifacts/train40_system_20261005/teacher/b.npz"
    )
    excluded = root / "workspace/other.npz"
    for path, payload in ((producer, b"producer"), (teacher, b"teacher"), (excluded, b"other")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o444)
    rows = [
        {
            "relative": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in (producer, teacher, excluded)
    ]
    manifest = {"profile": "event_producers", "files": rows}
    manifest_path = root / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sealed = []
    for path in (producer, teacher, excluded):
        stat = path.stat()
        sealed.append(
            {
                "relative": path.relative_to(root).as_posix(),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino,
                "device": stat.st_dev,
                "mode": stat.st_mode,
            }
        )
    seal = {
        "schema": "ejepa_immutable_local_bundle_seal_v1",
        "status": "SEALED",
        "root": str(root.resolve()),
        "manifest_sha256": _sha(manifest_path),
        "files": sealed,
    }
    seal_path = root.parent / f"{root.name}.SEAL.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    return root, seal_path, [producer, teacher]


def test_inventory_is_manifest_bound_and_prefix_scoped(tmp_path: Path) -> None:
    root, seal, expected = _sealed_stage(tmp_path)
    inventory, provenance = build_inventory(root, seal, expected_root=root)
    assert [item.path for item in inventory] == expected
    assert provenance["selected_files"] == 2
    assert provenance["selected_bytes"] == sum(path.stat().st_size for path in expected)


def test_inventory_rejects_symlink_even_when_seal_claims_it(tmp_path: Path) -> None:
    root, seal_path, paths = _sealed_stage(tmp_path)
    target = paths[0]
    replacement = tmp_path / "replacement.npz"
    replacement.write_bytes(target.read_bytes())
    target.chmod(0o666)
    target.unlink()
    try:
        target.symlink_to(replacement)
    except OSError as error:
        pytest.skip(f"Symlink privilege unavailable: {error}")
    with pytest.raises(ValueError, match="changed"):
        build_inventory(root, seal_path, expected_root=root)


def test_cycle_is_bounded_records_full_ranges_and_does_not_mutate(tmp_path: Path) -> None:
    root, seal, paths = _sealed_stage(tmp_path)
    inventory, _ = build_inventory(root, seal, expected_root=root)
    hashes = [_sha(path) for path in paths]
    calls: list[tuple[int, int, int]] = []
    observations = iter(
        [
            {"effective_headroom_bytes": 3 * GIB},
            {"effective_headroom_bytes": 9 * GIB},
            {"effective_headroom_bytes": 9 * GIB},
        ]
    )

    def advise(_fd: int, offset: int, length: int, advice: int) -> None:
        calls.append((offset, length, advice))

    cursor, report = advise_cycle(
        inventory,
        0,
        target_headroom_bytes=8 * GIB,
        max_bytes=1024,
        max_files=2,
        observe=lambda: next(observations),
        advise=advise,
    )
    assert cursor == 1
    assert report["advised_files"] == 1
    assert report["ranges"] == [
        {
            "relative": inventory[0].relative,
            "offset": 0,
            "length": 0,
            "range_bytes": inventory[0].bytes,
        }
    ]
    assert calls == [(0, 0, int(getattr(os, "POSIX_FADV_DONTNEED", 4)))]
    assert [_sha(path) for path in paths] == hashes


def test_cycle_does_nothing_when_real_headroom_is_already_sufficient(tmp_path: Path) -> None:
    root, seal, _ = _sealed_stage(tmp_path)
    inventory, _ = build_inventory(root, seal, expected_root=root)
    calls: list[object] = []
    observation = {"effective_headroom_bytes": 9 * GIB}
    cursor, report = advise_cycle(
        inventory,
        0,
        observe=lambda: observation,
        advise=lambda *args: calls.append(args),
    )
    assert cursor == 0
    assert report["status"] == "IDLE_HEADROOM_OK"
    assert report["advised_bytes"] == 0
    assert calls == []


def test_cgroup_observation_reports_anon_and_file_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    membership = tmp_path / "cgroup"
    membership.write_text("0::/pod\n", encoding="utf-8")

    def snapshot(*_args: object) -> tuple[int, list[dict[str, object]]]:
        return 7 * GIB, [
            {
                "current": 41 * GIB,
                "memory_stat": {"anon": 32 * GIB, "file": 9 * GIB},
                "version": 2,
                "path": "/sys/fs/cgroup/pod",
                "kernel_limit": 61 * GIB,
                "effective_limit": 61 * GIB,
            }
        ]

    monkeypatch.setattr("scripts.runpod_event_cache_controller_v1.memory_snapshot", snapshot)
    observed = cgroup_observation(cgroup_root=tmp_path / "unused", membership_path=membership)
    assert observed["effective_headroom_bytes"] == 7 * GIB
    assert observed["anon_or_rss_bytes"] == 32 * GIB
    assert observed["file_cache_bytes"] == 9 * GIB


def test_advice_error_propagates(tmp_path: Path) -> None:
    root, seal, _ = _sealed_stage(tmp_path)
    inventory, _ = build_inventory(root, seal, expected_root=root)

    def fail(*_args: object) -> None:
        raise OSError("kernel advice failed")

    with pytest.raises(OSError, match="kernel advice failed"):
        advise_cycle(
            inventory,
            0,
            observe=lambda: {"effective_headroom_bytes": 3 * GIB},
            advise=fail,
        )
