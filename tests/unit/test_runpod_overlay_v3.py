"""Cgroup v1/v2, cache accounting and MooseFS routing without network/GPU calls."""

import importlib.util
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from scripts.runpod_resources_v3 import GIB, memory_snapshot
from scripts.runpod_technical_overlay_v3 import select_mount, validate_command, validate_run


def counters(path, version, current, maximum, stats="cache 12345\n"):
    path.mkdir(parents=True, exist_ok=True)
    usage = "memory.usage_in_bytes" if version == 1 else "memory.current"
    limit = "memory.limit_in_bytes" if version == 1 else "memory.max"
    (path / usage).write_text(str(current))
    (path / limit).write_text(str(maximum))
    (path / "memory.stat").write_text(stats)


def test_v1_hybrid_namespaced_mount_counts_cache(tmp_path):
    counters(tmp_path / "memory", 1, 25_000_000_000, 60_999_999_488)
    headroom, rows = memory_snapshot(
        tmp_path, "12:memory:/docker/id\n0::/docker/id", 46_000_000_000
    )
    assert headroom == 21_000_000_000
    assert rows[0]["version"] == 1
    assert rows[0]["namespace_root_fallback"]
    assert rows[0]["cache_subtracted_from_usage"] is False
    assert rows[0]["memory_stat"]["cache"] == 12345


def test_v2_respects_ancestor_limit(tmp_path):
    counters(tmp_path, 2, 39_000_000_000, 40_000_000_000)
    counters(tmp_path / "child", 2, 1_000_000_000, "max")
    headroom, _ = memory_snapshot(tmp_path, "0::/child", 46_000_000_000)
    assert headroom == 1_000_000_000


def test_v1_hidden_ancestor_hierarchical_limit(tmp_path):
    counters(tmp_path / "memory", 1, 12, 1000, "hierarchical_memory_limit 20\n")
    headroom, _ = memory_snapshot(tmp_path, "12:memory:/docker/id", 46_000_000_000)
    assert headroom == 8


@pytest.mark.parametrize("membership", ["bad", "5:cpu:/", "0::/../outside", "0::/\n0::/x"])
def test_invalid_membership_rejected(tmp_path, membership):
    with pytest.raises((ValueError, FileNotFoundError)):
        memory_snapshot(tmp_path, membership, 46_000_000_000)


def test_missing_and_corrupt_counters_rejected(tmp_path):
    counters(tmp_path, 2, 1, "max")
    (tmp_path / "memory.stat").unlink()
    with pytest.raises(FileNotFoundError):
        memory_snapshot(tmp_path, "0::/", 46_000_000_000)
    (tmp_path / "memory.stat").write_text("anon -1\n")
    with pytest.raises(ValueError):
        memory_snapshot(tmp_path, "0::/", 46_000_000_000)


def test_memory_budget_must_exceed_reserve(tmp_path):
    with pytest.raises(ValueError):
        memory_snapshot(tmp_path, "0::/", 2 * GIB)


def test_moosefs_identified_by_source_global_fuse_rejected():
    mount = (
        "2517 1955 0:57 /networkvolumes/id /workspace rw - fuse mfs\\043euro-3.runpod.net:9421 rw\n"
    )
    assert select_mount("/workspace/ejepa_technical/r4", mount)["source"].startswith("mfs#")
    for bad in [
        mount.replace("mfs\\043euro-3.runpod.net:9421", "global"),
        mount.replace(" - fuse ", " - fuse.g "),
    ]:
        with pytest.raises(ValueError):
            select_mount("/workspace/ejepa_technical/r4", bad)


def test_scientific_commands_and_bundle_outputs_rejected():
    validate_run("/workspace/ejepa_technical/r4", "/workspace/ejepa_bundle")
    validate_command(["resume-probe"])
    for command in ["producer", "finish-rotation"]:
        with pytest.raises(ValueError):
            validate_command([command])
    with pytest.raises(ValueError):
        validate_run("/workspace/ejepa_bundle/r4", "/workspace/ejepa_bundle")


def test_shared_df_is_bounded_by_user_quota(monkeypatch):
    from scripts import runpod_resources_v3 as resources

    class Root:
        def resolve(self):
            return PurePosixPath("/workspace/ejepa_technical/r4")

    monkeypatch.setattr(resources, "Path", PurePosixPath)
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**50))
    monkeypatch.setattr(
        resources.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="32000000000 /workspace\n"),
    )
    free, report = resources.storage_available(Root())
    assert free == 18_000_000_000
    assert report["user_confirmed_quota_bytes"] == 50_000_000_000


def test_unknown_volume_usage_fails_closed(monkeypatch):
    from scripts import runpod_resources_v3 as resources

    class Root:
        def resolve(self):
            return PurePosixPath("/workspace/ejepa_technical/r4")

    monkeypatch.setattr(resources, "Path", PurePosixPath)
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**50))
    monkeypatch.setattr(
        resources.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="unknown /workspace\n")
    )
    with pytest.raises(ValueError):
        resources.storage_available(Root())


def test_deadline_fail_closed_before_reading_proc(monkeypatch):
    # Import an isolated copy so changing its platform facade cannot affect Path.
    file = Path(__file__).parents[2] / "scripts/runpod_resources_v3.py"
    spec = importlib.util.spec_from_file_location("isolated_resources_v3", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Platform:
        name = "posix"

    monkeypatch.setattr(module, "os", Platform)
    monitor = module.LinuxResourceMonitor()
    monitor.deadline_timestamp = 1
    allowed, report = monitor.guard(Path("unused"), checkpoint_reservation=0, completed_updates=0)
    assert not allowed
    assert report["reason"] == "CLOUD_EXECUTION_DEADLINE"
