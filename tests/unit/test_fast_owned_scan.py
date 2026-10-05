"""Exact byte accounting and unchanged resource/science ceilings; no model or optimizer."""

from types import SimpleNamespace

import pytest

from operational.efficient_context import budget
from operational.efficient_context.fast_owned_scan import install, owned_bytes


def test_count_matches_legacy_nested_files_empty_directories_and_hard_links(tmp_path):
    tmp_path = tmp_path / "owned"
    tmp_path.mkdir()
    for relative, size in (("a.bin", 7), ("nested/a.bin", 19), ("nested/deep/b.bin", 83)):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
    (tmp_path / "empty").mkdir()
    (tmp_path / "hardlink.bin").hardlink_to(tmp_path / "a.bin")
    legacy = sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert owned_bytes(tmp_path) == legacy == 116


def test_empty_directory_counts_zero(tmp_path):
    tmp_path = tmp_path / "empty_owned"
    tmp_path.mkdir()
    assert owned_bytes(tmp_path) == 0


def test_directory_symlink_not_followed_and_file_symlink_matches_legacy(tmp_path):
    root, other = tmp_path / "owned", tmp_path / "outside"
    root.mkdir()
    other.mkdir()
    file = other / "fixture.bin"
    file.write_bytes(b"fixture")
    try:
        (root / "dir_link").symlink_to(other, target_is_directory=True)
        (root / "file_link").symlink_to(file)
    except OSError:
        pytest.skip("Windows symlink privilege unavailable; no system setting changed")
    assert owned_bytes(root) == sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) == 7


@pytest.mark.parametrize("count", [0, 10_000_000_000, 10_000_000_001])
def test_original_artifact_cap_is_preserved(tmp_path, monkeypatch, count):
    from operational.efficient_context import fast_owned_scan

    original = budget.require
    monkeypatch.setattr(budget, "require", original)
    monkeypatch.setattr(budget, "_artifact_scans", {})
    monkeypatch.setattr(fast_owned_scan, "owned_bytes", lambda path: count)
    c = SimpleNamespace(out=tmp_path, require_resources=lambda: None)
    install()
    if count > 10_000_000_000:
        with pytest.raises(InterruptedError, match="exceed10GB"):
            budget.require(c)
    else:
        budget.require(c)
    assert budget._artifact_scans[str(tmp_path)][1] == count


def test_every_original_resource_check_still_runs_on_cache_hit(tmp_path, monkeypatch):
    original = budget.require
    monkeypatch.setattr(budget, "require", original)
    monkeypatch.setattr(budget, "_artifact_scans", {})
    calls = []
    c = SimpleNamespace(out=tmp_path, require_resources=lambda: calls.append("resources"))
    install()
    budget.require(c)
    budget.require(c)
    assert calls == ["resources", "resources"]


def test_io_failure_does_not_publish_an_incomplete_count(tmp_path, monkeypatch):
    from operational.efficient_context import fast_owned_scan

    def fail(path):
        raise OSError("injected directory metadata failure")

    monkeypatch.setattr(fast_owned_scan, "owned_bytes", fail)
    monkeypatch.setattr(budget, "require", budget.require)
    monkeypatch.setattr(budget, "_artifact_scans", {})
    install()
    with pytest.raises(OSError, match="metadata failure"):
        budget.require(SimpleNamespace(out=tmp_path))
    assert not budget._artifact_scans


@pytest.mark.parametrize("mismatch", ["time", "cwd", "module"])
def test_handoff_refuses_an_unrelated_writer_before_any_wait_or_model(
    tmp_path, monkeypatch, mismatch
):
    import psutil

    from operational.efficient_context.common import ROOT, atomic_json
    from operational.efficient_context.garl_train_scandir import handoff

    atomic_json(tmp_path / "WRITER.lock", {"pid": 123, "create_time": 1.0})
    process = SimpleNamespace(
        create_time=lambda: 2.0 if mismatch == "time" else 1.0,
        cwd=lambda: str(tmp_path if mismatch == "cwd" else ROOT),
        cmdline=lambda: (
            ["python", "-m", "other_module"]
            if mismatch == "module"
            else ["python", "-m", "operational.efficient_context.garl_train_parallel_authorized"]
        ),
    )
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    with pytest.raises(ValueError, match="identify"):
        handoff(SimpleNamespace(out=tmp_path))
    assert not (tmp_path / "garl/QUOTA_SCAN_HANDOFF_INTENT.json").exists()


def test_quota_handoff_precedes_unchanged_recipe_and_removes_only_own_flag(tmp_path, monkeypatch):
    from operational.efficient_context import garl_train_parallel_authorized, garl_train_scandir

    path = tmp_path / "garl/QUOTA_SCANNER_FREEZE.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    calls = []
    monkeypatch.setattr(
        garl_train_scandir,
        "Campaign",
        lambda protocol: SimpleNamespace(out=tmp_path, freeze=lambda: calls.append("freeze")),
    )
    monkeypatch.setattr(garl_train_scandir, "handoff", lambda c: calls.append("safe_handoff"))
    monkeypatch.setattr(garl_train_scandir.sys, "argv", ["fixture", "--handoff-quota", "--resume"])

    def original_recipe():
        assert garl_train_scandir.sys.argv == ["fixture", "--resume"]
        calls.append("unchanged_parallel_recipe")
        return 0

    monkeypatch.setattr(garl_train_parallel_authorized, "main", original_recipe)
    assert garl_train_scandir.main() == 0
    assert calls == ["freeze", "safe_handoff", "unchanged_parallel_recipe"]
