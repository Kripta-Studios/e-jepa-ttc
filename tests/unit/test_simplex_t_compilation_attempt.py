"""Attempt publication lifecycle; stub workers never run inference or optimizers."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    script = (
        Path(__file__).resolve().parents[2] / "scripts/execute_simplex_t_compilation_attempt.py"
    )
    spec = importlib.util.spec_from_file_location("compilation_attempt_fixture", script)
    assert spec and spec.loader
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    worker = scripts / "run_simplex_t_companion.py"
    worker.write_bytes(b"# fixture only")
    monkeypatch.setattr(entry, "__file__", str(scripts / script.name))
    paths = tmp_path / "local.json"
    paths.write_text(json.dumps({"worktree": str(tmp_path)}))
    output = tmp_path / "artifacts/outer0"
    argv = [
        str(script),
        "--local-paths",
        str(paths),
        "--worker-sha256",
        hashlib.sha256(worker.read_bytes()).hexdigest(),
        "--output",
        str(output),
        "--outer",
        "0",
        "--pool",
        "D0",
        "--other-reserved-bytes",
        "0",
    ]
    monkeypatch.setattr(entry.sys, "argv", argv)
    monkeypatch.setattr(
        entry,
        "admitted",
        lambda *a: {"has_headroom": True, "written_volume_free_bytes": [100_000_000_000]},
    )
    return entry, output, argv


def test_pause_retained_then_fresh_attempt_published(setup, monkeypatch):
    entry, output, argv = setup
    attempts = []

    def worker(command, **kwargs):
        destination = Path(command[command.index("--output") + 1])
        if "--verify-only" in command:
            assert (destination / "COMPLETE").exists()
            return SimpleNamespace(returncode=0)
        destination.mkdir()
        attempts.append(destination)
        if len(attempts) == 1:
            (destination / "partial").write_bytes(b"retained")
            return SimpleNamespace(returncode=3)
        (destination / "COMPLETE").write_bytes(b"complete")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", worker)
    assert entry.main() == 3
    assert not output.exists()
    partial = attempts[0] / "partial"
    stamp = partial.stat().st_mtime_ns
    argv.append("--resume")
    assert entry.main() == 0
    assert (output / "COMPLETE").read_bytes() == b"complete"
    assert partial.read_bytes() == b"retained" and partial.stat().st_mtime_ns == stamp
    assert entry.main() == 0
    assert len(attempts) == 2
    assert not output.with_name("outer0.compile.lock").exists()


def test_failed_verification_cannot_publish(setup, monkeypatch):
    entry, output, _ = setup

    def worker(command, **kwargs):
        destination = Path(command[command.index("--output") + 1])
        if "--verify-only" in command:
            return SimpleNamespace(returncode=1)
        destination.mkdir()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", worker)
    assert entry.main() == 1
    assert not output.exists()
    assert len(list(output.parent.glob("outer0.attempt_*"))) == 1


def test_verify_does_not_create_lock_or_output(setup, monkeypatch):
    entry, output, argv = setup
    argv.append("--verify-only")
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=10))
    assert entry.main() == 10
    assert not output.parent.exists()


def test_denied_resource_has_no_writes(setup, monkeypatch):
    entry, output, _ = setup
    monkeypatch.setattr(entry, "admitted", lambda *a: {"has_headroom": False})
    assert entry.main() == 3
    assert not output.parent.exists()
