"""Source identity dispatch uses real CLI contracts with no-fit stub workers."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mode", ["new", "resume", "verify", "missing"])
def test_source_preparation_dispatch(tmp_path, monkeypatch, mode):
    script = Path(__file__).resolve().parents[2] / "scripts/execute_simplex_t_source_preparation.py"
    spec = importlib.util.spec_from_file_location("source_dispatch_fixture", script)
    assert spec and spec.loader
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("prepare_simplex_t_source_configuration.py", "run_simplex_t_companion.py"):
        (scripts / name).write_bytes(b"fixture")
    monkeypatch.setattr(entry, "__file__", str(scripts / script.name))
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"worktree": str(tmp_path)}))
    config = tmp_path / "artifacts/source.json"
    config.parent.mkdir()
    if mode != "missing":
        config.write_bytes(b"validated fixture")
    output = tmp_path / "artifacts/preparation"
    if mode == "resume":
        output.mkdir()
        (output / "SOURCE_PREPARATION.json").write_bytes(b"retained")
    digest = hashlib.sha256(b"fixture").hexdigest()
    argv = [
        str(script),
        "--local-paths",
        str(local),
        "--source-config",
        str(config),
        "--output",
        str(output),
        "--builder-sha256",
        digest,
        "--worker-sha256",
        digest,
        "--other-reserved-bytes",
        "0",
    ]
    if mode == "verify":
        argv.append("--verify-only")
    monkeypatch.setattr(entry.sys, "argv", argv)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            assert "--verify-only" in command
            return SimpleNamespace(returncode=10 if mode == "missing" else 0)
        assert "--source-identities" in command
        assert ("--resume" in command) == (mode == "resume")
        assert ("--verify-only" in command) == (mode == "verify")
        assert (
            command[command.index("--source-config-sha256") + 1]
            == hashlib.sha256(config.read_bytes()).hexdigest()
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", run)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert entry.main() == (10 if mode == "missing" else 0)
    assert len(calls) == (1 if mode == "missing" else 2)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
