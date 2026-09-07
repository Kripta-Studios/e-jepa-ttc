"""Late DENSE binding is prerequisite checked; workers here are no-fit stubs."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/execute_simplex_t_bound_dense_replay.py"
    spec = importlib.util.spec_from_file_location("bound_dense_fixture", script)
    assert spec and spec.loader
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    builder = scripts / "prepare_simplex_t_dense_launch.py"
    worker = scripts / "execute_simplex_t_expanded_replay.py"
    for path in (builder, worker):
        path.write_bytes(b"# fixture only")
    monkeypatch.setattr(entry, "__file__", str(scripts / script.name))
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"worktree": str(tmp_path)}))
    launch = tmp_path / "artifacts/launch.json"
    argv = [
        str(script),
        "--local-paths",
        str(local),
        "--launch",
        str(launch),
        "--builder-sha256",
        hashlib.sha256(builder.read_bytes()).hexdigest(),
        "--worker-sha256",
        hashlib.sha256(worker.read_bytes()).hexdigest(),
        "--other-reserved-bytes",
        "0",
        "--max-new-queries",
        "42943",
    ]
    monkeypatch.setattr(entry.sys, "argv", argv)
    return entry, argv, launch


@pytest.mark.parametrize("code", [3, 10, 1])
def test_builder_not_ready_never_dispatches_replay(setup, monkeypatch, code):
    entry, argv, launch = setup
    argv.append("--verify-only")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert Path(command[2]).name == "prepare_simplex_t_dense_launch.py"
        assert "--verify-only" in command
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(entry.subprocess, "run", run)
    assert entry.main() == code
    assert len(calls) == 1 and not launch.parent.exists()


@pytest.mark.parametrize("verify", [False, True])
def test_validated_launch_hash_passed_to_worker(setup, monkeypatch, verify):
    entry, argv, launch = setup
    report = launch.with_name("report.json")
    argv.extend(["--verify-only"] if verify else ["--report", str(report)])
    if verify:
        launch.parent.mkdir()
        launch.write_bytes(b"validated fixture")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if Path(command[2]).name == "prepare_simplex_t_dense_launch.py":
            if not verify:
                launch.parent.mkdir()
                launch.write_bytes(b"validated fixture")
        else:
            assert (
                command[command.index("--config-sha256") + 1]
                == hashlib.sha256(launch.read_bytes()).hexdigest()
            )
            assert ("--verify-only" in command) == verify
            assert ("--report" in command) != verify
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", run)
    assert entry.main() == 0
    assert len(calls) == 2 and not report.exists()


def test_compile_binds_correct_reuse_and_retains_dense_reservation(setup, monkeypatch):
    entry, argv, launch = setup
    scripts = Path(entry.__file__).parent
    for name in ("execute_simplex_t_compilation_attempt.py", "run_simplex_t_companion.py"):
        (scripts / name).write_bytes(b"compile fixture")
    digest = hashlib.sha256(b"compile fixture").hexdigest()
    argv.extend(
        [
            "--compile-outer",
            "1",
            "--compilation-controller-sha256",
            digest,
            "--compilation-worker-sha256",
            digest,
            "--verify-only",
        ]
    )
    launch.parent.mkdir()
    launch.write_text(
        json.dumps(
            {
                "reserved_output_bytes": 8443789312,
                "d0_reuse": {
                    "1": {
                        "compiled": "artifacts/simplex_t/T1/compiled_context/outer1",
                        "compiled_sha256": "a" * 64,
                    }
                },
            }
        )
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            assert Path(command[2]).name == "execute_simplex_t_compilation_attempt.py"
            assert command[command.index("--reuse-d0-compiled-sha256") + 1] == "a" * 64
            assert command[command.index("--other-reserved-bytes") + 1] == "8443789312"
            assert command[command.index("--outer") + 1] == "1"
            assert "--verify-only" in command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(entry.subprocess, "run", run)
    assert entry.main() == 0 and len(calls) == 2
