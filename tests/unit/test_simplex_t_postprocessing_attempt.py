"""Postprocessing retry routing; scientific worker is stubbed, no optimizer runs."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def entry(tmp_path, monkeypatch):
    script = (
        Path(__file__).resolve().parents[2] / "scripts/execute_simplex_t_postprocessing_attempt.py"
    )
    spec = importlib.util.spec_from_file_location("post_attempt_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    worker = scripts / "postprocess_simplex_t_campaign.py"
    worker.write_bytes(b"# worker fixture only")
    monkeypatch.setattr(module, "__file__", str(scripts / script.name))
    root = tmp_path / "artifacts/attempts"
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "schema": "simplex_t_postprocessing_launch_v3",
                "roots": {"work": str(tmp_path)},
                "output": str(root / "analysis"),
                "delivery": {"output": str(root / "delivery"), "analysis_commit": "a" * 40},
                "freeze_sha256": "b" * 64,
            }
        )
    )
    argv = [
        str(script),
        "--template",
        str(template),
        "--template-sha256",
        hashlib.sha256(template.read_bytes()).hexdigest(),
        "--worker-sha256",
        hashlib.sha256(worker.read_bytes()).hexdigest(),
        "--attempt-root",
        str(root),
        "--other-reserved-bytes",
        "0",
        "--own-reserved-bytes",
        "100000000",
    ]
    monkeypatch.setattr(module.sys, "argv", argv)
    monkeypatch.setattr(
        module,
        "admitted",
        lambda *a: {
            "has_headroom": True,
            "written_volume_free_bytes": [100_000_000_000],
        },
    )
    return module, root, argv, template, worker


def test_resource_pause_preserves_attempt_and_resume_only_relocates_outputs(entry, monkeypatch):
    module, root, argv, template, _ = entry
    calls = []

    def run(command, **kwargs):
        launch = Path(command[command.index("--launch") + 1])
        config = json.loads(launch.read_text(encoding="utf-8"))
        assert config == module.relocated(json.loads(template.read_text()), launch.parent)
        calls.append(command)
        if "--verify-only" in command:
            return SimpleNamespace(
                returncode=0 if (launch.parent / "fixture_complete").exists() else 10
            )
        if len(calls) == 1:
            (launch.parent / "fixture_partial").write_bytes(b"preserve me")
            return SimpleNamespace(returncode=3)
        (launch.parent / "fixture_complete").write_bytes(b"not scientific evidence")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main() == 3
    first = next(root.glob("attempt_*"))
    preserved = {p.name: p.read_bytes() for p in first.iterdir()}
    argv.append("--resume")
    assert module.main() == 0
    assert len(list(root.glob("attempt_*"))) == 2
    assert {p.name: p.read_bytes() for p in first.iterdir()} == preserved
    argv.append("--verify-only")
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }
    assert module.main() == 0
    assert {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    } == before


@pytest.mark.parametrize("fault", ["resource", "worker", "template", "missing"])
def test_attempt_rejection_and_incomplete_verification_do_not_create_outputs(
    entry, monkeypatch, fault
):
    module, root, argv, template, worker = entry
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("worker started"))
    if fault == "resource":
        monkeypatch.setattr(
            module,
            "admitted",
            lambda *a: {
                "has_headroom": False,
                "written_volume_free_bytes": [100_000_000_000],
            },
        )
    elif fault == "worker":
        worker.write_bytes(b"changed")
    elif fault == "template":
        template.write_bytes(b"{}")
    else:
        argv.append("--verify-only")
    if fault in {"worker", "template"}:
        with pytest.raises(ValueError, match="SHA256"):
            module.main()
    else:
        assert module.main() == (3 if fault == "resource" else 10)
    assert not root.exists()


def test_changed_scientific_field_in_attempt_is_not_resumable(entry, monkeypatch):
    module, root, argv, _, _ = entry
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=3))
    assert module.main() == 3
    record_path = next(root.glob("attempt_*/ATTEMPT.json"))
    launch = record_path.parent / "launch.json"
    config = json.loads(launch.read_text(encoding="utf-8"))
    config["freeze_sha256"] = "c" * 64
    launch.write_text(json.dumps(config))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["launch_sha256"] = hashlib.sha256(launch.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record))
    argv.append("--resume")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("worker started"))
    with pytest.raises(ValueError, match="lineage"):
        module.main()
