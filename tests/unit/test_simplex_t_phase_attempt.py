"""Attempt metadata must not alter scientific bindings or overwrite receipts."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from e_jepa_ttc.simplex_t.resource_bundle import ResourceAttempt, resource_bundle_members


@pytest.fixture
def entry():
    path = Path(__file__).resolve().parents[2] / "scripts/execute_simplex_t_phase_attempt.py"
    spec = importlib.util.spec_from_file_location("phase_attempt_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attempts_only_change_receipt_and_retain_template(entry, tmp_path):
    original = {
        "schema": "simplex_t_frozen_phase_launch_v2",
        "roots": {"work": str(tmp_path)},
        "resource_receipt": "original.json",
        "freeze_sha256": "a" * 64,
        "execution": "same_execution",
        "publication": "same_publication",
        "availability": {"D1": True},
    }
    source = tmp_path / "template.json"
    payload = json.dumps(original).encode()
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    paths = []
    receipts = []
    for _ in range(2):
        launch, pin = entry.materialize_attempt(source, digest, tmp_path)
        assert hashlib.sha256(launch.read_bytes()).hexdigest() == pin
        actual = json.loads(launch.read_bytes())
        receipt = Path(actual.pop("resource_receipt"))
        expected = {k: v for k, v in original.items() if k != "resource_receipt"}
        assert actual == expected
        assert receipt.parent == launch.parent
        assert not receipt.exists()
        paths.append(launch)
        receipts.append(receipt)
    assert len(set(paths)) == len(set(receipts)) == 2
    assert source.read_bytes() == payload


@pytest.mark.parametrize("fault", ["hash", "work", "schema"])
def test_invalid_template_creates_no_attempt(entry, tmp_path, fault):
    source = tmp_path / "template.json"
    other = tmp_path / "other"
    other.mkdir()
    source.write_text(
        json.dumps(
            {
                "schema": "wrong" if fault == "schema" else "simplex_t_frozen_phase_launch_v2",
                "roots": {"work": str(other if fault == "work" else tmp_path)},
                "resource_receipt": "old.json",
            }
        )
    )
    digest = "0" * 64 if fault == "hash" else hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        entry.materialize_attempt(source, digest, tmp_path)
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("allowed", [False, True])
def test_cli_admission_and_resume_dispatch(entry, tmp_path, monkeypatch, allowed):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(entry, "__file__", str(scripts / "execute_simplex_t_phase_attempt.py"))
    worker = scripts / "execute_simplex_t_frozen_phase.py"
    worker.write_text("# never executed by this fixture\n")
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "schema": "simplex_t_frozen_phase_launch_v2",
                "roots": {"work": str(tmp_path)},
                "resource_receipt": "old.json",
            }
        )
    )
    report = tmp_path / "artifacts/attempt.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--template",
            str(template),
            "--template-sha256",
            hashlib.sha256(template.read_bytes()).hexdigest(),
            "--worker-sha256",
            hashlib.sha256(worker.read_bytes()).hexdigest(),
            "--report",
            str(report),
            "--other-reserved-bytes",
            "0",
            "--own-reserved-bytes",
            "2097152",
            "--resume",
        ],
    )
    monkeypatch.setattr(
        entry,
        "admitted",
        lambda _: {
            "has_headroom": allowed,
            "written_volume_free_bytes": [90_000_000_000],
        },
    )
    calls = []

    def run(command, *, check):
        assert check is False
        assert report.exists()  # lineage is published before worker execution
        assert command[-1] == "--resume"
        assert json.loads(report.read_text(encoding="utf-8"))["command"] == command
        calls.append(command)
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(entry.subprocess, "run", run)
    assert entry.main() == 3
    assert len(calls) == int(allowed)
    assert report.exists() == allowed
    if not allowed:
        assert not (tmp_path / "artifacts").exists()


def test_materialized_launch_real_cli_receipt_and_transport(entry, tmp_path, monkeypatch):
    """Real CLI receipt writer and transport, with model execution explicitly stubbed."""
    runner_path = Path(__file__).resolve().parents[2] / "scripts/execute_simplex_t_frozen_phase.py"
    spec = importlib.util.spec_from_file_location("phase_receipt_integration", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(runner.torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(
        runner,
        "admitted",
        lambda _: {
            "has_headroom": True,
            "process_tree_rss_bytes": 100_000_000,
            "host_available_bytes": 10_000_000_000,
            "written_volume_free_bytes": [90_000_000_000],
        },
    )
    template = tmp_path / "template.json"
    original = dict(
        schema="simplex_t_frozen_phase_launch_v2",
        roots={"work": str(tmp_path)},
        resource_receipt="unused.json",
        local_paths="local.json",
        source_configuration="sources.json",
        source_configuration_sha256="b" * 64,
        evidence_profile="evidence.json",
        evidence_profile_sha256="c" * 64,
        freeze="freeze.json",
        freeze_sha256="a" * 64,
        execution="same_execution",
        publication="same_publication",
        stage="T2",
        availability={},
        publications={},
    )
    template.write_text(json.dumps(original), encoding="utf-8")
    template_hash = hashlib.sha256(template.read_bytes()).hexdigest()
    attempts = []
    for index, status in enumerate(("PAUSED_RESOURCE", "PUBLISHED")):
        launch, digest = entry.materialize_attempt(template, template_hash, tmp_path)

        def execute(expected_resume=bool(index), expected_status=status, **kwargs):
            assert kwargs["resource_ok"]()
            assert kwargs["resume"] == expected_resume
            assert kwargs["execution"] == Path("same_execution")
            return {"status": expected_status}

        monkeypatch.setattr(runner, "execute_configured_phase", execute)
        arguments = [
            "runner",
            "--launch",
            str(launch),
            "--launch-sha256",
            digest,
            "--other-reserved-bytes",
            "0",
            "--own-reserved-bytes",
            "2097152",
        ]
        if index:
            arguments.append("--resume")
        monkeypatch.setattr("sys.argv", arguments)
        assert runner.main() == (3 if index == 0 else 0)
        receipt = Path(json.loads(launch.read_bytes())["resource_receipt"])
        attempts.append(
            ResourceAttempt(
                launch,
                digest,
                receipt,
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
            )
        )
    members = resource_bundle_members(
        attempts,
        work_root=tmp_path,
        freeze_sha256="a" * 64,
        completed_stages={"T2"},
        resource_ok=lambda: True,
    )
    assert len(members) == 4
    assert hashlib.sha256(template.read_bytes()).hexdigest() == template_hash
