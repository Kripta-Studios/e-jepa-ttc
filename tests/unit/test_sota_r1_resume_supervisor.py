"""CPU-only admission, ownership, and durable recovery tests for R1 control."""

import base64
import json

import pytest

from operational.sota_eval import r1_resume_supervisor as supervisor


class Owner:
    pid = 123

    def __init__(self, *, children=(), created=4.5, module=None):
        self.child_list = children
        self.created = created
        self.module = module or "operational.efficient_context.r1_gib_supervisor"
        self.terminated = False
        self.waited = False

    def create_time(self):
        return self.created

    def name(self):
        return "python.exe"

    def cmdline(self):
        return ["python.exe", "-m", self.module]

    def children(self, recursive=True):
        return self.child_list

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        assert timeout == 15
        self.waited = True


@pytest.mark.parametrize(
    "module",
    [
        "operational.sota_eval.fcwd_run",
        "operational.sota_eval.cost",
        "operational.sota_eval.prefetch",
        "operational.sota_eval.full_prefetch",
        "operational.sota_eval.campaign",
        "operational.evttc_transfer.run",
        "operational.evttc_rgb_transfer.run",
        "operational.train40_system.engine",
        "operational.train40_system.engine_profiled",
        "operational.train40_system.garl_predictions",
        "operational.train40_system.history_features",
        "operational.train40_system.history_resources8_transport_graph",
        "operational.efficient_context.r1_gib_measure",
    ],
)
def test_actual_gpu_work_and_prefetch_block(module):
    assert supervisor.is_heavy("python.exe", ["python", "-m", module])
    assert not supervisor.is_heavy("powershell.exe", ["powershell", "-Command", module])


@pytest.mark.parametrize(
    ("module", "arguments"),
    [
        ("operational.sota_eval.publication", ["--require-complete"]),
        ("operational.sota_eval.scoring", []),
        ("operational.sota_eval.followups", ["--verify-only"]),
        ("pytest", ["tests/unit"]),
        ("operational.sota_eval.fcwd_run", ["--score"]),
        ("operational.sota_eval.campaign", ["--device", "cpu"]),
    ],
)
def test_cpu_only_entrypoints_do_not_block_r1(module, arguments):
    assert not supervisor.is_heavy("python.exe", ["python", "-m", module, *arguments])


def test_stage_protection_and_control_exclusion():
    assert supervisor.is_heavy("python.exe", ["python", "-m", "runner", "--stage75"])
    assert not supervisor.is_heavy(
        "python.exe", ["python", "-m", "operational.sota_eval.publication", "--note=stage75"]
    )
    assert not supervisor.is_heavy("python.exe", Owner().cmdline())


@pytest.mark.parametrize(
    "owner", [Owner(created=8), Owner(children=[object()]), Owner(module="other.training")]
)
def test_retire_rejects_wrong_owner_or_active_children(owner):
    with pytest.raises((ValueError, RuntimeError)):
        supervisor.retire_owner(owner, dict(pid=123, create_time=4.5))
    assert not owner.terminated


def test_retire_only_idle_exact_owner():
    owner = Owner()
    supervisor.retire_owner(owner, dict(pid=123, create_time=4.5))
    assert owner.terminated and owner.waited


def fixture_handoff(tmp_path):
    output, artifacts = tmp_path / "output", tmp_path / "control"
    output.mkdir()
    artifacts.mkdir()
    marker = output / "STOP_REQUEST"
    marker.write_bytes(b"root-owned pause")
    identity = dict(pid=123, create_time=4.5)
    supervisor.atomic_json(output / "WRITER.lock", identity)
    supervisor.atomic_json(output / "CHAIN_STATE.json", dict(status="WAITING", stop_requested=True))
    fragment = output / "measurement/fragments/one.json"
    fragment.parent.mkdir(parents=True)
    fragment.write_bytes(b"immutable paired evidence")
    receipt = artifacts / "root-receipt.json"
    supervisor.atomic_json(
        receipt,
        dict(
            path=str(marker),
            sha256=supervisor.digest(marker),
            owner="sota_campaign_20261008",
            preserve_pairs=1,
        ),
    )
    config = dict(
        handoff=True,
        old_owner=identity,
        ownership_receipt=str(receipt),
        ownership_receipt_sha256=supervisor.digest(receipt),
    )
    return output, artifacts, config, fragment


def test_handoff_exact_owner_preserves_all_receipts(tmp_path, monkeypatch):
    import psutil

    output, artifacts, config, fragment = fixture_handoff(tmp_path)
    old_sha = supervisor.digest(fragment)
    owner = Owner()
    monkeypatch.setattr(supervisor, "same_process", lambda identity: True)
    monkeypatch.setattr(psutil, "Process", lambda pid: owner)
    supervisor.handoff(output, artifacts, config)
    assert owner.terminated and owner.waited
    assert not (output / "STOP_REQUEST").exists()
    record = supervisor.read(artifacts / "HANDOFF.json")
    assert record["status"] == "COMPLETE"
    assert base64.b64decode(record["stop_bytes_b64"]) == b"root-owned pause"
    assert supervisor.digest(fragment) == old_sha
    assert (output / "WRITER.lock").exists()  # Child Lease preserves stale owner itself.


def test_changed_stop_refuses_handoff(tmp_path, monkeypatch):
    output, artifacts, config, _ = fixture_handoff(tmp_path)
    (output / "STOP_REQUEST").write_bytes(b"another owner")
    monkeypatch.setattr(supervisor, "retire_owner", lambda *args: pytest.fail("must not retire"))
    with pytest.raises(ValueError, match="marker changed"):
        supervisor.handoff(output, artifacts, config)
    assert (output / "STOP_REQUEST").read_bytes() == b"another owner"


def test_handoff_crash_after_removal_recovers_without_another_mutation(tmp_path, monkeypatch):
    output, artifacts, config, _ = fixture_handoff(tmp_path)
    record = dict(
        status="OLD_OWNER_RETIRED",
        expected_owner=config["old_owner"],
        receipt_snapshot=supervisor.r1_resume.receipt_snapshot(output),
    )
    supervisor.atomic_json(artifacts / "HANDOFF.json", record)
    (output / "STOP_REQUEST").unlink()
    monkeypatch.setattr(supervisor, "same_process", lambda identity: False)
    monkeypatch.setattr(supervisor, "retire_owner", lambda *args: pytest.fail("already retired"))
    supervisor.handoff(output, artifacts, config)
    assert supervisor.read(artifacts / "HANDOFF.json")["status"] == "COMPLETE"


def test_new_stop_does_not_replace_unknown_marker(tmp_path):
    output, artifacts, _, _ = fixture_handoff(tmp_path)
    supervisor.stop_child(output, artifacts, "EXTERNAL_HEAVY")
    assert (output / "STOP_REQUEST").read_bytes() == b"root-owned pause"
    assert not (artifacts / "OWN_STOP.json").exists()


def test_only_owned_temporary_stop_is_released(tmp_path, monkeypatch):
    # Exercise recovery before its deadline regardless of the machine's date.
    monkeypatch.setattr(
        supervisor, "DEADLINE", supervisor.datetime.max.replace(tzinfo=supervisor.UTC)
    )
    output, artifacts, _, _ = fixture_handoff(tmp_path)
    (output / "STOP_REQUEST").unlink()
    supervisor.stop_child(output, artifacts, "EXTERNAL_HEAVY")
    supervisor.clear_recovered_own_stop(output, artifacts, [dict(pid=999)])
    assert (output / "STOP_REQUEST").exists()
    supervisor.clear_recovered_own_stop(output, artifacts, [])
    assert not (output / "STOP_REQUEST").exists()
    supervisor.stop_child(output, artifacts, "DEADLINE")
    supervisor.clear_recovered_own_stop(output, artifacts, [])
    assert (output / "STOP_REQUEST").exists()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("InterruptedError: R1 needs RAM\nValueError: invalid receipt", "INTEGRITY_PERMANENT"),
        ("InterruptedError: R1 needs RAM", "RESOURCE_TRANSIENT"),
        ("InterruptedError: stop requested", "PRESERVED_STOP"),
        ("OSError: disconnected disk", "RESOURCE_TRANSIENT"),
        ("RuntimeError: checksum mismatch", "INTEGRITY_PERMANENT"),
        ("", "PERMANENT_FAILURE"),
    ],
)
def test_only_attempt_terminal_exception_controls_retry(text, expected):
    assert supervisor.classify_failure(text) == expected


def test_orphan_attempt_reconciliation_preserves_saved_work(tmp_path, monkeypatch):
    output, artifacts, _, _ = fixture_handoff(tmp_path)
    log = artifacts / "attempt.log"
    log.write_text("InterruptedError: R1 needs RAM\n", encoding="utf-8")
    path = artifacts / "attempt.json"
    supervisor.atomic_json(
        path,
        dict(
            status="RUNNING",
            child=dict(pid=1, create_time=2),
            log=str(log),
            before_snapshot=supervisor.r1_resume.receipt_snapshot(output),
        ),
    )
    monkeypatch.setattr(supervisor, "same_process", lambda identity: False)
    monkeypatch.setattr(supervisor.time, "time", lambda: 100)
    record = supervisor.reconcile_attempt(output, path, 1800)
    assert record["failure_kind"] == "RESOURCE_TRANSIENT"
    assert record["retry_after_epoch"] == 1900
    assert supervisor.reconcile_attempt(output, path, 1800) == record


def test_source_change_stops_before_resume(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("original", encoding="utf-8")
    supervisor.atomic_json(tmp_path / "SUPERVISOR_GIB_FREEZE.json", dict(sources={}))
    config = dict(
        parent_freeze_sha256=supervisor.digest(tmp_path / "SUPERVISOR_GIB_FREEZE.json"),
        sources={"source.py": supervisor.digest(source)},
    )
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    supervisor.verify_sources(tmp_path, config)
    source.write_text("edited", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        supervisor.verify_sources(tmp_path, config)


def test_default_cli_plan_never_transfers_or_launches(tmp_path, monkeypatch):
    monkeypatch.setattr(
        supervisor.sys, "argv", ["r1_resume_supervisor", "--artifacts", str(tmp_path)]
    )
    monkeypatch.setattr(supervisor, "configuration", lambda args: dict(mode="plan"))
    monkeypatch.setattr(supervisor, "run", lambda *args: pytest.fail("planning must not execute"))
    assert supervisor.main() == 0
    assert json.loads((tmp_path / "PLAN.json").read_text())["mode"] == "plan"


def test_finalization_calls_original_bundle_and_verifies_all_members(tmp_path, monkeypatch):
    output, artifacts, _, _ = fixture_handoff(tmp_path)
    supervisor.atomic_json(
        output / "measurement/RESULT.json",
        dict(status="COMPLETE", pairs=768, gpu_measured=True, summaries=[]),
    )
    monkeypatch.setattr(supervisor, "verify_sources", lambda *args: None)
    monkeypatch.setattr(
        supervisor.r1_resume,
        "plan",
        lambda output: dict(
            completed_pairs=768, receipt_snapshot=supervisor.r1_resume.receipt_snapshot(output)
        ),
    )
    verification = supervisor.finalize(output, artifacts, dict(os_cache_history="interrupted"))
    assert verification["status"] == "PASSED"
    assert supervisor.digest(output / "R1_ESSENTIAL.zip") == verification["sha256"]
    assert supervisor.read(artifacts / "FINAL_VERIFICATION.json")["receipts"] == 768
    assert (
        supervisor.read(output / "RESUME_RECOVERY_ADMISSION.json")["os_cache_history"]
        == "interrupted"
    )
