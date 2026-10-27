import argparse
import hashlib
import json

import pytest

from operational.final_refit.auto_resume import GIB, ResumeGate, Supervisor, classify_exit


def test_status_publication_retries_sharing_denial_without_truncation(tmp_path, monkeypatch):
    from operational.final_refit import auto_resume as module

    path = tmp_path / "status.json"
    path.write_text('{"old": true}', encoding="utf-8")
    original = module.os.replace
    attempts = []
    sleeps = []

    def replace(source, destination):
        attempts.append(source)
        assert json.loads(path.read_text()) == {"old": True}
        if len(attempts) < 4:
            raise PermissionError("transient sharing denial")
        original(source, destination)

    monkeypatch.setattr(module.os, "replace", replace)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    module.atomic_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"new": True}
    assert sleeps == [0.25] * 3
    assert len(set(attempts)) == 1
    assert not attempts[0].exists()


@pytest.mark.parametrize("error, expected_sleeps", [(PermissionError, 40), (OSError, 0)])
def test_status_publication_fails_closed_after_bounded_retry(
    tmp_path, monkeypatch, error, expected_sleeps
):
    from operational.final_refit import auto_resume as module

    path = tmp_path / "status.json"
    path.write_text('{"old": true}', encoding="utf-8")
    sleeps = []

    def replace(source, destination):
        raise error("persistent failure")

    monkeypatch.setattr(module.os, "replace", replace)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    with pytest.raises(error):
        module.atomic_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"old": True}
    assert len(sleeps) == expected_sleeps
    assert len(list(tmp_path.glob("*.pending"))) == 1


def evidence():
    journal = {"completed_updates": 100, "durable_updates": 100, "pending_update_upper": 0}
    receipt = {"completed_updates": 100, "complete_state": True, "status": "PAUSED_RESOURCE"}
    pause = {
        "completed_updates": 100,
        "snapshot_valid": True,
        "host_available_bytes": 4 * GIB,
        "windows_commit_headroom_bytes": GIB,
    }
    return journal, receipt, pause


def test_clean_memory_pause_can_resume():
    assert classify_exit(3, *evidence(), manual=False) == "WAITING_MEMORY"


def test_manual_stop_is_never_restarted():
    assert classify_exit(3, *evidence(), manual=True) == "MANUAL_STOP"


def test_checkpoint_memory_error_is_not_restarted_implicitly():
    assert classify_exit(1, *evidence(), manual=False) == "BLOCKED_ERROR"


def test_uncommitted_updates_and_stale_pause_are_rejected():
    j, r, p = evidence()
    j["completed_updates"] = 101
    assert classify_exit(3, j, r, p, manual=False) == "BLOCKED_ERROR"


def test_other_resource_and_io_failures_are_not_memory_retries():
    j, r, p = evidence()
    p["windows_commit_headroom_bytes"] = 4 * GIB
    assert classify_exit(3, j, r, p, manual=False) == "BLOCKED_ERROR"
    assert classify_exit(3, j, r, {"kind": "transient_input_io"}, manual=False) == "BLOCKED_ERROR"


def test_untrusted_snapshot_is_not_memory_retry():
    j, r, p = evidence()
    p["snapshot_valid"] = False
    assert classify_exit(3, j, r, p, manual=False) == "BLOCKED_ERROR"


def test_complete_is_terminal_only_with_endpoint_receipt():
    j, r, p = evidence()
    assert classify_exit(0, j, r, p, manual=False) == "BLOCKED_ERROR"
    r.update(status="COMPLETE", scientific_endpoint=True)
    assert classify_exit(0, j, r, p, manual=False) == "COMPLETE"


def test_both_memory_limits_and_continuous_stability_are_required():
    gate = ResumeGate()
    assert not gate.ready(0, True, 3 * GIB, 3 * GIB)
    assert not gate.ready(29, True, 3 * GIB, 3 * GIB)
    assert not gate.ready(30, True, 3 * GIB, 2 * GIB - 1)
    assert not gate.ready(31, True, 3 * GIB, 3 * GIB)
    assert gate.ready(61, True, 2 * GIB, 2 * GIB)


def test_invalid_inventory_or_unknown_commit_resets_stability():
    gate = ResumeGate()
    gate.ready(0, True, 3 * GIB, 3 * GIB)
    assert not gate.ready(30, False, 3 * GIB, 3 * GIB)
    assert not gate.ready(60, True, 3 * GIB, None)
    assert not gate.ready(90, True, 3 * GIB, 3 * GIB)
    assert gate.ready(120, True, 3 * GIB, 3 * GIB)


def test_launch_resets_stability_for_next_producer():
    gate = ResumeGate()
    gate.ready(0, True, 3 * GIB, 3 * GIB)
    assert gate.ready(30, True, 3 * GIB, 3 * GIB)
    gate.reset()
    assert not gate.ready(31, True, 3 * GIB, 3 * GIB)


class FakeProcess:
    pid = 123

    def __init__(self, code):
        self.code = code

    def poll(self):
        return self.code


def make_supervisor(tmp_path):
    run = tmp_path / "run"
    fit = run / "fits" / "R_A5"
    fit.mkdir(parents=True)
    j, r, p = evidence()
    for name, value in zip(
        ["UPDATE_JOURNAL.json", "CHECKPOINT_RECEIPT.json", "RESOURCE_PAUSE.json"],
        [j, r, p],
        strict=True,
    ):
        (fit / name).write_text(json.dumps(value))
    return Supervisor(argparse.Namespace(run=run))


def test_child_memory_exit_schedules_backoff_once_and_keeps_other_child(tmp_path, monkeypatch):
    supervisor = make_supervisor(tmp_path)
    monkeypatch.setattr("operational.final_refit.auto_resume.time.monotonic", lambda: 1000)
    supervisor.children = {"R_A5": FakeProcess(3)}
    supervisor.states = {"R_A5": "RUNNING", "R_C2F": "RUNNING"}
    supervisor.observe_children()
    assert supervisor.states == {"R_A5": "WAITING_MEMORY", "R_C2F": "RUNNING"}
    assert supervisor.eligible_at["R_A5"] == 1060
    supervisor.observe_children()
    assert supervisor.retries["R_A5"] == 1
    supervisor.states["R_A5"] = "RUNNING"
    supervisor.observe_children()
    assert supervisor.eligible_at["R_A5"] == 1120
    assert supervisor.retries["R_A5"] == 2


def test_duplicate_start_is_rejected_before_spawning(tmp_path):
    supervisor = make_supervisor(tmp_path)
    supervisor.children = {"R_A5": FakeProcess(None)}
    with pytest.raises(RuntimeError, match="Duplicate"):
        supervisor.start_fit("R_A5")


def test_manual_marker_is_retained_and_prevents_retry(tmp_path):
    supervisor = make_supervisor(tmp_path)
    marker = supervisor.run / "fits" / "R_A5" / "STOP_REQUEST"
    marker.write_text("Manual pause")
    supervisor.children = {"R_A5": FakeProcess(3)}
    supervisor.states = {"R_A5": "RUNNING"}
    supervisor.observe_children()
    assert supervisor.states["R_A5"] == "MANUAL_STOP"
    assert supervisor.retries["R_A5"] == 0
    assert marker.read_text() == "Manual pause"


def test_failed_checkpoint_stays_blocked_without_retry(tmp_path):
    supervisor = make_supervisor(tmp_path)
    supervisor.children = {"R_A5": FakeProcess(1)}
    supervisor.states = {"R_A5": "RUNNING"}
    supervisor.observe_children()
    assert supervisor.states["R_A5"] == "BLOCKED_ERROR"
    assert supervisor.retries["R_A5"] == 0


def stale_resources():
    return {
        "snapshot_valid": False,
        "run_root_binding_valid": True,
        "snapshot_error": "ValueError: snapshot is stale",
        "host_available_bytes": 4 * GIB,
        "windows_commit_headroom_bytes": 4 * GIB,
    }


@pytest.mark.parametrize(
    "reason",
    [
        "ValueError: snapshot is stale",
        "ValueError: publisher PID identity is not live",
    ],
)
def test_clean_monitor_pause_is_recoverable_with_saved_state(reason):
    j, r, p = evidence()
    p.update(stale_resources(), snapshot_error=reason)
    assert classify_exit(3, j, r, p, manual=False) == "WAITING_MONITOR"
    assert classify_exit(3, j, r, p, manual=True) == "MANUAL_STOP"
    j["pending_update_upper"] = 1
    assert classify_exit(3, j, r, p, manual=False) == "BLOCKED_ERROR"


@pytest.mark.parametrize(
    "change",
    [
        {"snapshot_error": "ValueError: signature mismatch"},
        {"snapshot_error": "ValueError: snapshot sequence regressed"},
        {"run_root_binding_valid": False},
    ],
)
def test_monitor_integrity_failures_are_never_retried(change):
    j, r, p = evidence()
    p.update(stale_resources())
    p.update(change)
    assert classify_exit(3, j, r, p, manual=False) == "BLOCKED_ERROR"


def test_publisher_is_never_replaced_while_trainer_alive(tmp_path, monkeypatch):
    supervisor = make_supervisor(tmp_path)
    supervisor.publisher = FakeProcess(1)
    supervisor.children = {"R_A5": FakeProcess(None)}
    monkeypatch.setattr(supervisor, "start_publisher", lambda: pytest.fail("live trainer"))
    assert supervisor.recover_publisher(stale_resources(), 1000) is None
    assert supervisor.recover_publisher(stale_resources(), 2000) is None
    with pytest.raises(RuntimeError, match="producer is alive"):
        supervisor.stop_publisher()


def test_stalled_publisher_recovers_only_after_idle_grace_period(tmp_path, monkeypatch):
    supervisor = make_supervisor(tmp_path)
    supervisor.publisher = FakeProcess(None)
    calls = []
    new_monitor = object()
    monkeypatch.setattr(supervisor, "stop_publisher", lambda: calls.append("stop"))
    monkeypatch.setattr(supervisor, "start_publisher", lambda: new_monitor)
    assert supervisor.recover_publisher(stale_resources(), 1000) is None
    assert supervisor.recover_publisher(stale_resources(), 1059) is None
    assert calls == []
    assert supervisor.recover_publisher(stale_resources(), 1060) is new_monitor
    assert calls == ["stop"]
    assert supervisor.publisher_restarts == 1
    assert not supervisor.gate.ready(1060, True, 4 * GIB, 4 * GIB)
    assert supervisor.gate.ready(1090, True, 4 * GIB, 4 * GIB)


def test_dead_publisher_can_restart_immediately_but_retries_are_bounded(tmp_path, monkeypatch):
    supervisor = make_supervisor(tmp_path)
    supervisor.publisher = FakeProcess(1)
    new_monitor = object()
    monkeypatch.setattr(supervisor, "stop_publisher", lambda: None)
    monkeypatch.setattr(supervisor, "start_publisher", lambda: new_monitor)
    for i in range(3):
        assert supervisor.recover_publisher(stale_resources(), 1000 + i) is new_monitor
    with pytest.raises(RuntimeError, match="three recoveries"):
        supervisor.recover_publisher(stale_resources(), 2000)


def test_live_publisher_integrity_error_does_not_trigger_replacement(tmp_path, monkeypatch):
    supervisor = make_supervisor(tmp_path)
    supervisor.publisher = FakeProcess(None)
    resources = stale_resources()
    resources["snapshot_error"] = "ValueError: signature mismatch"
    monkeypatch.setattr(supervisor, "start_publisher", lambda: pytest.fail("integrity failure"))
    assert supervisor.recover_publisher(resources, 1000) is None
    assert supervisor.recover_publisher(resources, 2000) is None


def test_fresh_inventory_clears_outage_clock(tmp_path):
    supervisor = make_supervisor(tmp_path)
    supervisor.publisher = FakeProcess(None)
    assert supervisor.recover_publisher(stale_resources(), 1000) is None
    assert supervisor.monitor_unavailable_since == 1000
    assert supervisor.recover_publisher({"snapshot_valid": True}, 1010) is None
    assert supervisor.monitor_unavailable_since is None


def manual_resume_fixture(tmp_path):
    supervisor = make_supervisor(tmp_path)
    (supervisor.run / "PRODUCER_PLAN.json").write_text('{"executed_source_sha256": {}}')
    (supervisor.run / "AUTO_RESUME_STOP").write_text("User pause")
    for fit in ("R_A5", "R_C2F"):
        directory = supervisor.run / "fits" / fit
        directory.mkdir(exist_ok=True)
        checkpoint = directory / "checkpoint_last.pt"
        checkpoint.write_bytes(b"saved-state")
        j, r, p = evidence()
        j.update(identity_sha256="identity", execution_binding_sha256="binding")
        r.update(
            identity_sha256="identity",
            execution_binding_sha256="binding",
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(b"saved-state").hexdigest(),
        )
        p.update(run_root_binding_valid=True, windows_commit_headroom_bytes=4 * GIB)
        for name, value in zip(
            ["UPDATE_JOURNAL.json", "CHECKPOINT_RECEIPT.json", "RESOURCE_PAUSE.json"],
            [j, r, p],
            strict=True,
        ):
            (directory / name).write_text(json.dumps(value))
        (directory / "STOP_REQUEST").write_text("User pause")
    previous = {
        "fit_states": dict.fromkeys(("R_A5", "R_C2F"), "MANUAL_STOP"),
        "producer_exit_codes": dict.fromkeys(("R_A5", "R_C2F"), 3),
    }
    return supervisor, previous


def test_explicit_manual_resume_archives_markers_and_still_waits_for_resources(tmp_path):
    supervisor, previous = manual_resume_fixture(tmp_path)
    supervisor.resume_manual(previous)
    assert set(supervisor.states.values()) == {"WAITING_RESOURCES"}
    assert not (supervisor.run / "AUTO_RESUME_STOP").exists()
    assert not supervisor.manual("R_A5")
    assert len(list(supervisor.run.rglob("*.resumed_*"))) == 3
    assert len(list(supervisor.run.glob("MANUAL_RESUME_*.json"))) == 1
    assert not supervisor.gate.ready(0, True, 4 * GIB, 4 * GIB)


@pytest.mark.parametrize("failure", ["hash", "pending", "exit", "signature", "binding"])
def test_manual_resume_rejects_errors_before_archiving_any_marker(tmp_path, failure):
    supervisor, previous = manual_resume_fixture(tmp_path)
    directory = supervisor.run / "fits" / "R_C2F"
    if failure == "hash":
        (directory / "checkpoint_last.pt").write_bytes(b"corrupted")
    elif failure == "exit":
        previous["producer_exit_codes"]["R_C2F"] = 1
    else:
        name = "RESOURCE_PAUSE.json" if failure == "signature" else "UPDATE_JOURNAL.json"
        path = directory / name
        value = json.loads(path.read_text())
        if failure == "signature":
            value.update(snapshot_valid=False, snapshot_error="ValueError: signature mismatch")
        elif failure == "binding":
            value["execution_binding_sha256"] = "changed"
        else:
            value["pending_update_upper"] = 1
        path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError):
        supervisor.resume_manual(previous)
    assert (supervisor.run / "AUTO_RESUME_STOP").exists()
    assert (directory / "STOP_REQUEST").exists()
    assert not list(supervisor.run.rglob("*.resumed_*"))


@pytest.mark.parametrize("invalid_endpoint", [False, True])
def test_manual_resume_preserves_completed_fit_and_requires_endpoint(tmp_path, invalid_endpoint):
    supervisor, previous = manual_resume_fixture(tmp_path)
    previous["fit_states"]["R_A5"] = "COMPLETE"
    previous["producer_exit_codes"]["R_A5"] = 0
    path = supervisor.run / "fits" / "R_A5" / "CHECKPOINT_RECEIPT.json"
    receipt = json.loads(path.read_text())
    receipt.update(status="COMPLETE", scientific_endpoint=not invalid_endpoint)
    path.write_text(json.dumps(receipt))
    if invalid_endpoint:
        with pytest.raises(RuntimeError):
            supervisor.resume_manual(previous)
        assert (supervisor.run / "AUTO_RESUME_STOP").exists()
    else:
        supervisor.resume_manual(previous)
        assert supervisor.states == {"R_A5": "COMPLETE", "R_C2F": "WAITING_RESOURCES"}


def unreadable_resources():
    return {
        "snapshot_valid": True,
        "run_root_binding_valid": True,
        "process_coverage_bounded": False,
        "publisher_scan_error": None,
        "host_available_bytes": 8 * GIB,
        "windows_commit_headroom_bytes": 8 * GIB,
        "inspection_failures": [{"pid": 14756, "error": "AccessDenied", "rss_upper_bound": -1}],
    }


def test_saved_access_denied_pause_waits_without_terminating_supervision():
    j, r, p = evidence()
    p.update(unreadable_resources())
    assert classify_exit(3, j, r, p, manual=False) == "WAITING_MONITOR"
    assert classify_exit(3, j, r, p, manual=True) == "MANUAL_STOP"
    assert classify_exit(1, j, r, p, manual=False) == "BLOCKED_ERROR"
    r["complete_state"] = False
    assert classify_exit(3, j, r, p, manual=False) == "BLOCKED_ERROR"


@pytest.mark.parametrize(
    "change",
    [
        {"snapshot_valid": False},
        {"run_root_binding_valid": False},
        {"publisher_scan_error": "RuntimeError: unexpected failure"},
        {"process_coverage_bounded": True},
        {"inspection_failures": []},
        {"inspection_failures": [{"pid": 1, "error": "ValueError", "rss_upper_bound": -1}]},
    ],
)
def test_coverage_retry_does_not_swallow_other_failures(change):
    j, r, p = evidence()
    p.update(unreadable_resources())
    p.update(change)
    assert classify_exit(3, j, r, p, manual=False) == "BLOCKED_ERROR"


def test_persistent_access_denied_keeps_publisher_and_blocks_launch(tmp_path, monkeypatch):
    supervisor = make_supervisor(tmp_path)
    supervisor.publisher = FakeProcess(None)
    monkeypatch.setattr(supervisor, "start_publisher", lambda: pytest.fail("healthy publisher"))
    for now in (0, 60, 600, 3600):
        assert supervisor.recover_publisher(unreadable_resources(), now) is None
        assert not supervisor.gate.ready(now, False, 8 * GIB, 8 * GIB)
    assert supervisor.publisher_restarts == 0
    assert not supervisor.gate.ready(3610, True, 8 * GIB, 8 * GIB)
    assert not supervisor.gate.ready(3639, True, 8 * GIB, 8 * GIB)
    assert supervisor.gate.ready(3640, True, 8 * GIB, 8 * GIB)


def test_coverage_pause_schedules_one_retry_and_preserves_completed_other_fit(
    tmp_path, monkeypatch
):
    supervisor = make_supervisor(tmp_path)
    path = supervisor.run / "fits" / "R_A5" / "RESOURCE_PAUSE.json"
    pause = json.loads(path.read_text())
    pause.update(unreadable_resources())
    path.write_text(json.dumps(pause))
    supervisor.children = {"R_A5": FakeProcess(3)}
    supervisor.states = {"R_A5": "RUNNING", "R_C2F": "COMPLETE"}
    monkeypatch.setattr("operational.final_refit.auto_resume.time.monotonic", lambda: 1000)
    supervisor.observe_children()
    supervisor.observe_children()
    assert supervisor.states == {"R_A5": "WAITING_MONITOR", "R_C2F": "COMPLETE"}
    assert supervisor.retries["R_A5"] == 1
    assert supervisor.eligible_at["R_A5"] == 1060
