import json
from pathlib import Path

import pytest

from operational.final_refit import continuation_queue as queue


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def setup_queue(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    locator = tmp_path / "ACTIVE_RUNTIME.json"
    launcher = runtime / "launcher.json"
    write(
        locator,
        {
            "active_run_relative_to_this_file": "runtime",
            "launcher_receipt_relative_to_this_file": "runtime/launcher.json",
        },
    )
    write(
        launcher,
        {"status": "SUPERVISOR_EXITED", "fit_states": {"R_A5": "COMPLETE", "R_C2F": "COMPLETE"}},
    )
    write(runtime / "PRODUCER_PLAN.json", {"executed_source_sha256": {}})
    for fit in ("R_A5", "R_C2F"):
        directory = runtime / "fits" / fit
        directory.mkdir(parents=True)
        checkpoint = directory / "checkpoint_last.pt"
        checkpoint.write_bytes(b"test checkpoint payload")
        identity = {
            "completed_updates": 49932,
            "durable_updates": 49932,
            "pending_update_upper": 0,
            "identity_sha256": "a" * 64,
            "execution_binding_sha256": "b" * 64,
        }
        write(directory / "UPDATE_JOURNAL.json", identity)
        write(
            directory / "CHECKPOINT_RECEIPT.json",
            {
                **identity,
                "complete_state": True,
                "scientific_endpoint": True,
                "status": "COMPLETE",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": queue.file_digest(checkpoint),
            },
        )
    plan = {
        "schema": queue.SCHEMA,
        "repository": str(tmp_path),
        "runtime": str(runtime),
        "locator": str(locator),
        "fold_contract": str(tmp_path / "FOLD_CONTRACT.json"),
        "source_bindings": {},
        "submission_authorized": False,
        "nodes": [
            {"id": "RGB_ENDPOINTS", "depends_on": [], "executor": "wait_rgb"},
            {"id": "VERIFY_OOF_ROLES", "depends_on": ["RGB_ENDPOINTS"], "executor": "verify_roles"},
            {
                "id": "OOF_SCIENTIFIC_IMPLEMENTATION",
                "depends_on": ["VERIFY_OOF_ROLES"],
                "executor": "unimplemented",
            },
            {
                "id": "TEST12_PREDICTIONS",
                "depends_on": ["OOF_SCIENTIFIC_IMPLEMENTATION"],
                "executor": "unimplemented",
            },
        ],
    }
    plan["plan_sha256"] = queue.canonical_sha256(plan)
    output = tmp_path / "queue"
    output.mkdir()
    monkeypatch.setattr(queue.psutil, "process_iter", lambda *args: [])
    return plan, output, launcher


def test_waits_for_upstream_without_touching_oof(setup_queue, monkeypatch):
    plan, output, launcher = setup_queue
    value = queue.read(launcher)
    value["status"] = "SUPERVISING"
    value["fit_states"]["R_C2F"] = "RUNNING"
    write(launcher, value)
    monkeypatch.setattr(queue, "verify_roles", lambda _: pytest.fail("Too early for OOF"))
    state = queue.tick(plan, output)
    assert state["status"] == "WAITING_UPSTREAM"
    assert state["stages"]["TEST12_PREDICTIONS"] == "WAITING_DEPENDENCY"


def test_finished_producers_do_not_bypass_missing_scientific_runner(setup_queue, monkeypatch):
    plan, output, _ = setup_queue
    monkeypatch.setattr(queue, "verify_roles", lambda _: {"optimizer_updates": 0})
    state = queue.tick(plan, output)
    assert state["status"] == "BLOCKED_IMPLEMENTATION"
    assert state["stages"]["VERIFY_OOF_ROLES"] == "COMPLETE"
    assert state["stages"]["TEST12_PREDICTIONS"] == "WAITING_DEPENDENCY"
    assert state["scientific_updates_executed"] == 0
    assert state["submission_performed"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("complete_state", False),
        ("scientific_endpoint", False),
        ("completed_updates", 49931),
        ("identity_sha256", "changed"),
        ("execution_binding_sha256", "changed"),
        ("status", "PAUSED_RESOURCE"),
    ],
)
def test_inconsistent_endpoint_is_rejected(setup_queue, key, value):
    plan, output, _ = setup_queue
    path = Path(plan["runtime"]) / "fits/R_C2F/CHECKPOINT_RECEIPT.json"
    receipt = queue.read(path)
    receipt[key] = value
    write(path, receipt)
    with pytest.raises(ValueError, match="inconsistent"):
        queue.tick(plan, output)


def test_corrupt_checkpoint_rejected_before_oof(setup_queue):
    plan, output, _ = setup_queue
    (Path(plan["runtime"]) / "fits/R_C2F/checkpoint_last.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checkpoint hash"):
        queue.tick(plan, output)


@pytest.mark.parametrize("location", ["queue", "global", "fit"])
def test_manual_stops_are_preserved(setup_queue, location):
    plan, output, _ = setup_queue
    runtime = Path(plan["runtime"])
    stop = {
        "queue": output / "STOP_REQUEST",
        "global": runtime / "AUTO_RESUME_STOP",
        "fit": runtime / "fits/R_C2F/STOP_REQUEST",
    }[location]
    stop.write_text("manual")
    assert queue.tick(plan, output)["status"] == "MANUAL_STOP"
    assert stop.read_text() == "manual"


def test_duplicate_queue_lock_and_recovery_after_release(tmp_path):
    path = tmp_path / "queue.lock"
    with queue.queue_lock(path):
        with pytest.raises(OSError):
            with queue.queue_lock(path):
                pytest.fail("Duplicate owner admitted")
    with queue.queue_lock(path):
        pass


def test_plan_change_cannot_enable_implicit_submission(setup_queue):
    plan, _, _ = setup_queue
    plan["submission_authorized"] = True
    with pytest.raises(ValueError, match="digest"):
        queue.validate_plan(plan)
    plan.pop("plan_sha256")
    plan["plan_sha256"] = queue.canonical_sha256(plan)
    with pytest.raises(ValueError, match="cannot submit"):
        queue.validate_plan(plan)


def test_recovery_revalidates_completed_nodes(setup_queue, monkeypatch):
    plan, output, _ = setup_queue
    path = output / "QUEUE_PLAN.json"
    write(path, plan)
    monkeypatch.setattr(queue, "verify_roles", lambda _: {"optimizer_updates": 0})
    assert queue.run_queue(path, once=True) == 3
    (Path(plan["runtime"]) / "fits/R_C2F/checkpoint_last.pt").write_bytes(b"corrupt")
    assert queue.run_queue(path, once=True) == 3
    state = queue.read(output / "QUEUE_STATUS.json")
    assert state["status"] == "BLOCKED_INTEGRITY"
    assert "Checkpoint hash mismatch" in state["error"]


def test_dag_rejects_cycle_and_unregistered_executor():
    with pytest.raises(ValueError, match="dependency"):
        queue.topology([{"id": "a", "depends_on": ["b"], "executor": "wait_rgb"}])
    with pytest.raises(ValueError, match="Unregistered"):
        queue.topology([{"id": "a", "depends_on": [], "executor": "shell"}])


def test_plan_binds_actual_sources_and_complete_dependencies(tmp_path):
    repo = Path(queue.__file__).resolve().parents[2]
    runtime = tmp_path / "runtime"
    write(runtime / "PRODUCER_PLAN.json", {})
    folds = tmp_path / "FOLD_CONTRACT.json"
    write(folds, {})
    plan = queue.build_plan(repo, runtime, tmp_path / "locator.json", folds)
    queue.validate_plan(plan)
    nodes = {n["id"]: n for n in plan["nodes"]}
    assert len([n for n in nodes if n.endswith("_GENERATION")]) == 10
    assert nodes["TEST12_PREDICTIONS"]["depends_on"] == ["TEST12_INPUT_ADMISSION"]
    assert nodes["FINAL_E_CTX"]["depends_on"] == ["OOF_NORMALIZERS"]
    assert plan["end_to_end_executable"] is False
    assert nodes["FINAL_PHASE_ONLY"]["executor"] == "unimplemented"
    write(folds, {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        queue.validate_plan(plan)
