from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from operational.final_refit import scientific_oof_queue as queue


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class Monitor:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.calls = 0

    def guard(self, run_root: Path, **kwargs: object) -> tuple[bool, dict[str, Any]]:
        del run_root, kwargs
        self.calls += 1
        return self.ready, {
            "snapshot_valid": self.ready,
            "run_root_binding_valid": True,
            "process_coverage_bounded": True,
            "publisher_scan_error": None,
            "project_tree_rss_upper_bound_bytes": 1,
            "host_available_bytes": 2 * queue.GIB,
            "windows_commit_headroom_bytes": 2 * queue.GIB,
            "disk_free_after_reservation_bytes": 20_000_000_000,
        }


class Child:
    pid = 1234

    def __init__(self, code: int) -> None:
        self.returncode: int | None = code

    def poll(self) -> int:
        assert self.returncode is not None
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0


class PublisherChild(Child):
    pid = 4321
    returncode: int | None

    def __init__(self) -> None:
        self.returncode = None

    def poll(self) -> int | None:
        return self.returncode


class Launcher:
    def __init__(self, plan: dict[str, Any], modes: list[str]) -> None:
        self.plan = plan
        self.modes = modes
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: object) -> Child:
        del kwargs
        self.commands.append(command)
        stage_id = command[command.index("-StageId") + 1]
        result_path = Path(command[command.index("-ResultPath") + 1])
        intent_nonce = command[command.index("-IntentNonce") + 1]
        mode = self.modes.pop(0)
        code = {"complete": 0, "pause": 3, "error": 9}[mode]
        if mode != "error":
            stage = next(item for item in self.plan["stages"] if item["id"] == stage_id)
            receipt = {
                "stage_id": stage_id,
                "status": "COMPLETE" if mode == "complete" else "PAUSED_RESOURCE",
                "checkpoint_updates": 100,
                "completed_updates": 100 if mode == "complete" else 0,
                "durable_updates": 100 if mode == "complete" else 0,
                "pending_update_upper": 0,
                "manual_stop": False,
                "scientific_endpoint": mode == "complete",
            }
            if mode == "complete":
                evidence = Path(stage["receipt"]).parent / "admission.json"
                fragment_file = Path(stage["receipt"]).parent / "fragment.json"
                write(evidence, {"admitted": True})
                write(fragment_file, {"fragment": stage_id})
                fragments = {str(fragment_file.resolve()): queue.file_digest(fragment_file)}
                receipt.update(
                    admissions_sha256={str(evidence.resolve()): queue.file_digest(evidence)},
                    fragment_files_sha256=fragments,
                    fragment_sha256=queue.canonical_sha256(fragments),
                )
                if stage_id == "FINAL":
                    final = Path(stage["receipt"]).parent
                    artifacts = {
                        "model_seal": final / "MODEL_SEAL.json",
                        "head_export": final / "PHASE_ONLY_HEAD.ts",
                        "head_export_receipt": final / "PHASE_ONLY_HEAD.json",
                        "predictions": final / "test12/predictions.csv",
                        "prediction_receipt": final / "test12/PREDICTION_RECEIPT.json",
                        "submission_zip": final / "test12/submission.zip",
                        "package_receipt": final / "test12/submission.json",
                    }
                    for name in ("model_seal", "head_export_receipt"):
                        write(artifacts[name], {"artifact": name})
                    artifacts["head_export"].parent.mkdir(parents=True, exist_ok=True)
                    artifacts["head_export"].write_bytes(b"head")
                    artifacts["predictions"].parent.mkdir(parents=True, exist_ok=True)
                    artifacts["predictions"].write_text(
                        "sample_token,prediction\na,1\n", encoding="utf-8"
                    )
                    artifacts["submission_zip"].write_bytes(b"zip")
                    seal_sha = queue.file_digest(artifacts["model_seal"])
                    write(
                        artifacts["prediction_receipt"],
                        {
                            "schema": "scientific_oof_test12_prediction_receipt_v1",
                            "status": "COMPLETE",
                            "model_seal_sha256": seal_sha,
                            "predictions_sha256": queue.file_digest(artifacts["predictions"]),
                            "private_targets_read": False,
                            "external_submission": False,
                        },
                    )
                    package = {
                        "schema": "scientific_oof_test12_package_v1",
                        "status": "PACKAGED_NOT_SUBMITTED",
                        "model_seal_sha256": seal_sha,
                        "prediction_receipt_sha256": queue.file_digest(
                            artifacts["prediction_receipt"]
                        ),
                        "zip_path": str(artifacts["submission_zip"].resolve()),
                        "zip_sha256": queue.file_digest(artifacts["submission_zip"]),
                        "private_targets_read": False,
                        "external_submission": False,
                    }
                    package["receipt_sha256"] = queue.canonical_sha256(package)
                    write(artifacts["package_receipt"], package)
                    resolved = {name: str(path.resolve()) for name, path in artifacts.items()}
                    receipt.update(
                        all_head_endpoints_complete=True,
                        all_oof_inputs_complete=True,
                        artifact_paths=resolved,
                        artifact_sha256={
                            name: queue.file_digest(path) for name, path in artifacts.items()
                        },
                    )
                    receipt["receipt_sha256"] = queue.canonical_sha256(receipt)
            write(Path(stage["receipt"]), receipt)
        write(
            result_path,
            {
                "schema": "final_refit_scientific_oof_launcher_result_v1",
                "stage_id": stage_id,
                "intent_nonce": intent_nonce,
                "exit_code": code,
            },
        )
        return Child(code)


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repository = tmp_path / "repo"
    run = tmp_path / "run"
    repository.mkdir()
    run.mkdir()
    for relative in (
        "operational/final_refit/continuation_queue.py",
        "operational/final_refit/shared_resources.py",
        "operational/final_refit/scientific_oof_pipeline.py",
        "operational/final_refit/scientific_oof_final.py",
    ):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    legacy = tmp_path / "legacy.json"
    write(legacy, {"schema": "legacy"})
    config = run / "SCIENTIFIC_CONFIG.json"
    write(
        config,
        {
            "checkpoint_updates": 100,
            "physical_min_bytes": 2 * queue.GIB,
            "commit_min_bytes": 2 * queue.GIB,
        },
    )
    launcher = tmp_path / "launcher.ps1"
    launcher.write_text("Start-Process -WindowStyle Hidden", encoding="utf-8")
    python = tmp_path / "python.exe"
    powershell = tmp_path / "powershell.exe"
    python.write_bytes(b"python")
    powershell.write_bytes(b"powershell")
    plan = queue.build_plan(
        repository,
        run,
        legacy,
        config,
        launcher,
        python_executable=python,
        powershell_executable=powershell,
    )
    monkeypatch.setattr(queue, "rgb_endpoints", lambda _: (True, {"verified": True}))
    monkeypatch.setattr(
        queue.psutil,
        "Process",
        lambda pid=None: SimpleNamespace(create_time=lambda: 10.0, is_running=lambda: False),
    )
    return plan, run, config


def make_queue(
    plan: dict[str, Any],
    modes: list[str],
    *,
    monitor: Monitor | None = None,
    clock: list[float] | None = None,
) -> tuple[queue.ScientificOOFQueue, Launcher]:
    launcher = Launcher(plan, modes)
    values = clock if clock is not None else [1_000.0]
    scheduler = queue.ScientificOOFQueue(
        plan,
        monitor=monitor or Monitor(),  # type: ignore[arg-type]
        popen_factory=launcher,
        monotonic=lambda: values[0],
        sleeper=lambda _: None,
    )
    return scheduler, launcher


def test_plan_freezes_exact_sequential_commands_and_policy(prepared) -> None:
    plan, run, _ = prepared
    queue.validate_plan(plan)
    assert [stage["id"] for stage in plan["stages"]] == [
        "ROTATION_0",
        "ROTATION_1",
        "ROTATION_2",
        "ROTATION_3",
        "ROTATION_4",
        "FINAL",
    ]
    assert plan["stages"][0]["arguments"] == [
        "--run",
        str(run),
        "--rotation",
        "0",
    ]
    assert plan["stages"][-1]["arguments"] == ["--run", str(run)]
    assert plan["resource_policy"]["checkpoint_updates"] == 100
    assert plan["submission_authorized"] is False
    assert plan["python_executable"] in plan["source_bindings"]
    assert plan["powershell_executable"] in plan["source_bindings"]


def test_build_refuses_missing_scientific_executor(prepared) -> None:
    plan, _, _ = prepared
    Path(plan["repository"], "operational/final_refit/scientific_oof_final.py").unlink()
    with pytest.raises(FileNotFoundError, match="unavailable OOF executors"):
        queue.build_plan(
            Path(plan["repository"]),
            Path(plan["run"]),
            Path(plan["legacy_continuation_plan"]),
            Path(plan["scientific_config"]),
            Path(plan["launcher"]),
            python_executable=Path(plan["python_executable"]),
            powershell_executable=Path(plan["powershell_executable"]),
        )


def test_waits_for_rgb_before_resources_or_spawn(prepared, monkeypatch) -> None:
    plan, _, _ = prepared
    monkeypatch.setattr(queue, "rgb_endpoints", lambda _: (False, {"running": True}))
    monitor = Monitor()
    scheduler, launcher = make_queue(plan, [], monitor=monitor)
    state = scheduler.tick()
    assert state["status"] == "WAITING_RGB"
    assert monitor.calls == 0
    assert launcher.commands == []


def test_resource_snapshot_blocks_spawn(prepared) -> None:
    plan, _, _ = prepared
    monitor = Monitor(ready=False)
    scheduler, launcher = make_queue(plan, [], monitor=monitor)
    assert scheduler.tick()["status"] == "WAITING_RESOURCES"
    assert launcher.commands == []


def test_queue_creates_session_and_owns_publisher_until_cleanup(prepared) -> None:
    plan, _, _ = prepared

    class BecomingReady(Monitor):
        def guard(self, run_root: Path, **kwargs: object) -> tuple[bool, dict[str, Any]]:
            self.calls += 1
            self.ready = self.calls >= 2
            return super().guard(run_root, **kwargs)

    monitor = BecomingReady(ready=False)
    child = PublisherChild()
    commands: list[list[str]] = []

    def factory(command: list[str], **kwargs: object) -> PublisherChild:
        del kwargs
        commands.append(command)
        return child

    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    scheduler = queue.ScientificOOFQueue(
        plan,
        popen_factory=factory,
        monotonic=lambda: clock[0],
        sleeper=sleep,
    )
    scheduler.monitor = monitor  # type: ignore[assignment]
    state = scheduler.load_state()
    scheduler._ensure_publisher(state)
    assert Path(plan["resource_session"]).is_file()
    assert commands[0][1:4] == ["-m", "operational.final_refit.shared_resources", "publish"]
    assert state["publisher"]["owned"] is True
    scheduler._cleanup_owned_publisher(state)
    assert child.returncode == 0


def test_executes_six_stages_sequentially_through_hidden_launcher(prepared) -> None:
    plan, _, _ = prepared
    scheduler, launcher = make_queue(plan, ["complete"] * 6)
    for index in range(6):
        state = scheduler.tick()
        assert state["stages"][plan["stages"][index]["id"]]["status"] == "COMPLETE"
        assert len(launcher.commands) == index + 1
    assert scheduler.tick()["status"] == "COMPLETE"
    first = launcher.commands[0]
    assert first[:6] == [
        plan["powershell_executable"],
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
    ]
    assert first[first.index("-Module") + 1] == plan["stages"][0]["module"]
    assert first[first.index("-Rotation") + 1] == "0"
    assert "-Rotation" not in launcher.commands[-1]


def test_only_receipt_proven_resource_pause_retries(prepared) -> None:
    plan, _, _ = prepared
    clock = [100.0]
    scheduler, launcher = make_queue(plan, ["pause", "complete"], clock=clock)
    state = scheduler.tick()
    assert state["status"] == "WAITING_RESOURCES"
    assert state["stages"]["ROTATION_0"]["attempts"] == 1
    scheduler.tick()
    assert len(launcher.commands) == 1
    clock[0] = 161.0
    state = scheduler.tick()
    assert state["stages"]["ROTATION_0"]["status"] == "COMPLETE"
    assert state["stages"]["ROTATION_0"]["attempts"] == 2


def test_nonresource_failure_is_terminal_and_not_retried(prepared) -> None:
    plan, _, _ = prepared
    scheduler, launcher = make_queue(plan, ["error"])
    assert scheduler.tick()["status"] == "BLOCKED_ERROR"
    assert scheduler.tick()["status"] == "BLOCKED_ERROR"
    assert len(launcher.commands) == 1


def test_manual_stop_is_persistent_and_never_consumed(prepared) -> None:
    plan, run, _ = prepared
    stop = run / "STOP_REQUEST"
    stop.write_text("user stop", encoding="utf-8")
    scheduler, launcher = make_queue(plan, [])
    assert scheduler.tick()["status"] == "MANUAL_STOP"
    assert stop.read_text(encoding="utf-8") == "user stop"
    assert launcher.commands == []


def test_completed_receipt_is_revalidated_on_restart(prepared) -> None:
    plan, _, _ = prepared
    scheduler, _ = make_queue(plan, ["complete"])
    scheduler.tick()
    receipt = Path(plan["stages"][0]["receipt"])
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["fragment_sha256"] = "corrupt"
    write(receipt, value)
    with pytest.raises(ValueError, match="fragment"):
        scheduler.tick()


def test_completed_receipt_rehashes_every_bound_artifact(prepared) -> None:
    plan, _, _ = prepared
    scheduler, _ = make_queue(plan, ["complete"])
    scheduler.tick()
    receipt = json.loads(Path(plan["stages"][0]["receipt"]).read_text(encoding="utf-8"))
    admission = Path(next(iter(receipt["admissions_sha256"])))
    write(admission, {"admitted": False})
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        scheduler.tick()


def test_final_receipt_rejects_missing_or_stale_zip(prepared) -> None:
    plan, _, _ = prepared
    scheduler, _ = make_queue(plan, ["complete"] * 6)
    for _ in range(6):
        scheduler.tick()
    final_receipt = Path(plan["stages"][-1]["receipt"])
    receipt = json.loads(final_receipt.read_text(encoding="utf-8"))
    Path(receipt["artifact_paths"]["submission_zip"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact digest mismatch: submission_zip"):
        scheduler.tick()


def test_running_owner_without_result_is_never_relaunched(prepared) -> None:
    plan, _, _ = prepared
    scheduler, launcher = make_queue(plan, [])
    state = scheduler.load_state()
    state["running"] = {
        "stage_id": "ROTATION_0",
        "pid": 999,
        "create_time": 1.0,
        "result_path": str(Path(plan["run"]) / "missing.json"),
    }
    scheduler.save_state(state)
    with pytest.raises(ValueError, match="without a durable result"):
        scheduler.tick()
    assert launcher.commands == []


def test_launch_intent_is_durable_before_process_creation(prepared) -> None:
    plan, _, _ = prepared
    observed = {"durable": False}
    launcher = Launcher(plan, ["complete"])

    def factory(command: list[str], **kwargs: object) -> Child:
        del kwargs
        state = queue.read_json_shared(Path(plan["state_path"]))
        observed["durable"] = bool(
            state["status"] == "LAUNCHING"
            and state["running"]["intent_nonce"] == command[command.index("-IntentNonce") + 1]
        )
        return launcher(command)

    scheduler = queue.ScientificOOFQueue(
        plan,
        monitor=Monitor(),  # type: ignore[arg-type]
        popen_factory=factory,
        sleeper=lambda _: None,
    )
    scheduler.tick()
    assert observed["durable"] is True


def test_ambiguous_launch_without_handshake_never_relaunches(prepared) -> None:
    plan, _, _ = prepared
    scheduler, launcher = make_queue(plan, [])
    state = scheduler.load_state()
    state["running"] = {
        "stage_id": "ROTATION_0",
        "pid": None,
        "create_time": None,
        "intent_nonce": "n",
        "intent_path": str(Path(plan["run"]) / "absent.intent.json"),
        "result_path": str(Path(plan["run"]) / "absent.result.json"),
    }
    scheduler.save_state(state)
    with pytest.raises(ValueError, match="Expired launch intent"):
        scheduler.tick()
    assert launcher.commands == []


def test_plan_and_config_drift_are_rejected(prepared) -> None:
    plan, _, config = prepared
    write(config, {"checkpoint_updates": 99})
    with pytest.raises(ValueError, match="changed"):
        queue.validate_plan(plan)


def test_exact_two_gib_resource_threshold() -> None:
    values = {
        "snapshot_valid": True,
        "run_root_binding_valid": True,
        "process_coverage_bounded": True,
        "publisher_scan_error": None,
        "project_tree_rss_upper_bound_bytes": 1,
        "host_available_bytes": 2 * queue.GIB,
        "windows_commit_headroom_bytes": 2 * queue.GIB,
        "disk_free_after_reservation_bytes": 10_000_000_000,
    }
    assert queue._resource_ready(False, values)
    values["windows_commit_headroom_bytes"] -= 1
    assert not queue._resource_ready(True, values)
