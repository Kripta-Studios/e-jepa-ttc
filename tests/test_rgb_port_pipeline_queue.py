from __future__ import annotations

from pathlib import Path

import pytest

from operational.rgb_port.accounting import atomic_write_json, read_json_shared, sha256_file
from operational.rgb_port.run import _module_matches
from operational.rgb_port_pipeline_v2 import contracts
from operational.rgb_port_pipeline_v2.contracts import (
    CANARY,
    EVENT_FIT_IDS,
    build_pipeline_freeze,
    validate_pipeline_freeze,
)
from operational.rgb_port_pipeline_v2.queue import (
    ACCELERATED_MODULE,
    PIPELINE_MODULE,
    _augmented_config,
    _route_command,
    _runtime_members,
)
from operational.rgb_port_pipeline_v2.receipts import (
    validate_fit_lineage,
    write_report_extension,
)


def test_composed_route_wraps_only_exact_event_producers(tmp_path: Path) -> None:
    freeze = tmp_path / "PIPELINE_FREEZE.json"
    freeze.write_text("{}", encoding="utf-8")

    def accelerated(command: list[object], _repository: Path) -> list[str]:
        fit = str(command[command.index("--fit-id") + 1])
        return [
            "python",
            "-m",
            ACCELERATED_MODULE,
            "--acceleration-freeze",
            "V1.json",
            "--",
            "--fit-id",
            fit,
            "--device",
            "cuda",
        ]

    for fit_id in EVENT_FIT_IDS:
        routed = _route_command(accelerated, freeze, ["--fit-id", fit_id], tmp_path)
        assert routed[:3] == ["python", "-m", PIPELINE_MODULE]
        assert routed[3:6] == ["--pipeline-freeze", str(freeze.resolve()), "--"]
        assert routed[6:9] == ["--acceleration-freeze", "V1.json", "--"]
        assert routed[-4:] == ["--fit-id", fit_id, "--device", "cuda"]
    for fit_id in ("R_A5", "R_C2F"):
        resolved = accelerated(["--fit-id", fit_id], tmp_path)
        assert _route_command(accelerated, freeze, ["--fit-id", fit_id], tmp_path) == resolved

    def substring(_command: list[object], _repository: Path) -> list[str]:
        return ["python", "-m", f"client.{ACCELERATED_MODULE}", "--fit-id", EVENT_FIT_IDS[0]]

    assert _route_command(substring, freeze, [], tmp_path)[2] != PIPELINE_MODULE


def test_config_extension_adds_exact_marker_and_dynamic_bundle(tmp_path: Path) -> None:
    run = tmp_path / "run"
    receipts = run / f"fits/{EVENT_FIT_IDS[0]}/pipeline_checkpoints"
    receipts.mkdir(parents=True)
    atomic_write_json(receipts / "checkpoint_009000.json", {"sealed": True})
    config = {
        "run_root": str(run),
        "resources": {"heavy_command_markers": [], "project_command_markers": []},
        "package": {"members": []},
    }
    value = _augmented_config(config)
    assert PIPELINE_MODULE in value["resources"]["heavy_command_markers"]
    assert PIPELINE_MODULE in value["resources"]["project_command_markers"]
    assert "repo-tree:operational/rgb_port_pipeline_v2" in value["package"]["members"]
    assert (
        f"fits/{EVENT_FIT_IDS[0]}/pipeline_checkpoints/checkpoint_009000.json"
        in value["package"]["members"]
    )
    assert config["resources"]["heavy_command_markers"] == []
    assert _module_matches(["python", "-m", PIPELINE_MODULE], [PIPELINE_MODULE])
    assert not _module_matches(["pytest", PIPELINE_MODULE], [PIPELINE_MODULE])


def _lineage_fixture(tmp_path: Path, *, completed: int) -> tuple[Path, Path, dict]:
    run = tmp_path / "run"
    fit_id = EVENT_FIT_IDS[0]
    fit = run / "fits" / fit_id
    versions = fit / "checkpoint_versions"
    versions.mkdir(parents=True)
    checkpoint = versions / f"checkpoint_{completed:06d}.pt"
    checkpoint.write_bytes(f"checkpoint-{completed}".encode())
    identity = "fit-identity"
    atomic_write_json(
        fit / "CHECKPOINT_POINTER.json",
        {
            "completed_updates": completed,
            "version": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "identity_sha256": identity,
        },
    )
    source = tmp_path / "pipeline.py"
    source.write_bytes(b"pipeline")
    source_freeze = tmp_path / "SOURCE_FREEZE.json"
    source_freeze.write_bytes(b"source-v2")
    acceleration = tmp_path / "ACCELERATION_FREEZE.json"
    acceleration.write_bytes(b"acceleration-v1")
    baseline = tmp_path / "V1_RATE_BASELINE.json"
    baseline.write_bytes(b"baseline")
    origin_checkpoint = tmp_path / "origin.pt"
    origin_checkpoint.write_bytes(b"origin")
    admission = tmp_path / "admission.json"
    atomic_write_json(admission, {"recovery_failures": 0, "resource_failures": 0})
    freeze_path = run / "PIPELINE_FREEZE.json"
    freeze = {
        "schema": "rgb_port_pipeline_freeze_v2",
        "status": "FROZEN",
        "identity_sha256": "pipeline-identity",
        "event_fit_ids": list(EVENT_FIT_IDS),
        "scope": "HOST_PREFETCH_EVENT_ONLY",
        "original_source_freeze_path": str(source_freeze),
        "original_source_freeze_sha256": sha256_file(source_freeze),
        "original_acceleration_freeze_path": str(acceleration),
        "original_acceleration_freeze_sha256": sha256_file(acceleration),
        "original_acceleration_freeze_identity_sha256": "acceleration-identity",
        "source_sha256": {str(source): sha256_file(source)},
        "orchestration_source_sha256": {str(source): sha256_file(source)},
        "admissions": {
            "cpu": {"path": str(admission), "sha256": sha256_file(admission)},
            "gpu": {"path": str(admission), "sha256": sha256_file(admission)},
        },
        "performance_baseline": {"path": str(baseline), "sha256": sha256_file(baseline)},
        "origins": {
            fit_id: {
                "fit_id": fit_id,
                "mode": "PAUSED_FULL_CHECKPOINT",
                "completed_updates": 8600,
                "checkpoint_path": str(origin_checkpoint),
                "checkpoint_sha256": sha256_file(origin_checkpoint),
                "identity_sha256": identity,
            },
            EVENT_FIT_IDS[1]: {
                "fit_id": EVENT_FIT_IDS[1],
                "mode": "FRESH_UNSTARTED",
                "completed_updates": 0,
            },
        },
        "canary": CANARY,
    }
    atomic_write_json(freeze_path, freeze)
    return run, freeze_path, freeze


def test_native_origin_allowed_but_progress_requires_typed_receipt(tmp_path: Path) -> None:
    run, freeze_path, freeze = _lineage_fixture(tmp_path / "origin", completed=8600)
    fit_id = EVENT_FIT_IDS[0]
    origin = freeze["origins"][fit_id]
    pointer = read_json_shared(run / f"fits/{fit_id}/CHECKPOINT_POINTER.json")
    origin_path = Path(origin["checkpoint_path"])
    origin_path.write_bytes(
        (run / f"fits/{fit_id}/checkpoint_versions" / pointer["version"]).read_bytes()
    )
    origin["checkpoint_sha256"] = sha256_file(origin_path)
    atomic_write_json(freeze_path, freeze)
    validate_fit_lineage(run, freeze, freeze_path, fit_id)

    advanced, advanced_freeze_path, advanced_freeze = _lineage_fixture(
        tmp_path / "advanced", completed=8700
    )
    with pytest.raises(FileNotFoundError, match="runtime or pending"):
        validate_fit_lineage(advanced, advanced_freeze, advanced_freeze_path, fit_id)


def test_report_extension_records_real_wrapper_resume_command(tmp_path: Path) -> None:
    run, freeze_path, freeze = _lineage_fixture(tmp_path, completed=8600)
    path = write_report_extension(run, freeze, freeze_path)
    value = read_json_shared(path)
    assert value["pipeline_freeze_sha256"] == freeze["identity_sha256"]
    assert PIPELINE_MODULE.rsplit(".", 1)[0] + ".queue resume" in value["resume_command"]
    assert set(value["event_fits"]) == set(EVENT_FIT_IDS)


def test_typed_runtime_timing_and_checkpoint_receipt_bind_progress(tmp_path: Path) -> None:
    run, freeze_path, freeze = _lineage_fixture(tmp_path, completed=8900)
    fit_id = EVENT_FIT_IDS[0]
    fit = run / "fits" / fit_id
    pointer = read_json_shared(fit / "CHECKPOINT_POINTER.json")
    freeze_file_sha = sha256_file(freeze_path)
    common = {
        "fit_id": fit_id,
        "pipeline_freeze_sha256": freeze["identity_sha256"],
        "pipeline_freeze_file_sha256": freeze_file_sha,
        "original_acceleration_freeze_sha256": freeze["original_acceleration_freeze_sha256"],
        "original_acceleration_freeze_identity_sha256": freeze[
            "original_acceleration_freeze_identity_sha256"
        ],
        "source_sha256": freeze["source_sha256"],
    }
    atomic_write_json(
        fit / "PIPELINE_RUNTIME.json",
        {
            "schema": "rgb_port_pipeline_runtime_v2",
            "status": "ACTIVE",
            **common,
            "mode": "CANARY",
            "optimizer_updates": 0,
            "completed_updates": 8900,
            "last_saved_update": 8900,
            "checkpoint_sha256": pointer["checkpoint_sha256"],
            "last_checkpoint_sha256": pointer["checkpoint_sha256"],
        },
    )
    receipts = fit / "pipeline_checkpoints"
    timings = receipts / "timings_008900.json"
    atomic_write_json(
        timings,
        {
            "schema": "rgb_port_pipeline_timings_v2",
            "status": "SNAPSHOT",
            **common,
            "mode": "DEPTH1_EVENT_PREFETCH",
            "completed_updates": 8900,
            "checkpoint_sha256": pointer["checkpoint_sha256"],
            "total_seconds": 180.0,
            "canary_marks": [
                {"completed_updates": 8700, "science_seconds": 60.0},
                {"completed_updates": 8800, "science_seconds": 120.0},
                {"completed_updates": 8900, "science_seconds": 180.0},
            ],
            "timings": {
                "submitted": 12,
                "consumed": 12,
                "cancelled": 0,
                "errors": 0,
                "order_mismatches": 0,
                "outstanding_max": 1,
                "outstanding_current": 0,
            },
        },
    )
    receipt = receipts / "checkpoint_008900.json"
    atomic_write_json(
        receipt,
        {
            "schema": "rgb_port_pipeline_checkpoint_receipt_v2",
            "status": "COMPLETE",
            **common,
            "mode": "DEPTH1_EVENT_PREFETCH",
            "completed_updates": 8900,
            "checkpoint_version": pointer["version"],
            "checkpoint_sha256": pointer["checkpoint_sha256"],
            "checkpoint_identity_sha256": pointer["identity_sha256"],
            "timings_snapshot": timings.name,
            "timings_sha256": sha256_file(timings),
        },
    )
    atomic_write_json(
        fit / "PIPELINE_CANARY.json",
        {
            "schema": "rgb_port_pipeline_canary_v2",
            "status": "PASSED",
            **common,
            "mode": "DEPTH1_EVENT_PREFETCH",
            "origin_update": 8600,
            "boundary_update": 8900,
            "scheduled_updates": 300,
            "observed_seconds": 180.0,
            "observed_updates_per_minute": 100.0,
            "subwindow_updates_per_minute": [100.0, 100.0, 100.0],
            "subwindow_median_updates_per_minute": 100.0,
            "subwindow_minimum_updates_per_minute": 100.0,
            "subwindow_floor_updates_per_minute": CANARY["minimum_subwindow_rate_inclusive"],
            "subwindow_gate_passed": True,
            "minimum_updates_per_minute_exclusive": CANARY["minimum_updates_per_minute_exclusive"],
            "strict_rate_gate_passed": True,
            "integrity_gates": {
                "byte_parity": True,
                "schedule_order_unchanged": True,
                "finite_outputs": True,
                "recovery_failures": 0,
                "resource_failures": 0,
            },
            "admission_gates": {
                "rng_cursor_unchanged": True,
                "deferred_oserror_passed": True,
                "checkpoint_integrity": True,
            },
            "prefetch_counters": {
                "submitted": 12,
                "consumed": 12,
                "cancelled": 0,
                "errors": 0,
                "order_mismatches": 0,
                "outstanding_max": 1,
                "outstanding_current": 0,
            },
            "safety_gates_passed": True,
            "foreground_wait_ratio": 0.55,
            "overlap_fraction": 0.45,
            "foreground_wait_ratio_gate": "DIAGNOSTIC_ONLY_NO_THRESHOLD",
            "checkpoint_sha256": pointer["checkpoint_sha256"],
            "boundary_checkpoint_status": "PAUSED_REQUESTED",
            "journal_baseline": {
                "completed_updates": 8600,
                "durable_updates": 8600,
                "recovery_upper": 0,
                "pending_update_upper": 0,
                "identity_sha256": pointer["identity_sha256"],
            },
            "journal_end": {
                "completed_updates": 8900,
                "durable_updates": 8900,
                "recovery_upper": 0,
                "pending_update_upper": 0,
                "identity_sha256": pointer["identity_sha256"],
            },
            "journal_recovery_delta": 0,
            "journal_state_exact": True,
        },
    )
    validate_fit_lineage(run, freeze, freeze_path, fit_id)
    timing_value = read_json_shared(timings)
    timing_value["completed_updates"] = 8899
    atomic_write_json(timings, timing_value)
    with pytest.raises(RuntimeError, match="evidence changed"):
        validate_fit_lineage(run, freeze, freeze_path, fit_id)


def test_runtime_members_are_stable_for_orphan_adoption(tmp_path: Path) -> None:
    run = tmp_path / "run"
    assert _runtime_members(run) == _runtime_members(run)


def test_builder_seals_dynamic_origin_and_both_parent_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    fit = run / f"fits/{EVENT_FIT_IDS[0]}"
    versions = fit / "checkpoint_versions"
    versions.mkdir(parents=True)
    checkpoint = versions / "checkpoint_008777.pt"
    checkpoint.write_bytes(b"dynamic paused checkpoint")
    digest = sha256_file(checkpoint)
    identity = "fit-identity"
    atomic_write_json(
        fit / "CHECKPOINT_POINTER.json",
        {
            "completed_updates": 8777,
            "version": checkpoint.name,
            "checkpoint_sha256": digest,
            "identity_sha256": identity,
        },
    )
    atomic_write_json(
        fit / "CHECKPOINT_RECEIPT.json",
        {
            "completed_updates": 8777,
            "checkpoint_sha256": digest,
            "identity_sha256": identity,
            "status": "PAUSED_RESOURCE",
        },
    )
    (run / "SOURCE_FREEZE.json").write_bytes(b"source-v2")
    acceleration_path = run / "ACCELERATION_FREEZE.json"
    acceleration_path.write_bytes(b"acceleration-v1")
    acceleration = {"identity_sha256": "acceleration-identity"}
    monkeypatch.setattr(contracts, "validate_acceleration_freeze", lambda _path: acceleration)
    baseline_dir = run / "current_bottleneck_20261009"
    baseline_dir.mkdir()
    atomic_write_json(
        baseline_dir / "V1_RATE_BASELINE.json",
        {
            "schema": "rgb_port_pipeline_performance_baseline_v1",
            "updates": 1000,
            "updates_per_minute": CANARY["baseline_updates_per_minute"],
            "acceleration_freeze_file_sha256": sha256_file(acceleration_path),
        },
    )
    admissions: list[Path] = []
    repository = contracts._repository()
    source_sha256 = {
        str(path.resolve()): sha256_file(path) for path in contracts._executed_paths(repository)
    }
    orchestration_source_sha256 = {
        str(path.resolve()): sha256_file(path)
        for path in contracts._orchestration_paths(repository)
    }
    admission_values: list[dict] = []
    for kind in ("cpu", "gpu"):
        admission = tmp_path / f"{kind}.json"
        admission_value = {
            "schema": f"test_{kind}_admission_v1",
            "status": "PASSED",
            "admission_kind": kind,
            "event_fit_ids": list(EVENT_FIT_IDS),
            "optimizer_updates": 0,
            "original_acceleration_freeze_sha256": sha256_file(acceleration_path),
            "original_acceleration_freeze_identity_sha256": acceleration["identity_sha256"],
            "source_sha256": source_sha256,
            "orchestration_source_sha256": orchestration_source_sha256,
            **(
                {
                    "batch_byte_parity": True,
                    "schedule_order_unchanged": True,
                    "rng_cursor_unchanged": True,
                    "deferred_oserror_passed": True,
                }
                if kind == "cpu"
                else {
                    "finite_outputs": True,
                    "recovery_failures": 0,
                    "resource_failures": 0,
                    "checkpoint_integrity": True,
                    "paused_a5_origin": {
                        "fit_id": EVENT_FIT_IDS[0],
                        "completed_updates": 8777,
                        "checkpoint_sha256": digest,
                        "identity_sha256": identity,
                    },
                }
            ),
        }
        atomic_write_json(
            admission,
            admission_value,
        )
        admissions.append(admission)
        admission_values.append(admission_value)

    cpu_tampered = {**admission_values[0], "source_sha256": {}}
    atomic_write_json(admissions[0], cpu_tampered)
    with pytest.raises(ValueError, match="outside the authorized"):
        build_pipeline_freeze(run, admissions[0], admissions[1])
    atomic_write_json(admissions[0], admission_values[0])

    gpu_tampered = {
        **admission_values[1],
        "paused_a5_origin": {
            **admission_values[1]["paused_a5_origin"],
            "completed_updates": 8776,
        },
    }
    atomic_write_json(admissions[1], gpu_tampered)
    with pytest.raises(ValueError, match="outside the authorized"):
        build_pipeline_freeze(run, admissions[0], admissions[1])
    atomic_write_json(admissions[1], admission_values[1])

    value = build_pipeline_freeze(run, admissions[0], admissions[1])
    assert len(value["source_sha256"]) == 2
    assert len(value["orchestration_source_sha256"]) == 4
    for item in value["admissions"].values():
        sealed = read_json_shared(Path(item["path"]))
        assert sealed["source_sha256"] == value["source_sha256"]
        assert sealed["orchestration_source_sha256"] == value["orchestration_source_sha256"]
    assert value["origins"][EVENT_FIT_IDS[0]]["completed_updates"] == 8777
    assert value["origins"][EVENT_FIT_IDS[1]]["mode"] == "FRESH_UNSTARTED"
    assert sha256_file(Path(value["origins"][EVENT_FIT_IDS[0]]["checkpoint_path"])) == digest
    assert validate_pipeline_freeze(run / "PIPELINE_FREEZE.json") == value
