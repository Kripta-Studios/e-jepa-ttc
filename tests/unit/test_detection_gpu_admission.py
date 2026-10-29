from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from e_jepa_ttc.detection.campaign_training import load_campaign_config, sha256_file
from e_jepa_ttc.detection.gpu_admission_harness import (
    ARM_NAMES,
    REQUIRED_BINDING_ROLES,
    SCHEMA,
    Tolerances,
    bind_plan,
    load_plan,
    validate_checkpoint_lineage,
    validate_initial_state,
    verify_metrics,
)

ROOT = Path(__file__).parents[2]


def _checkpoint(
    path: Path,
    model_value: float = 1.0,
    *,
    config_sha256: str = "1" * 64,
    protocol_sha256: str = "2" * 64,
    preregistration_sha256: str = "3" * 64,
) -> None:
    torch.save(
        {
            "model": {"weight": torch.tensor([model_value])},
            "optimizer": {"state": {}, "param_groups": []},
            "scheduler": {"last_update": 100},
            "sampler": {"epoch": 0, "cursor": 400},
            "rng": {"torch_cpu": torch.arange(4, dtype=torch.uint8)},
            "progress": {"training_seconds": 1.0},
            "data_member_sha256": {"token": "a" * 64},
            "update": 100,
            "best_metric": 0.2,
            "config_sha256": config_sha256,
            "protocol_sha256": protocol_sha256,
            "runtime_provenance": {
                "runtime_preregistration_sha256": preregistration_sha256
            },
        },
        path,
    )


def _plan(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/detection_campaign/rgb_r0_d4096_seed7.yaml"
    config = load_campaign_config(config_path)
    manifest_path = Path(config.data.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old_producer = tmp_path / "old_runner.py"
    optimized_producer = tmp_path / "optimized_runner.py"
    performance = tmp_path / "performance.json"
    media_admission = tmp_path / "media_admission.json"
    for path in (old_producer, optimized_producer, performance, media_admission):
        path.write_text("{}\n", encoding="utf-8")
    old_prereg = tmp_path / "old_prereg.json"
    optimized_prereg = tmp_path / "optimized_prereg.json"
    for path, producer in (
        (old_prereg, old_producer),
        (optimized_prereg, optimized_producer),
    ):
        path.write_text(
            json.dumps(
                {
                    "config_sha256": config.sha256,
                    "protocol_sha256": "2" * 64,
                    "dataset_media_sha256": manifest["source_media_sha256"],
                    "code_sha256": {f"scripts/{producer.name}": sha256_file(producer)},
                }
            ),
            encoding="utf-8",
        )
    checkpoints: dict[str, tuple[Path, Path]] = {}
    for role, prereg in (("old", old_prereg), ("optimized", optimized_prereg)):
        directory = tmp_path / f"{role}_checkpoints"
        directory.mkdir()
        resume = directory / "last.pt"
        selected = directory / "selected.pt"
        for path in (resume, selected):
            _checkpoint(
                path,
                config_sha256=config.sha256,
                preregistration_sha256=sha256_file(prereg),
            )
        checkpoints[role] = resume, selected
    arms: dict[str, Any] = {}
    for name in ARM_NAMES:
        role = "optimized" if name == "optimized" else "old"
        resume, selected = checkpoints[role]
        output = tmp_path / name
        output.mkdir()
        metrics = output / "metrics.jsonl"
        rows = []
        for index in range(16):
            loss = 1.0 - index * 0.01
            if name == "old_B":
                loss += index * 1e-6
            elif name == "optimized":
                loss += index * 2e-6
            rows.append(
                {
                    "update": 101 + index,
                    "sample_tokens": [f"token-{index}"],
                    "learning_rates": [0.001],
                    "loss": loss,
                    "losses": {"loss_center": loss / 2, "loss_depth": loss / 3},
                    "gradient_norm_before_clip": 2.0 + index * 1e-6,
                    "preprocessing_seconds": 2.0 if name != "optimized" else 1.0,
                }
            )
        metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        summary = output / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "schema": "eap_detection_campaign_run_v1",
                    "status": "BUDGET_STOPPED",
                    "final_update": 116,
                    "external_execution_stop_requested": 116,
                    "stop_reason": "external_execution_stop_after_completed_update",
                    "config_sha256": config.sha256,
                    "evaluation_protocol_sha256": "2" * 64,
                    "runtime_provenance": {
                        "runtime_preregistration_sha256": sha256_file(
                            optimized_prereg if name == "optimized" else old_prereg
                        )
                    },
                    "preprocessing_seconds_median": 2.0,
                    "step_seconds_median": 1.0,
                    "stage_training_seconds": 3.0,
                    "cumulative_training_seconds": 10.0,
                    "peak_allocated_bytes": 10,
                    "peak_reserved_bytes": 20,
                    "performance_metrics": None,
                }
            ),
            encoding="utf-8",
        )
        evaluation = output / "evaluation.json"
        evaluation.write_text(
            json.dumps(
                {
                    "metrics_v2": {
                        "schema": "phase_ttc_detection_evaluation_result_v2",
                        "protocol_sha256": "2" * 64,
                        "official_eap_score": False,
                        "sample_count": 16,
                        "inference_seconds_median": 0.1,
                        "inference_seconds_total": 1.6,
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        arms[name] = {
            "name": name,
            "resume_checkpoint": {"path": str(resume), "sha256": sha256_file(resume)},
            "selected_checkpoint": {"path": str(selected), "sha256": sha256_file(selected)},
            "command": [
                sys.executable,
                str(optimized_producer if name == "optimized" else old_producer),
                "--config",
                str(config_path),
                "--resume",
                str(resume),
                "--output",
                str(output),
                "--stop-after-update",
                "116",
                *(
                    ["--performance-config", str(performance)]
                    if name == "optimized"
                    else []
                ),
            ],
            "start_update": 100,
            "stop_after_update": 116,
            "output": str(output),
            "metrics_path": str(metrics),
            "summary_path": str(summary),
            "evaluation_path": str(evaluation),
        }
    return bind_plan(
        {
            "schema": SCHEMA,
            "arm_order": list(ARM_NAMES),
            "updates": 16,
            "batch_size": 1,
            "candidate_policy": "ENGINEERING_NON_CANDIDATE_DISCARD_OUTPUTS",
            "graphics_baseline": [],
            "config_sha256": config.sha256,
            "protocol_sha256": "2" * 64,
            "dataset_manifest_sha256": config.data.expected_manifest_sha256,
            "execution_receipt_path": str(tmp_path / "execution.json"),
            "resources": {
                "minimum_available_physical_ram_bytes": 1,
                "minimum_free_vram_bytes": 1,
            },
            "tolerances": Tolerances().__dict__,
            "bindings": [
                {
                    "role": role,
                    "path": str(
                        {
                            "config": config_path,
                            "dataset_manifest": manifest_path,
                            "media_admission": media_admission,
                            "old_runtime_preregistration": old_prereg,
                            "old_producer": old_producer,
                            "optimized_runtime_preregistration": optimized_prereg,
                            "optimized_producer": optimized_producer,
                            "performance_config": performance,
                            "python_executable": Path(sys.executable),
                        }[role]
                    ),
                    "sha256": sha256_file(
                        {
                            "config": config_path,
                            "dataset_manifest": manifest_path,
                            "media_admission": media_admission,
                            "old_runtime_preregistration": old_prereg,
                            "old_producer": old_producer,
                            "optimized_runtime_preregistration": optimized_prereg,
                            "optimized_producer": optimized_producer,
                            "performance_config": performance,
                            "python_executable": Path(sys.executable),
                        }[role]
                    ),
                }
                for role in sorted(REQUIRED_BINDING_ROLES)
            ],
            "arms": arms,
        }
    )


def _write_rebound(plan: dict[str, Any], path: Path) -> None:
    raw = dict(plan)
    raw.pop("plan_sha256")
    path.write_text(json.dumps(bind_plan(raw)), encoding="utf-8")


def test_plan_and_initial_state_are_exact(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    loaded = load_plan(path)
    validate_checkpoint_lineage(loaded)
    assert validate_initial_state(loaded)["resume_components"]


def test_initial_state_tamper_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    changed = tmp_path / "changed.pt"
    _checkpoint(changed, 2.0)
    plan["arms"]["optimized"]["resume_checkpoint"] = {
        "path": str(changed),
        "sha256": sha256_file(changed),
    }
    with pytest.raises(ValueError, match="exact initial continuation state"):
        validate_initial_state(plan)


def test_metrics_admit_repeat_bounded_drift_and_consistent_speedup(tmp_path: Path) -> None:
    result = verify_metrics(_plan(tmp_path))
    assert result["status"] == "PASS_ENGINEERING_ADMISSION_ONLY"
    assert result["candidate_decision"] is None
    assert result["extrapolation_to_8192_updates"] is None


def test_first_loss_has_separate_strict_ceiling(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    metrics = Path(plan["arms"]["optimized"]["metrics_path"])
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    rows[0]["loss"] += 2e-6
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="loss drift"):
        verify_metrics(plan)


def test_sample_order_drift_is_rejected_before_performance(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    metrics = Path(plan["arms"]["optimized"]["metrics_path"])
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    rows[3]["sample_tokens"] = ["different"]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Sample order differs"):
        verify_metrics(plan)


def test_resource_admission_rejects_competing_compute_without_gpu_calls(
    monkeypatch: Any, tmp_path: Path
) -> None:
    script = Path(__file__).parents[2] / "scripts/detection_gpu_admission.py"
    spec = importlib.util.spec_from_file_location("gpu_admission_script_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "_gpu_processes",
        lambda: [{"pid": 999, "type": "C", "name": "python.exe", "used_memory": "N/A"}],
    )
    monkeypatch.setattr(module, "_process_snapshot", lambda: {999: (1, 10)})
    monkeypatch.setattr(module, "_cim_process_snapshot", lambda: {999: (1, 10)})
    monkeypatch.setattr(module, "_verify_graphics_baseline_live", lambda plan: None)
    monkeypatch.setattr(module, "_nvidia_query", lambda arguments: ["12000"])
    monkeypatch.setattr(module, "_physical_available_bytes", lambda: 16_000_000_000)
    with pytest.raises(RuntimeError, match="Unapproved GPU"):
        module.resource_admission(_plan(tmp_path))


def test_wddm_baseline_and_launcher_descendant_are_distinguished(
    monkeypatch: Any, tmp_path: Path
) -> None:
    script = Path(__file__).parents[2] / "scripts/detection_gpu_admission.py"
    spec = importlib.util.spec_from_file_location("gpu_identity_script_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = _plan(tmp_path)
    plan["graphics_baseline"] = [
        {
            "pid": 100,
            "creation_time_us": 10,
            "type": "C+G",
            "name": "explorer.exe",
            "used_memory": "N/A",
        }
    ]
    monkeypatch.setattr(
        module,
        "_gpu_processes",
        lambda: [
            {"pid": 100, "type": "C+G", "name": "explorer.exe", "used_memory": "N/A"},
            {"pid": 300, "type": "C", "name": "python.exe", "used_memory": "N/A"},
        ],
    )
    monkeypatch.setattr(
        module,
        "_process_snapshot",
        lambda: {100: (1, 10), 200: (1, 20), 300: (200, 30)},
    )
    monkeypatch.setattr(
        module,
        "_cim_process_snapshot",
        lambda: {100: (1, 10), 200: (1, 20), 300: (200, 30)},
    )
    assert module._gpu_competitors(plan, {200, 300}) == []
    assert [row["pid"] for row in module._gpu_competitors(plan, set())] == [300]
    assert module._descendant_pids(200, 20) == {200, 300}


def test_new_c_plus_g_process_is_not_auto_whitelisted(monkeypatch: Any, tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "scripts/detection_gpu_admission.py"
    spec = importlib.util.spec_from_file_location("gpu_new_graphics_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = _plan(tmp_path)
    plan["graphics_baseline"] = []
    monkeypatch.setattr(
        module,
        "_gpu_processes",
        lambda: [{"pid": 101, "type": "C+G", "name": "new.exe", "used_memory": "N/A"}],
    )
    monkeypatch.setattr(module, "_process_snapshot", lambda: {101: (1, 11)})
    monkeypatch.setattr(module, "_cim_process_snapshot", lambda: {101: (1, 11)})
    assert [row["pid"] for row in module._gpu_competitors(plan, set())] == [101]


def test_protected_reused_c_plus_g_pid_uses_cim_fallback(
    monkeypatch: Any, tmp_path: Path
) -> None:
    script = Path(__file__).parents[2] / "scripts/detection_gpu_admission.py"
    spec = importlib.util.spec_from_file_location("gpu_reused_graphics_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = _plan(tmp_path)
    plan["graphics_baseline"] = [
        {
            "pid": 100,
            "creation_time_us": 10,
            "type": "C+G",
            "name": "protected.exe",
            "used_memory": "N/A",
        }
    ]
    monkeypatch.setattr(
        module,
        "_gpu_processes",
        lambda: [
            {"pid": 100, "type": "C+G", "name": "replacement.exe", "used_memory": "N/A"}
        ],
    )
    monkeypatch.setattr(module, "_process_snapshot", lambda: {})
    monkeypatch.setattr(module, "_cim_process_snapshot", lambda: {100: (1, 11)})
    assert [row["pid"] for row in module._gpu_competitors(plan, set())] == [100]


def test_duplicate_binding_and_command_drift_are_rejected(tmp_path: Path) -> None:
    duplicated = _plan(tmp_path / "duplicate")
    duplicated["bindings"].append(dict(duplicated["bindings"][0]))
    path = tmp_path / "duplicate-plan.json"
    _write_rebound(duplicated, path)
    with pytest.raises(ValueError, match="binding roles"):
        load_plan(path)

    drift_root = tmp_path / "drift"
    drift_root.mkdir()
    drifted = _plan(drift_root)
    drifted["arms"]["old_A"]["command"][1] = str(drift_root / "wrong.py")
    drift_path = tmp_path / "drift-plan.json"
    _write_rebound(drifted, drift_path)
    with pytest.raises(ValueError, match="producer differs"):
        load_plan(drift_path)


def test_stale_metric_path_and_noncontiguous_updates_are_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    plan["arms"]["old_A"]["metrics_path"] = str(tmp_path / "stale.jsonl")
    path = tmp_path / "stale-plan.json"
    _write_rebound(plan, path)
    with pytest.raises(ValueError, match="escapes its new arm output"):
        load_plan(path)

    valid = _plan(tmp_path / "updates")
    metrics = Path(valid["arms"]["old_A"]["metrics_path"])
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    rows[2]["update"] = 999
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous admission interval"):
        verify_metrics(valid)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_resource_thresholds_must_be_positive_integers(tmp_path: Path, invalid: object) -> None:
    plan = _plan(tmp_path)
    plan["resources"]["minimum_free_vram_bytes"] = invalid
    path = tmp_path / "resource-plan.json"
    _write_rebound(plan, path)
    with pytest.raises(ValueError, match="positive integer"):
        load_plan(path)


def test_empty_evaluation_artifact_is_rejected(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    Path(plan["arms"]["optimized"]["evaluation_path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation schema/protocol"):
        verify_metrics(plan)


def test_resume_checkpoint_update_must_equal_preregistered_start(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    resume = Path(plan["arms"]["optimized"]["resume_checkpoint"]["path"])
    payload = torch.load(resume, map_location="cpu", weights_only=True)
    payload["update"] = 99
    torch.save(payload, resume)
    plan["arms"]["optimized"]["resume_checkpoint"]["sha256"] = sha256_file(resume)
    with pytest.raises(ValueError, match="start_update"):
        validate_checkpoint_lineage(plan)
