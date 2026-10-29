"""Failure-retention contract for the CPU-only optimized migration command."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_script():
    path = Path(__file__).parents[2] / "scripts/migrate_detection_campaign_optimized.py"
    spec = importlib.util.spec_from_file_location(
        "migrate_detection_campaign_optimized", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_boundary_is_retained_without_cuda_or_process_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script()
    old_run = tmp_path / "old"
    checkpoint = old_run / "checkpoints/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not-the-pinned-checkpoint")
    provenance = {"runtime_preregistration_sha256": "1" * 64}
    monkeypatch.setattr(
        module,
        "_preregistration_contract",
        lambda *_args: (
            {},
            {},
            provenance,
            {"runtime_preregistration_sha256": "2" * 64},
            {"status": "PASS_TEST_CONTRACT"},
        ),
    )
    monkeypatch.setattr(
        module,
        "verify_source_state_receipt",
        lambda _path, *, source_run: {"status": "PASS_TEST_STOPPED"},
    )
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="explicitly pinned"):
        module.build_migration_bundle(
            old_run=old_run,
            target_preregistration_dir=tmp_path / "target",
            output=output,
            expected_last_sha256="0" * 64,
            expected_selected_sha256=None,
            performance_config_relative="configs/performance.json",
            source_state_receipt=tmp_path / "stopped.json",
        )
    failure = json.loads((output / "FAILURE.json").read_text(encoding="utf-8"))
    assert failure == {
        "schema": "eap_detection_campaign_checkpoint_migration_failure_v1",
        "status": "FAIL_RETAINED",
        "error_type": "ValueError",
        "error": "Source checkpoint differs from the explicitly pinned SHA-256",
        "producer_was_stopped": False,
        "cuda_or_model_constructed": False,
    }
    assert checkpoint.read_bytes() == b"not-the-pinned-checkpoint"
    assert not (output / "migration_receipt.json").exists()


def test_parent_summary_preserves_chain_and_migrated_selected_sibling(
    tmp_path: Path,
) -> None:
    module = load_script()
    old_run = tmp_path / "old_run"
    old_summary = {
        "status": "BUDGET_STOPPED",
        "final_update": 1024,
        "selected_checkpoint": {
            "status": "AVAILABLE",
            "path": "old/selected.pt",
            "sha256": "1" * 64,
            "best_metric": 0.4,
        },
        "parent_run_chain": [
            {
                "summary_path": "ancestor/summary.json",
                "summary_sha256": "2" * 64,
                "status": "BUDGET_STOPPED",
                "final_update": 512,
                "selected_checkpoint": None,
            }
        ],
    }
    module._write_once(old_run / "summary.json", old_summary)
    output = tmp_path / "bundle"
    summary = module._migration_parent_summary(
        old_run=old_run,
        output=output,
        boundary={
            "status": "PASS_COMPLETE_EVALUATION_BOUNDARY",
            "selected_update": 1024,
        },
        migrated_last={
            "update": 1024,
            "best_metric": 0.4,
            "destination_full_sha256": "3" * 64,
        },
        migrated_selected={
            "update": 1024,
            "best_metric": 0.4,
            "destination_full_sha256": "4" * 64,
        },
        old_preregistration_sha256="5" * 64,
        old_source_receipt_sha256="7" * 64,
        source_state={"receipt_sha256": "6" * 64},
        recovery_boundary_sha256="8" * 64,
    )
    assert len(summary["parent_run_chain"]) == 2
    appended = summary["parent_run_chain"][-1]
    assert appended["summary_sha256"] == module.sha256_file(old_run / "summary.json")
    assert appended["selected_checkpoint"] == old_summary["selected_checkpoint"]
    selected = summary["selected_checkpoint"]
    assert Path(selected["path"]) == (output / "migrated/selected.pt").resolve()
    assert selected["sha256"] == "4" * 64
    assert summary["migration_lineage"]["migrated_last_sha256"] == "3" * 64
    assert summary["migration_lineage"]["source_receipt_sha256"] == "7" * 64
    assert summary["migration_lineage"]["recovery_boundary_sha256"] == "8" * 64


def test_intentional_stop_parent_uses_recovery_boundary_without_fake_summary(
    tmp_path: Path,
) -> None:
    module = load_script()
    output = tmp_path / "bundle"
    summary = module._migration_parent_summary(
        old_run=tmp_path / "unfinished_run",
        output=output,
        boundary={"selected_update": 3072},
        migrated_last={
            "update": 3072,
            "best_metric": 0.6,
            "destination_full_sha256": "1" * 64,
        },
        migrated_selected={
            "update": 3072,
            "best_metric": 0.6,
            "destination_full_sha256": "2" * 64,
        },
        old_preregistration_sha256="3" * 64,
        old_source_receipt_sha256="4" * 64,
        source_state={
            "receipt_sha256": "5" * 64,
            "source_stop_status": "INTENTIONALLY_STOPPED_FOR_PERFORMANCE_ADMISSION",
        },
        recovery_boundary_sha256="6" * 64,
        unfinished_chain={
            "preregistration_sha256": "7" * 64,
            "parent_run_chain": [
                {
                    "summary_path": "r0_1024/summary.json",
                    "summary_sha256": "8" * 64,
                    "status": "BUDGET_STOPPED",
                    "final_update": 1024,
                    "selected_checkpoint": {"sha256": "9" * 64},
                }
            ],
        },
    )
    assert summary["parent_run_chain"] == [
        {
            "summary_path": "r0_1024/summary.json",
            "summary_sha256": "8" * 64,
            "status": "BUDGET_STOPPED",
            "final_update": 1024,
            "selected_checkpoint": {"sha256": "9" * 64},
        },
        {
            "summary_path": None,
            "summary_sha256": None,
            "status": "RECOVERY_BOUNDARY_FROM_INTENTIONAL_STOP",
            "final_update": 3072,
            "selected_checkpoint": None,
        }
    ]
    assert summary["migration_lineage"]["source_summary_path"] is None
    assert summary["migration_lineage"]["unfinished_preregistration_sha256"] == (
        "7" * 64
    )


def test_unfinished_preregistration_preserves_real_nonempty_parent_shape(
    tmp_path: Path,
) -> None:
    module = load_script()
    ancestor = tmp_path / "r0_1024"
    selected_path = ancestor / "checkpoints/selected.pt"
    selected_path.parent.mkdir(parents=True)
    selected_path.write_bytes(b"selected-1024")
    selected = {
        "status": "AVAILABLE",
        "path": str(selected_path),
        "sha256": module.sha256_file(selected_path),
        "best_metric": 0.043,
    }
    ancestor_summary = ancestor / "summary.json"
    module._write_once(
        ancestor_summary,
        {
            "status": "BUDGET_STOPPED",
            "final_update": 1024,
            "selected_checkpoint": selected,
        },
    )
    entry = {
        "summary_path": str(ancestor_summary),
        "summary_sha256": module.sha256_file(ancestor_summary),
        "status": "BUDGET_STOPPED",
        "final_update": 1024,
        "selected_checkpoint": selected,
    }
    old_run = tmp_path / "r0_8192"
    config = tmp_path / "r0.yaml"
    config.write_text("experiment: r0\n", encoding="utf-8")
    provenance = {"runtime_preregistration_sha256": "a" * 64}
    module._write_once(
        old_run / "preregistration.json",
        {
            "schema": "eap_detection_campaign_run_v1",
            "status": "RUNNING",
            "run_name": "r0_8192",
            "config": str(config),
            "config_sha256": "b" * 64,
            "evaluation_protocol_sha256": "c" * 64,
            "runtime_provenance": provenance,
            "start_update": 1024,
            "parent_run_chain": [entry],
        },
    )
    result = module._verified_unfinished_parent_chain(
        old_run=old_run,
        source_state={"config_path": str(config)},
        old_runtime_prereg={"config_sha256": "b" * 64, "protocol_sha256": "c" * 64},
        old_provenance=provenance,
        checkpoint_update=3072,
    )
    assert result["start_update"] == 1024
    assert result["parent_run_chain"] == [entry]
    roots = module._evaluation_roots_for_resumed_boundary(
        old_run=old_run,
        scheduled_updates=[1024, 2048, 3072, 4096],
        checkpoint_update=3072,
        unfinished_chain=result,
    )
    assert roots == {1024: ancestor, 2048: old_run, 3072: old_run}
    selected_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="selected checkpoint bytes"):
        module._verified_unfinished_parent_chain(
            old_run=old_run,
            source_state={"config_path": str(config)},
            old_runtime_prereg={
                "config_sha256": "b" * 64,
                "protocol_sha256": "c" * 64,
            },
            old_provenance=provenance,
            checkpoint_update=3072,
        )
