from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from operational.rgb_port.accounting import (
    OwnerLease,
    UpdateLedger,
    atomic_write_json,
    identity_is_live,
    process_create_time,
    read_json_shared,
    sha256_file,
)
from operational.rgb_port.evaluate import (
    common_cap60_metrics,
    paired_group_bootstrap,
    score_signed_ttc,
)
from operational.rgb_port.package import build_bundle, verify_bundle
from operational.rgb_port.predict_heads import _frozen_heads
from operational.rgb_port.prepare_features import validate_endpoint_freeze
from operational.rgb_port.run import (
    FIT_IDS,
    _artifact_task_complete,
    _freeze_fit_endpoint,
    _load_config,
    _module_matches,
    _snapshot_global_qa,
    _sync_completion_copy,
    _task_limits,
    _terminal_artifact_blocker,
    _validate_config,
    _write_parent_round_freeze,
    _write_round_freeze,
)
from operational.rgb_port.transfer import _validate_endpoint_freeze


def test_atomic_json_and_idempotent_accounting(tmp_path: Path) -> None:
    state = tmp_path / "STATE.json"
    atomic_write_json(state, {"value": 3})
    assert read_json_shared(state) == {"value": 3}
    ledger = UpdateLedger(tmp_path / "ACCOUNTING.json", {"fit": 10})
    ledger.charge(
        event_id="one", fit_id="fit", category="scientific", charged_updates=4, evidence={"x": 1}
    )
    ledger.charge(
        event_id="one", fit_id="fit", category="scientific", charged_updates=4, evidence={"x": 1}
    )
    assert ledger.totals(ledger.initialize())["scientific"] == 4
    with pytest.raises(ValueError, match="redefined"):
        ledger.charge(
            event_id="one",
            fit_id="fit",
            category="scientific",
            charged_updates=5,
            evidence={"x": 1},
        )


def test_owner_lease_replaces_dead_or_reused_pid_identity(tmp_path: Path) -> None:
    path = tmp_path / "OWNER.json"
    lease = OwnerLease(path)
    atomic_write_json(path, {"pid": 2_147_483_647, "create_time": 1.0})
    owner = lease.acquire()
    assert owner["pid"] == os.getpid()
    lease.release()
    created = process_create_time(os.getpid())
    assert created is not None
    atomic_write_json(path, {"pid": os.getpid(), "create_time": created + 60.0})
    replacement = lease.acquire()
    assert replacement["create_time"] == created
    lease.release()


def test_owner_identity_access_denied_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    def denied(pid: int) -> None:
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", denied)
    with pytest.raises(RuntimeError, match="Cannot verify PID"):
        identity_is_live({"pid": 123, "create_time": 1.0})


def test_accounting_admits_prospective_fit_upper_before_optimizer(tmp_path: Path) -> None:
    ledger = UpdateLedger(tmp_path / "ACCOUNTING.json", {"fit": 10})
    projected = ledger.admit_fit_attempt(fit_id="fit", scientific_upper=10, recovery_upper=100)
    assert projected["scientific"] == 10
    assert projected["recovery"] == 100
    with pytest.raises(RuntimeError, match="per-fit"):
        ledger.admit_fit_attempt(fit_id="fit", scientific_upper=11, recovery_upper=0)


def test_missing_bucket_and_failed_row_make_official_mid_inadmissible() -> None:
    result = score_signed_ttc([1.0, 4.0, 8.0], [1.1, 4.1, np.nan], ["a", "a", "a"])
    assert result["formula_admitted"] is False
    assert result["group_macro_bucket_MiD"] is None
    assert result["num_failed_predictions"] == 1
    assert result["bins"]["negative"]["count"] == 0


def test_native_and_common_cap_are_separate() -> None:
    target = [1.0, 4.0, 8.0, -2.0]
    prediction = [100.0, 4.2, 7.5, -2.1]
    groups = ["a"] * 4
    native = score_signed_ttc(target, prediction, groups)
    common = common_cap60_metrics(target, prediction, groups)
    assert native["diagnostics"]["mae_s"] != common["diagnostics"]["mae_s"]
    assert common["analysis_scope"] == "common_prediction_cap_60s_separate_from_native"


def test_group_bootstrap_resamples_whole_groups() -> None:
    target = np.tile(np.asarray([1.0, 4.0, 8.0, -2.0]), 2)
    groups = np.repeat(np.asarray(["a", "b"]), 4)
    result = paired_group_bootstrap(target, target + 0.1, target + 0.2, groups, repetitions=20)
    assert result["status"] == "COMPLETE"
    assert result["group_count"] == 2


def test_group_macro_midtakes_equal_group_weight_and_keeps_wrong_sign() -> None:
    target = [1.0, 4.0, 8.0, -2.0, 1.0, 1.0, 4.0, 8.0, -2.0]
    prediction = [1.1, 4.1, 8.1, 2.0, 2.0, 2.0, 4.1, 8.1, -2.1]
    groups = ["short"] * 4 + ["long"] * 5
    result = score_signed_ttc(target, prediction, groups)
    assert result["formula_admitted"] is True
    assert result["official_comparison_admitted"] is False
    assert result["bins"]["crucial"]["groups_with_bucket"] == 2
    assert result["diagnostics"]["sign_error_rate"] > 0


def test_outside_protocol_target_remains_in_secondary_metrics() -> None:
    result = score_signed_ttc(
        [1.0, 4.0, 8.0, -2.0, 12.0],
        [1.1, 4.1, 8.1, -2.1, 11.0],
        ["a"] * 5,
    )
    assert result["formula_admitted"] is True
    assert result["num_targets_outside_protocol"] == 1
    assert result["diagnostics"]["mae_s"] is not None


def test_crucial_overestimate_rte_tails_are_measured_only_on_overestimates() -> None:
    result = score_signed_ttc(
        [1.0, 2.0, 4.0, 8.0, -2.0],
        [2.0, 1.0, 4.1, 8.1, -2.1],
        ["a"] * 5,
    )
    assert result["diagnostics"]["crucial_positive_overestimate_rte_p90_pct"] == 100.0
    assert result["diagnostics"]["crucial_positive_overestimate_rte_p95_pct"] == 100.0


def test_bundle_contains_full_payload_and_verifies(tmp_path: Path) -> None:
    atomic_write_json(tmp_path / "RGB_PORT_STATE.json", {"tasks": {"x": {"status": "WAITING"}}})
    (tmp_path / "model.pt").write_bytes(b"full-state")
    bundle = tmp_path / "essential.zip"
    result = build_bundle(tmp_path, bundle, ["RGB_PORT_STATE.json", "model.pt"])
    assert result["status"] == "VERIFIED"
    assert verify_bundle(bundle)["bundle_sha256"] == result["bundle_sha256"]
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
    assert manifest["campaign_status"] == "INCOMPLETE"


def test_bundle_rejects_private_or_parent_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_bundle(tmp_path, tmp_path / "x.zip", ["../private/labels.csv"])


def test_heavy_owner_match_requires_real_python_module_argv() -> None:
    modules = ["operational.sota_eval.r1_resume"]
    assert _module_matches(["python", "-m", modules[0], "--run"], modules)
    assert not _module_matches(["rg", modules[0], "docs"], modules)
    assert not _module_matches(["pytest", "test_operational.sota_eval.r1_resume"], modules)


def test_external_block_receipt_is_terminal_without_becoming_failure(tmp_path: Path) -> None:
    receipt = tmp_path / "FCWD_RGB_STATUS.json"
    atomic_write_json(
        receipt,
        {
            "schema": "rgb_port_transfer_external_status_v1",
            "status": "BLOCKED_EXTERNAL",
            "reason": "RGB_TO_EVENT_CALIBRATION_MISSING",
        },
    )
    task = {
        "completion_artifact": str(receipt),
        "completion_schema": "rgb_port_transfer_external_status_v1",
        "terminal_artifact_status": "BLOCKED_EXTERNAL",
    }
    assert _terminal_artifact_blocker(task, tmp_path) == "RGB_TO_EVENT_CALIBRATION_MISSING"


def test_recovery_qa_completion_requires_exact_four_updates_and_matching_journal(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "ROOT_QA_RECEIPT.json"
    journal = tmp_path / "TECHNICAL_JOURNAL.json"
    test_name = str(Path("tests/test_rgb_port_producer_resume_technical.py"))
    task = {
        "completion_artifact": str(receipt),
        "completion_schema": "rgb_port_root_qa_v1",
        "completion_status_optional": True,
        "technical_qa_contract": {
            "journal": str(journal),
            "test": test_name,
            "expected_updates": 4,
        },
    }
    atomic_write_json(
        receipt,
        {
            "schema": "rgb_port_root_qa_v1",
            "exit_code": 0,
            "pending_upper": 0,
            "technical_updates_before": 1,
            "technical_updates_after": 5,
            "tests": {test_name: "sha"},
        },
    )
    atomic_write_json(
        journal,
        {"schema": "rgb_port_technical_journal_v1", "completed": 5, "pending_upper": 0},
    )
    assert _artifact_task_complete(task, tmp_path)
    value = read_json_shared(receipt)
    value["technical_updates_after"] = 4
    atomic_write_json(receipt, value)
    assert not _artifact_task_complete(task, tmp_path)


def test_global_qa_snapshot_and_typed_recovery_receipt_are_separate(tmp_path: Path) -> None:
    global_receipt = tmp_path / "ROOT_QA_RECEIPT.json"
    journal = tmp_path / "TECHNICAL_JOURNAL.json"
    atomic_write_json(global_receipt, {"schema": "rgb_port_root_qa_v1", "exit_code": 0})
    atomic_write_json(
        journal,
        {"schema": "rgb_port_technical_journal_v1", "completed": 1, "pending_upper": 0},
    )
    _snapshot_global_qa(tmp_path)
    atomic_write_json(global_receipt, {"schema": "rgb_port_root_qa_v1", "exit_code": 1})
    task = {
        "completion_copy_from": str(global_receipt),
        "completion_artifact": str(tmp_path / "qa" / "RECOVERY_QA_RECEIPT.json"),
    }
    _sync_completion_copy(task, tmp_path)
    assert read_json_shared(tmp_path / "qa" / "GLOBAL_QA_ADMISSION.json")["exit_code"] == 0
    assert read_json_shared(tmp_path / "qa" / "RECOVERY_QA_RECEIPT.json")["exit_code"] == 1


def test_execution_graph_has_only_authorized_fits_and_exact_actual_budget() -> None:
    config = _load_config(Path("configs/rgb_port/execution.json"))
    _validate_config(config)
    limits = _task_limits(config)
    assert sum(limits[fit_id] for fit_id in FIT_IDS) == 150_000
    assert all(limits[fit_id] == 30_330 for fit_id in FIT_IDS[:4])
    assert all(limits[fit_id] == 6_840 for fit_id in FIT_IDS[4:6])
    assert all(limits[fit_id] == 2_500 for fit_id in FIT_IDS[6:])
    tasks = {task["id"]: task for task in config["tasks"]}
    assert tasks["RECOVERY_QA"]["depends"] == ["P_RGB_CACHE"]
    assert all("RECOVERY_QA" in tasks[fit_id]["depends"] for fit_id in FIT_IDS[:4])
    assert tasks["PROFILE_ROUTES"]["heavy"] is True
    device_index = tasks["PROFILE_ROUTES"]["command"].index("--device")
    assert tasks["PROFILE_ROUTES"]["command"][device_index + 1] == "cuda"
    assert tasks["REPORT"]["depends"] == ["V_EVALUATION"]
    assert tasks["REPORT"]["soft_depends"] == [
        "PROFILE_ROUTES",
        "DEV32_SCORE",
        "FCWD_RGB_STATUS",
    ]
    assert "optional:report/REPORT_RECEIPT.json" in config["package"]["members"]


def _mock_complete_fit(run: Path, fit_id: str, updates: int, *, torch_state: bool) -> None:
    import torch

    fit = run / "fits" / fit_id
    fit.mkdir(parents=True)
    checkpoint = fit / "checkpoint_last.pt"
    if torch_state:
        torch.save(
            {
                "identity": {
                    "fit_id": fit_id,
                    "normalizer_sha256": "normalizer",
                    "parent_sha256": "parent",
                }
            },
            checkpoint,
        )
    else:
        checkpoint.write_bytes(f"full-state:{fit_id}".encode())
    checkpoint_sha = sha256_file(checkpoint)
    atomic_write_json(
        fit / "CHECKPOINT_RECEIPT.json",
        {
            "schema": "rgb_port_checkpoint_receipt_v1",
            "fit_id": fit_id,
            "status": "COMPLETE",
            "completed_updates": updates,
            "scientific_endpoint": True,
            "complete_state": True,
            "accumulation_index": 0,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "identity_sha256": f"identity-{fit_id}",
        },
    )
    atomic_write_json(fit / "UPDATE_JOURNAL.json", {"completed_updates": updates})
    atomic_write_json(fit / "RUN_PROVENANCE.json", {"status": "COMPLETE"})
    _freeze_fit_endpoint(fit_id, run)


def test_round_freezes_satisfy_feature_transfer_and_head_loader_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import operational.rgb_port.predict_heads as predict_heads

    config = _load_config(Path("configs/rgb_port/execution.json"))
    limits = _task_limits(config)
    for fit_id in FIT_IDS[:6]:
        _mock_complete_fit(tmp_path, fit_id, limits[fit_id], torch_state=False)
    _write_parent_round_freeze(config, tmp_path)
    parent_freeze = tmp_path / "rounds" / "PRODUCERS_PAIR_ENDPOINT_FREEZE.json"
    feature_bindings = validate_endpoint_freeze(parent_freeze)
    transfer_bindings = _validate_endpoint_freeze(parent_freeze)
    assert feature_bindings == transfer_bindings
    assert set(feature_bindings) == set(FIT_IDS[:6])

    for fit_id in FIT_IDS[6:]:
        _mock_complete_fit(tmp_path, fit_id, limits[fit_id], torch_state=True)
    _write_round_freeze(config, tmp_path)
    head_round = read_json_shared(tmp_path / "rounds" / "HEADS_V_ROUND_FREEZE.json")
    assert all(
        entry["status"] == "COMPLETE"
        and entry["scientific_endpoint"] is True
        and Path(entry["checkpoint_path"]).is_file()
        for entry in head_round["endpoints"].values()
    )
    monkeypatch.setattr(predict_heads, "load_head_endpoint", lambda *_args, **_kwargs: object())
    models, bindings = _frozen_heads(tmp_path)
    assert set(models) == set(FIT_IDS[6:])
    assert set(bindings) == set(FIT_IDS[6:])
