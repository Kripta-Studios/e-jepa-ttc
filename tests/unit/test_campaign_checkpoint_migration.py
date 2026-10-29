"""CPU-only safety tests for optimized campaign checkpoint migration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from e_jepa_ttc.detection.campaign_checkpoint_migration import (
    assert_deep_exact,
    checkpoint_selection_identity,
    component_sha256,
    migrate_checkpoint_provenance,
    stable_copy_checkpoint,
    typed_state_sha256,
    verify_evaluation_boundary,
    verify_runtime_transition,
    verify_snapshot,
    verify_source_state_receipt,
    windows_process_creation_time_100ns,
)
from e_jepa_ttc.detection.campaign_preregistration import (
    base_runtime_relative_paths,
    optimized_runtime_relative_paths,
)
from e_jepa_ttc.detection.campaign_training import canonical_sha256, sha256_file


def checkpoint(provenance: dict[str, object], *, update: int = 8, best: float | None = 0.4):
    model = OrderedDict(weight=torch.arange(6, dtype=torch.float32).reshape(2, 3))
    model._metadata = OrderedDict([("", {"version": 1})])  # type: ignore[attr-defined]
    return {
        "schema": "eap_detection_campaign_checkpoint_v1",
        "producer": "scripts/train_detection_campaign.py",
        "inference_contract": "campaign_v1_strict_loader_required",
        "config_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "runtime_provenance": provenance,
        "runtime_provenance_sha256": canonical_sha256(provenance),
        "update": update,
        "best_metric": best,
        "progress": {
            "training_seconds": 2.0,
            "empty_target_exposures": 0,
            "valid_velocity_object_exposures": 4,
        },
        "data_member_sha256": {"s:f": "c" * 64},
        "model": model,
        "optimizer": {"state": {0: {"step": torch.tensor(8)}}},
        "scheduler": {"update": update},
        "sampler": {"epoch": 1, "position": 3, "order": [2, 0, 1]},
        "rng": {"torch_cpu": torch.arange(4, dtype=torch.uint8), "torch_cuda": []},
    }


def metric_payload(protocol: str, value: float, sample_count: int = 2) -> dict[str, object]:
    return {
        "metrics_v2": {
            "protocol_sha256": protocol,
            "official_eap_score": False,
            "sample_count": sample_count,
            "population": {"identity_check": True},
            "predictions": {"identity_check": True},
            "dev_AP101_supported_classes": value,
        }
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def test_typed_hash_distinguishes_types_order_stride_and_metadata() -> None:
    assert typed_state_sha256([1, 23]) != typed_state_sha256([12, 3])
    assert typed_state_sha256([1, 2]) != typed_state_sha256((1, 2))
    assert typed_state_sha256({"a": 1, "b": 2}) != typed_state_sha256(
        {"b": 2, "a": 1}
    )
    contiguous = torch.arange(12).reshape(3, 4)
    transposed = contiguous.t()
    assert typed_state_sha256(contiguous) != typed_state_sha256(transposed)
    left = OrderedDict([("x", torch.tensor([1]))])
    right = OrderedDict([("x", torch.tensor([1]))])
    left._metadata = {"": {"version": 1}}  # type: ignore[attr-defined]
    right._metadata = {"": {"version": 2}}  # type: ignore[attr-defined]
    assert typed_state_sha256(left) != typed_state_sha256(right)
    with pytest.raises(ValueError, match="metadata"):
        assert_deep_exact(left, right)


def test_stable_copy_requires_explicit_sha_and_never_changes_source(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint-bytes")
    before = source.read_bytes()
    destination = tmp_path / "boundary/last.pt"
    receipt = stable_copy_checkpoint(
        source, destination, expected_source_sha256=sha256_file(source)
    )
    assert destination.read_bytes() == before == source.read_bytes()
    assert receipt["status"] == "PASS_STABLE_ATOMIC_COPY"
    with pytest.raises(FileExistsError):
        stable_copy_checkpoint(source, destination, expected_source_sha256=sha256_file(source))


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics only")
def test_stable_copy_never_blocks_concurrent_producer_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e_jepa_ttc.detection.campaign_checkpoint_migration as migration

    source = tmp_path / "last.pt"
    replacement = tmp_path / "next.pt"
    second_replacement = tmp_path / "next_again.pt"
    destination = tmp_path / "pinned/last.pt"
    source.write_bytes(b"old-checkpoint" * 1024)
    replacement.write_bytes(b"new-checkpoint" * 1024)
    second_replacement.write_bytes(b"newer-checkpoint" * 1024)
    source_sha256 = sha256_file(source)
    link_created = threading.Event()
    first_replacement_finished = threading.Event()
    pinned_hash_started = threading.Event()
    second_replacement_finished = threading.Event()
    replacement_error: list[BaseException] = []
    original_link = os.link
    original_hash = migration.sha256_file

    def synchronized_link(src, dst):
        original_link(src, dst)
        link_created.set()
        assert first_replacement_finished.wait(timeout=5)

    def synchronized_hash(path):
        if Path(path) == destination and not pinned_hash_started.is_set():
            pinned_hash_started.set()
            assert second_replacement_finished.wait(timeout=5)
        return original_hash(path)

    def replace_from_producer() -> None:
        try:
            assert link_created.wait(timeout=5), "stable pin never linked its source"
            os.replace(replacement, source)
            first_replacement_finished.set()
            assert pinned_hash_started.wait(timeout=5), "stable pin never began hashing"
            os.replace(second_replacement, source)
        except BaseException as error:  # pragma: no cover - asserted in caller
            replacement_error.append(error)
        finally:
            first_replacement_finished.set()
            second_replacement_finished.set()

    monkeypatch.setattr(os, "link", synchronized_link)
    monkeypatch.setattr(migration, "sha256_file", synchronized_hash)
    producer = threading.Thread(target=replace_from_producer)
    producer.start()
    receipt = migration._experimental_detached_hardlink_pin(
        source,
        destination,
        expected_source_sha256=source_sha256,
    )
    producer.join(timeout=5)
    assert not producer.is_alive()
    assert replacement_error == []
    assert receipt["status"] == "EXPERIMENTAL_NOT_ADMITTED_DETACHED_HARDLINK_PIN"
    assert destination.read_bytes() == b"old-checkpoint" * 1024
    assert source.read_bytes() == b"newer-checkpoint" * 1024


def test_migration_changes_only_runtime_provenance(tmp_path: Path) -> None:
    old = {"runtime_preregistration_sha256": "1" * 64}
    new = {"runtime_preregistration_sha256": "2" * 64}
    source = tmp_path / "pinned.pt"
    torch.save(checkpoint(old), source)
    before = torch.load(source, map_location="cpu", weights_only=True)
    destination = tmp_path / "migrated.pt"
    receipt = migrate_checkpoint_provenance(
        source, destination, old_provenance=old, new_provenance=new
    )
    after = torch.load(destination, map_location="cpu", weights_only=True)
    assert receipt["component_sha256"] == component_sha256(before) == component_sha256(after)
    assert receipt["parent_full_sha256"] == sha256_file(source)
    assert after["runtime_provenance"] == new
    for key in before:
        if key not in {"runtime_provenance", "runtime_provenance_sha256"}:
            assert_deep_exact(before[key], after[key], key)


def test_evaluation_gate_requires_monitor_pilot_full_and_selected_consistency(
    tmp_path: Path,
) -> None:
    provenance = {"runtime_preregistration_sha256": "1" * 64}
    last = checkpoint(provenance, update=4096, best=0.5)
    selected = checkpoint_selection_identity(
        checkpoint(provenance, update=4096, best=0.5)
    )
    for update, value in ((1024, 0.4), (2048, 0.3), (4096, 0.5)):
        write_json(
            tmp_path / f"evaluation/selection_update_{update:06d}.json",
            metric_payload("b" * 64, value),
        )
        write_json(
            tmp_path / f"evaluation/pilot_parity_update_{update:06d}.json",
            metric_payload("b" * 64, value),
        )
    with pytest.raises(ValueError, match="full_selection"):
        verify_evaluation_boundary(
            tmp_path,
            checkpoint=last,
            selected=selected,
            evaluation_updates=(1024, 2048, 4096),
            full_selection_updates=(4096,),
            selection_metric="dev_AP101_supported_classes",
        )
    write_json(
        tmp_path / "evaluation/full_selection_update_004096.json",
        metric_payload("b" * 64, 0.5, 12),
    )
    receipt = verify_evaluation_boundary(
        tmp_path,
        checkpoint=last,
        selected=selected,
        evaluation_updates=(1024, 2048, 4096),
        full_selection_updates=(4096,),
        selection_metric="dev_AP101_supported_classes",
    )
    assert receipt["selected_update"] == 4096
    assert receipt["required_full_selection_updates"] == [4096]


def test_evaluation_gate_uses_first_strict_max_for_selected(tmp_path: Path) -> None:
    provenance = {"runtime_preregistration_sha256": "1" * 64}
    last = checkpoint(provenance, update=2048, best=0.5)
    selected = checkpoint_selection_identity(
        checkpoint(provenance, update=1024, best=0.5)
    )
    for update in (1024, 2048):
        for panel in ("selection", "pilot_parity"):
            write_json(
                tmp_path / f"evaluation/{panel}_update_{update:06d}.json",
                metric_payload("b" * 64, 0.5),
            )
    receipt = verify_evaluation_boundary(
        tmp_path,
        checkpoint=last,
        selected=selected,
        evaluation_updates=(1024, 2048),
        full_selection_updates=(),
        selection_metric="dev_AP101_supported_classes",
    )
    assert receipt["selected_update"] == 1024


def preregistrations() -> tuple[dict[str, object], dict[str, object]]:
    performance = "configs/detection_campaign/performance/indexed_shared_v1.json"
    historical = set(base_runtime_relative_paths()) - {
        "src/e_jepa_ttc/detection/campaign_preregistration.py",
        "src/e_jepa_ttc/detection/evaluation_ap_fast.py",
    }
    extra = {"data/admission/media_admission.json"}
    old_code = {path: "1" * 64 for path in sorted(historical | extra)}
    target_code = {
        path: old_code.get(path, "2" * 64)
        for path in sorted(set(optimized_runtime_relative_paths(performance)) | extra)
    }
    for path in (
        "scripts/train_detection_campaign.py",
        "scripts/smoke_monodgp_portable.py",
        "src/e_jepa_ttc/detection/development_metrics.py",
        "src/e_jepa_ttc/detection/evaluation_v2.py",
    ):
        target_code[path] = "3" * 64
    full_model = {
        "backbone": "resnet50",
        "num_classes": 8,
        "num_queries": 50,
        "group_num": 11,
        "predict_velocity": True,
        "velocity_loss_coef": 0.1,
        "device": "cuda",
        "dec_layers": 3,
    }
    common = {
        "schema": "eap_detection_campaign_runtime_preregistration_v1",
        "config": {"recipe": "r0"},
        "config_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "source_receipt": {"revision": "fixed"},
        "source_receipt_sha256": "c" * 64,
        "portable_source_sha256": {"model.py": "d" * 64},
        "portable_source_sha256_manifest": "e" * 64,
        "dataset_media_sha256": "f" * 64,
        "media_admission_receipt_sha256": "0" * 64,
        "media_admission_population_sha256": "4" * 64,
        "matmul_precision": "highest",
    }
    old = {
        **common,
        "effective_model": full_model,
        "effective_model_sha256": canonical_sha256(full_model),
        "code_sha256": old_code,
        "code_sha256_manifest": canonical_sha256(old_code),
    }
    target = {
        **common,
        "effective_model": full_model,
        "effective_model_sha256": canonical_sha256(full_model),
        "code_sha256": target_code,
        "code_sha256_manifest": canonical_sha256(target_code),
        "runtime_extension": {"schema": "eap_detection_indexed_runtime_v1"},
    }
    return old, target


def test_runtime_transition_is_exact_allowlist_and_model_projection() -> None:
    old, target = preregistrations()
    receipt = verify_runtime_transition(
        old,
        target,
        performance_config_relative=(
            "configs/detection_campaign/performance/indexed_shared_v1.json"
        ),
    )
    assert receipt["status"] == "PASS_INDEXED_AP_RUNTIME_ONLY"
    changed = dict(target)
    changed["dataset_media_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="dataset_media"):
        verify_runtime_transition(
            old,
            changed,
            performance_config_relative=(
                "configs/detection_campaign/performance/indexed_shared_v1.json"
            ),
        )
    changed = dict(target)
    changed_model = dict(changed["effective_model"])  # type: ignore[arg-type]
    changed_model["num_queries"] = 51
    changed["effective_model"] = changed_model
    changed["effective_model_sha256"] = canonical_sha256(changed_model)
    with pytest.raises(ValueError, match="effective_model"):
        verify_runtime_transition(
            old,
            changed,
            performance_config_relative=(
                "configs/detection_campaign/performance/indexed_shared_v1.json"
            ),
        )


def test_snapshot_verification_rejects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "code/module.py"
    path.parent.mkdir(parents=True)
    path.write_text("value = 1\n", encoding="utf-8")
    manifest = {"code/module.py": sha256_file(path)}
    assert verify_snapshot(tmp_path, manifest)["status"] == "PASS_EXACT_SOURCE_SNAPSHOT"
    path.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ"):
        verify_snapshot(tmp_path, manifest)


def stopped_receipt_fixture(
    tmp_path: Path, *, status: str, exit_code: int, pid: int, creation: int
) -> tuple[Path, Path, list[str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_run = tmp_path / "run"
    producer = tmp_path / "train_detection_campaign.py"
    producer.write_text("# frozen producer\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("experiment: r0\n", encoding="utf-8")
    command = [
        "python",
        str(producer),
        "--config",
        str(config),
        "--output",
        str(source_run),
    ]
    ended_at = "2026-10-29T18:23:23Z"
    evidence = tmp_path / "exit.json"
    write_json(
        evidence,
        {
            "schema": "eap_detection_process_exit_evidence_v1",
            "status": status,
            "pid": pid,
            "creation_time_filetime_100ns": creation,
            "command": command,
            "returncode": exit_code,
            "ended_at_utc": ended_at,
            "source_run": str(source_run.resolve()),
        },
    )
    receipt = tmp_path / "stopped.json"
    write_json(
        receipt,
        {
            "schema": "eap_detection_stopped_process_receipt_v1",
            "status": status,
            "pid": pid,
            "creation_time_filetime_100ns": creation,
            "command": command,
            "exit_code": exit_code,
            "ended_at_utc": ended_at,
            "exit_evidence_path": str(evidence),
            "exit_evidence_sha256": sha256_file(evidence),
            "source_run": str(source_run),
            "producer_path": str(producer),
            "config_path": str(config),
            "config_sha256": sha256_file(config),
        },
    )
    return receipt, source_run, command


@pytest.mark.skipif(os.name != "nt", reason="Windows process identity only")
def test_stopped_receipt_rejects_same_live_pid_creation_identity(tmp_path: Path) -> None:
    creation = windows_process_creation_time_100ns(os.getpid())
    assert creation is not None
    receipt, source_run, _ = stopped_receipt_fixture(
        tmp_path,
        status="INTENTIONALLY_STOPPED_FOR_PERFORMANCE_ADMISSION",
        exit_code=1,
        pid=os.getpid(),
        creation=creation,
    )
    with pytest.raises(RuntimeError, match="still alive"):
        verify_source_state_receipt(receipt, source_run=source_run)


@pytest.mark.skipif(os.name != "nt", reason="Windows process identity only")
def test_stopped_receipt_accepts_absent_process_and_binds_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e_jepa_ttc.detection.campaign_checkpoint_migration as migration

    receipt, source_run, _ = stopped_receipt_fixture(
        tmp_path,
        status="INTENTIONALLY_STOPPED_FOR_PERFORMANCE_ADMISSION",
        exit_code=1,
        pid=2_147_483_647,
        creation=1,
    )
    monkeypatch.setattr(migration, "_live_windows_processes", lambda: [])
    result = verify_source_state_receipt(receipt, source_run=source_run)
    assert result["status"] == "PASS_RECEIPT_PROCESS_IDENTITY_NOT_LIVE"
    assert result["observed_pid_creation_time_filetime_100ns"] is None


@pytest.mark.skipif(os.name != "nt", reason="Windows process identity only")
def test_intentional_stop_records_nonzero_without_completion_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e_jepa_ttc.detection.campaign_checkpoint_migration as migration

    receipt, source_run, _ = stopped_receipt_fixture(
        tmp_path,
        status="INTENTIONALLY_STOPPED_FOR_PERFORMANCE_ADMISSION",
        exit_code=-15,
        pid=2_147_483_647,
        creation=1,
    )
    monkeypatch.setattr(migration, "_live_windows_processes", lambda: [])
    result = verify_source_state_receipt(receipt, source_run=source_run)
    assert result["exit_code"] == -15
    assert result["training_completion_claimed"] is False


@pytest.mark.skipif(os.name != "nt", reason="Windows process identity only")
def test_stopped_receipt_rejects_unrelated_command_missing_evidence_and_stale_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e_jepa_ttc.detection.campaign_checkpoint_migration as migration

    receipt_path, source_run, _ = stopped_receipt_fixture(
        tmp_path,
        status="INTENTIONALLY_STOPPED_FOR_PERFORMANCE_ADMISSION",
        exit_code=1,
        pid=2_147_483_647,
        creation=1,
    )
    monkeypatch.setattr(migration, "_live_windows_processes", lambda: [])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["command"] = ["python", str(tmp_path / "unrelated.py")]
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="lacks --config/--output"):
        verify_source_state_receipt(receipt_path, source_run=source_run)

    receipt_path, source_run, _ = stopped_receipt_fixture(
        tmp_path / "missing",
        status="INTENTIONALLY_STOPPED_FOR_PERFORMANCE_ADMISSION",
        exit_code=1,
        pid=2_147_483_647,
        creation=1,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    evidence_path = Path(receipt["exit_evidence_path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.pop("source_run")
    write_json(evidence_path, evidence)
    receipt["exit_evidence_sha256"] = sha256_file(evidence_path)
    write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="evidence fields"):
        verify_source_state_receipt(receipt_path, source_run=source_run)

def test_completed_segment_binds_queue_summary_and_no_live_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import e_jepa_ttc.detection.campaign_checkpoint_migration as migration

    source_run = tmp_path / "run"
    summary = source_run / "summary.json"
    write_json(summary, {"status": "BUDGET_STOPPED", "final_update": 1024})
    queue = tmp_path / "queue_child.json"
    command = ["python", "train_detection_campaign.py", "--output", str(source_run)]
    write_json(
        queue,
        {
            "schema": "phase_ttc_detection_campaign_queue_child_v1",
            "status": "EXITED_ZERO",
            "command": command,
            "output": str(source_run),
            "returncode": 0,
            "timed_out": False,
        },
    )
    receipt = tmp_path / "completed.json"
    write_json(
        receipt,
        {
            "schema": "eap_detection_completed_segment_receipt_v1",
            "status": "COMPLETED_SEGMENT",
            "queue_child_path": str(queue),
            "queue_child_sha256": sha256_file(queue),
            "summary_path": str(summary),
            "summary_sha256": sha256_file(summary),
            "final_update": 1024,
        },
    )
    monkeypatch.setattr(migration, "_live_windows_processes", lambda: [])
    result = verify_source_state_receipt(receipt, source_run=source_run)
    assert result["status"] == "PASS_COMPLETED_SEGMENT_NO_LIVE_OWNER"
    monkeypatch.setattr(
        migration,
        "_live_windows_processes",
        lambda: [{"ProcessId": 22, "CommandLine": " ".join(command)}],
    )
    with pytest.raises(RuntimeError, match="still owns"):
        verify_source_state_receipt(receipt, source_run=source_run)
    monkeypatch.setattr(migration, "_live_windows_processes", lambda: [])
    write_json(summary, {"status": "BUDGET_STOPPED", "final_update": 2048})
    stale = json.loads(receipt.read_text(encoding="utf-8"))
    stale["summary_sha256"] = sha256_file(summary)
    write_json(receipt, stale)
    with pytest.raises(ValueError, match="queue/summary contract"):
        verify_source_state_receipt(receipt, source_run=source_run)


@pytest.mark.skipif(os.name != "nt", reason="Windows live-owner inventory only")
def test_stopped_receipt_rejects_wrong_pid_while_real_owner_is_alive(
    tmp_path: Path,
) -> None:
    receipt, source_run, command = stopped_receipt_fixture(
        tmp_path,
        status="INTENTIONALLY_STOPPED_FOR_PERFORMANCE_ADMISSION",
        exit_code=1,
        pid=2_147_483_647,
        creation=1,
    )
    producer = Path(command[1])
    producer.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    child_command = [sys.executable, *command[1:]]
    child = subprocess.Popen(child_command)
    try:
        with pytest.raises(RuntimeError, match="still owns"):
            verify_source_state_receipt(receipt, source_run=source_run)
    finally:
        child.terminate()
        child.wait(timeout=10)
