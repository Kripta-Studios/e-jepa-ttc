from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import runpod_test12_sequence_queue as queue


def _write(path: Path, value: dict[str, Any] | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


def test_sequences_are_derived_from_exact_frozen_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequences = [f"sequence{i:02d}" for i in range(12)]
    inventory = tmp_path / "inventory.json"
    _write(
        inventory,
        {"files": [{"path": f"data/test/{sequence}/events.h5"} for sequence in sequences]},
    )
    monkeypatch.setattr(queue, "INVENTORY", inventory)
    monkeypatch.setattr(queue, "INVENTORY_SHA256", queue.digest(inventory))
    assert queue.frozen_sequences() == sequences


def test_only_exact_predict_and_package_commands_are_admitted() -> None:
    sequences = {"sequence00"}
    assert queue.command_name(queue.expected_command("sequence00"), sequences) == "sequence00"
    assert queue.command_name(queue.expected_command(queue.PACKAGE), sequences) == queue.PACKAGE
    with pytest.raises(ValueError, match="Unrecognized"):
        queue.command_name([*queue.expected_command("sequence00"), "--device", "cpu"], sequences)


def test_completion_requires_bound_admission_and_feature_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "RUN", tmp_path)
    monkeypatch.setattr(queue, "RGB", tmp_path / "rgb")
    root = tmp_path / "final/test12/staged_sequences"
    admission = root / "sequence00.admission.json"
    feature = root / "sequence00.features.npz"
    _write(admission, {"status": "PASS"})
    _write(feature, b"features")
    _write(
        root / "sequence00.complete.json",
        {
            "schema": "test12_staged_sequence_complete_v1",
            "status": "COMPLETE",
            "sequence_id": "sequence00",
            "private_targets_read": False,
            "external_submission": False,
            "canonical_tokens": ["token"],
            "admission_sha256": queue.digest(admission),
            "feature_cache_sha256": queue.digest(feature),
        },
    )
    complete = root / "sequence00.complete.json"
    _write(
        tmp_path / "rgb/consumed/sequence00.json",
        {
            "schema": "test12_staged_rgb_consumed_v1",
            "status": "PASS",
            "sequence_id": "sequence00",
            "private_targets_read": False,
            "outputs": [
                {
                    "path": str(complete),
                    "sha256": queue.digest(complete),
                    "bytes": complete.stat().st_size,
                }
            ],
        },
    )
    assert queue.sequence_complete("sequence00")
    feature.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Malformed"):
        queue.sequence_complete("sequence00")


def test_registration_waits_for_exact_exec(monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = "sequence00"

    class Child:
        pid = 123

        @staticmethod
        def poll() -> None:
            return None

    class Process:
        pid = 123

        @staticmethod
        def cmdline() -> list[str]:
            return queue.expected_command(sequence)

        @staticmethod
        def status() -> str:
            return "running"

        @staticmethod
        def create_time() -> float:
            return 1.0

    monkeypatch.setattr(queue.psutil, "Process", lambda _pid: Process())
    record = queue.wait_registration(Child(), sequence, {sequence}, timeout=0.1)  # type: ignore[arg-type]
    assert record is not None and record["name"] == sequence


def test_registration_returns_none_for_fast_exit() -> None:
    class Child:
        pid = 123

        @staticmethod
        def poll() -> int:
            return 1

    assert queue.wait_registration(Child(), "sequence00", {"sequence00"}) is None  # type: ignore[arg-type]


def test_readiness_waits_for_both_sources_and_rejects_wrong_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event, rgb = tmp_path / "event", tmp_path / "rgb"
    monkeypatch.setattr(queue, "EVENT", event)
    monkeypatch.setattr(queue, "RGB", rgb)
    _write(event / "sequences/sequence00.json", {"status": "COMPLETE"})
    assert not queue.source_ready("sequence00")
    _write(
        rgb / "sequences/sequence00.json",
        {"status": "READY", "sequence_id": "other"},
    )
    with pytest.raises(ValueError, match="another sequence"):
        queue.source_ready("sequence00")


def test_complete_without_consumed_is_repairable_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "RUN", tmp_path)
    monkeypatch.setattr(queue, "RGB", tmp_path / "rgb")
    root = tmp_path / "final/test12/staged_sequences"
    admission = root / "sequence00.admission.json"
    feature = root / "sequence00.features.npz"
    _write(admission, {"status": "PASS"})
    _write(feature, b"features")
    _write(
        root / "sequence00.complete.json",
        {
            "schema": "test12_staged_sequence_complete_v1",
            "status": "COMPLETE",
            "sequence_id": "sequence00",
            "private_targets_read": False,
            "external_submission": False,
            "canonical_tokens": ["token"],
            "admission_sha256": queue.digest(admission),
            "feature_cache_sha256": queue.digest(feature),
        },
    )
    assert not queue.sequence_complete("sequence00")


def test_resource_pause_classification_is_narrow(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text("Traceback: ResourcePause: resource reserve unavailable")
    assert queue.resource_pause(log)
    log.write_text("ValueError: scientific binding differs")
    assert not queue.resource_pause(log)


def test_package_requires_all_sequence_completion_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "RUN", tmp_path)
    sequence = "sequence00"
    complete = tmp_path / f"final/test12/staged_sequences/{sequence}.complete.json"
    receipt = tmp_path / "final/test12/PREDICTION_RECEIPT.json"
    package = tmp_path / "final/test12/PHASE_ONLY_S7_TRAIN40_SCIENTIFIC_OOF.json"
    archive = package.with_suffix(".zip")
    _write(complete, {"status": "COMPLETE"})
    _write(
        receipt,
        {
            "status": "COMPLETE",
            "private_targets_read": False,
            "external_submission": False,
            "staged_sequence_completion_sha256": {sequence: queue.digest(complete)},
        },
    )
    _write(archive, b"zip")
    package_value = {
        "schema": "scientific_oof_test12_package_v1",
        "status": "PACKAGED_NOT_SUBMITTED",
        "prediction_receipt_sha256": queue.digest(receipt),
        "private_targets_read": False,
        "external_submission": False,
        "zip_sha256": queue.digest(archive),
    }
    package_value["receipt_sha256"] = queue.canonical(package_value)
    _write(package, package_value)
    assert queue.package_complete([sequence])


def test_partial_prediction_receipt_is_resumable_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "RUN", tmp_path)
    receipt = tmp_path / "final/test12/PREDICTION_RECEIPT.json"
    _write(receipt, {"status": "COMPLETE"})
    assert not queue.package_complete(["sequence00"])
