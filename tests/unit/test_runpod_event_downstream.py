from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from operational.runpod_migration.event_queue import (
    _needs_context_replay,
    event_dependency_state,
    run_event_downstream,
)
from scripts.runpod_event_queue_v1 import _canonical, _raw_gate, _verify_supplemental


def test_event_queue_waits_for_endpoint_admission_not_progress(tmp_path: Path) -> None:
    run = tmp_path / "run"
    progress = run / "rotations/2/fits/E_A5/PROGRESS.json"
    progress.parent.mkdir(parents=True)
    progress.write_text("{}\n", encoding="utf-8")
    result = run_event_downstream(
        run,
        2,
        deadline_utc=datetime.now(UTC) + timedelta(minutes=1),
        raw_gate=lambda: None,
    )
    assert result["stage"] == "WAIT_E_A5"
    assert result["status"] == "WAITING_DEPENDENCY"
    assert result["rgb_model_inference_executed"] is False


def test_event_queue_rejects_naive_deadline(tmp_path: Path) -> None:
    tmp_path.joinpath("run").mkdir()
    with pytest.raises(ValueError, match="timezone-aware"):
        run_event_downstream(
            tmp_path / "run", 2, deadline_utc=datetime.now(), raw_gate=lambda: None
        )


def test_event_queue_calls_only_event_branches() -> None:
    source = inspect.getsource(run_event_downstream)
    assert '"PAIR_E"' in source
    assert '"E_CTX"' in source
    assert "PAIR_R" not in source
    assert "R_CTX" not in source


def test_event_context_replays_until_fragment_and_receipt_are_durable(tmp_path: Path) -> None:
    assert _needs_context_replay(tmp_path, {"E_CTX": True}) is True
    fragment = tmp_path / "generation/E_CTX/fragment.npz"
    fragment.parent.mkdir(parents=True)
    fragment.write_bytes(b"fragment")
    assert _needs_context_replay(tmp_path, {"E_CTX": True}) is True
    fragment.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    assert _needs_context_replay(tmp_path, {"E_CTX": True}) is False


def test_pair_endpoint_is_retained_while_c2f_is_absent(tmp_path: Path) -> None:
    fits = tmp_path / "run/rotations/3/fits"
    for fit in ("E_A5", "PAIR_E"):
        path = fits / fit / "ENDPOINT_ADMISSION.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")
    assert event_dependency_state(tmp_path / "run", 3) == {
        "E_A5": True,
        "E_C2F": False,
        "PAIR_E": True,
        "E_CTX": False,
    }


def test_raw_gate_waits_then_accepts_exact_complete_receipt(tmp_path: Path) -> None:
    path = tmp_path / "EVENT_RAW_COVERAGE_RECEIPT.json"
    assert _raw_gate(path, "a" * 64) is False
    receipt = {
        "schema": "ejepa_event_raw_coverage_v1",
        "status": "VERIFIED",
        "raw_deferred_manifest_sha256": "a" * 64,
        "files": 41,
        "sequences": 40,
        "total_bytes": 449_003_803_674,
        "hashes_verified": 41,
        "availability_verified": True,
    }
    receipt["receipt_sha256"] = _canonical(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert _raw_gate(path, "a" * 64) is True
    receipt["total_bytes"] += 1
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid or changed"):
        _raw_gate(path, "a" * 64)


def test_supplemental_gate_binds_thirteen_metadata_files(tmp_path: Path) -> None:
    path = tmp_path / "SUPPLEMENTAL_ADMISSION.json"
    receipt = {
        "schema": "ejepa_event_downstream_supplemental_admission_v1",
        "status": "VERIFIED",
        "scientific_config_sha256": "a" * 64,
        "files": 13,
        "total_bytes": 81_182_038,
        "hashes_verified": 13,
        "mapping_verified": True,
        "manifest_sha256": "b" * 64,
        "immutable_seal_sha256": "c" * 64,
    }
    receipt["receipt_sha256"] = _canonical(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    observed = _verify_supplemental(path, "a" * 64)
    assert set(observed) == {"event_supplemental_admission_sha256"}
