"""Reject unsafe recovery while admitting post-PAIR materialization pauses."""

import importlib.util
import json
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "input_pause",
    Path(__file__).resolve().parents[2] / "scripts/runpod_event_resource_watchdog_v12.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def paused(tmp_path):
    def write(p, d):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d))

    task = {"run": str(tmp_path), "rotation": 3}
    owner = {"pid": 100, "start_ticks": 500, "started_epoch": time.time() - 60}
    write(
        tmp_path / "EVENT_DOWNSTREAM_PAUSE_R3.json",
        {
            "schema": "ejepa_event_downstream_pause_v1",
            "status": "PAUSED_RECOVERABLE",
            "reason": "resource reserve unavailable",
            "manual": False,
            "rotation": 3,
            "pause_classification": "AUTOMATIC_RESOURCE_PAUSE",
            "owner_pid": 100,
            "owner_start_ticks": 500,
            "resources": {},
        },
    )
    base = tmp_path / "rotations/3"
    fit = base / "fits/PAIR_E"
    fit.mkdir(parents=True)
    checkpoint = fit / "checkpoint_last.pt"
    checkpoint.write_bytes(b"fixture")
    version = fit / "checkpoint_versions/checkpoint_000010.pt"
    version.parent.mkdir()
    version.write_bytes(b"fixture")
    sha = m.digest(checkpoint)
    receipt = {
        "status": "COMPLETE",
        "scientific_endpoint": True,
        "complete_state": True,
        "accumulation_index": 0,
        "completed_updates": 10,
        "identity_sha256": "identity",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha,
        "execution_binding_sha256": None,
        "recovery_upper": 0,
    }
    write(fit / "CHECKPOINT_RECEIPT.json", receipt)
    write(fit / "CHECKPOINT_POINTER.json", {**receipt, "version": version.name})
    write(
        fit / "UPDATE_JOURNAL.json", {**receipt, "durable_updates": 10, "pending_update_upper": 0}
    )
    endpoint = {
        "status": "PASS",
        "scientific_endpoint": True,
        "fit_id": "PAIR_E",
        "rotation": 3,
        "updates": 10,
        "checkpoint_identity_sha256": "identity",
        "checkpoint_sha256": sha,
        "checkpoint_receipt_sha256": m.digest(fit / "CHECKPOINT_RECEIPT.json"),
    }
    endpoint["admission_payload_sha256"] = m.canonical(endpoint)
    write(fit / "ENDPOINT_ADMISSION.json", endpoint)
    for parent in ["E_A5", "E_C2F"]:
        write(
            base / f"fits/{parent}/CHECKPOINT_RECEIPT.json",
            {"status": "COMPLETE", "checkpoint_sha256": parent},
        )
    parts = base / "refiner/E_CTX/parts"
    binding = {
        "schema": "scientific_oof_event_history_v1",
        "parents": {"PAIR_E": sha, "E_A5": "E_A5", "E_C2F": "E_C2F"},
    }
    write(parts / "BINDING.json", binding)
    data = parts / "part_000000.npz"
    data.write_bytes(b"inputfixture")
    write(
        data.with_suffix(".json"),
        {
            "optimizer_updates": 0,
            "rows": 1,
            "binding_sha256": m.canonical(binding),
            "sha256": m.digest(data),
        },
    )
    return task, owner, write


def test_complete_pair_can_resume(paused):
    task, owner, _ = paused
    assert m.completed_pair_input_pause(task, owner)


def test_unchanged_pair_prefix_cannot_reset_retries(tmp_path):
    root = tmp_path / "rotations/3/caches/PAIR_E.parts"
    root.mkdir(parents=True)
    (root / "part_000000.json").write_text("{}")
    calls = []

    def original(*args):
        calls.append(args)
        return "full_verifier_called"

    task = {"run": str(tmp_path), "rotation": 3}
    assert m.newer_pair_prefix(None, task, {"parts": 1}, original) is None
    assert calls == []
    (root / "part_000001.json").write_text("{}")
    assert m.newer_pair_prefix(None, task, {"parts": 1}, original) == "full_verifier_called"
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kind",
    [
        "manual",
        "wrong_owner",
        "checkpoint_corrupt",
        "head_started",
        "pending_update",
        "fragment_corrupt",
    ],
)
def test_unsafe_pause_rejected(paused, kind):
    task, owner, write = paused
    root = Path(task["run"])
    fit = root / "rotations/3/fits/PAIR_E"
    if kind == "manual":
        p = root / "EVENT_DOWNSTREAM_PAUSE_R3.json"
        d = json.loads(p.read_text())
        d["manual"] = True
        write(p, d)
    elif kind == "wrong_owner":
        owner["start_ticks"] += 1
    elif kind == "checkpoint_corrupt":
        (fit / "checkpoint_last.pt").write_bytes(b"changed")
    elif kind == "head_started":
        write(root / "rotations/3/fits/E_CTX/UPDATE_JOURNAL.json", {"completed_updates": 1})
    elif kind == "pending_update":
        p = fit / "UPDATE_JOURNAL.json"
        d = json.loads(p.read_text())
        d["pending_update_upper"] = 1
        write(p, d)
    else:
        (root / "rotations/3/refiner/E_CTX/parts/part_000000.npz").write_bytes(b"changed")
    assert not m.completed_pair_input_pause(task, owner)
