"""Fault injection and native head resume fixtures; zero optimizer updates or raw reads."""

import hashlib
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import psutil
import pytest

from e_jepa_ttc.efficient_context.garl_head import NativeHeadConfig
from e_jepa_ttc.simplex_t.model import TemporalConfig
from e_jepa_ttc.simplex_t.training import fit, load_checkpoint, state_digest
from e_jepa_ttc.simplex_t.work_budget import EngineWorkJournal, WorkBudget
from operational.efficient_context.common import Lease, atomic_json, digest, npz, read
from operational.efficient_context.garl_recovery import (
    completed_source,
    recover_head_transaction,
    runtime_measurements,
    verified_fragment,
)


@pytest.mark.parametrize("kind", ["missing", "orphan", "failed_atomic_temp", "committed"])
def test_fragment_requires_commit_receipt_and_reuses_without_rewriting(tmp_path, kind):
    path = tmp_path / "query00001.npz"
    expected = {"query": "train-fixture", "source": "s" * 64, "start": 128, "stop": 256}
    if kind in {"orphan", "committed"}:
        npz(path, features=np.ones((8, 3), np.float32))
    if kind == "failed_atomic_temp":
        path.with_suffix(".pending").write_bytes(b"incomplete fixture")
    if kind == "committed":
        atomic_json(path.with_suffix(".json"), {**expected, "sha256": digest(path)})
    before = path.read_bytes() if path.exists() else None
    assert verified_fragment(path, expected) == (kind == "committed")
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize("change", ["bytes", "missing_payload", "query", "source", "stop"])
def test_committed_fragment_corruption_is_refused_and_retained(tmp_path, change):
    path = tmp_path / "fragment.npz"
    expected = {"query": "train-fixture", "source": "s" * 64, "stop": 128}
    npz(path, features=np.ones((8, 3), np.float32))
    receipt = {**expected, "sha256": digest(path)}
    if change in {"query", "source", "stop"}:
        receipt[change] = "incorrect fixture"
    atomic_json(path.with_suffix(".json"), receipt)
    if change == "bytes":
        path.write_bytes(b"damaged fixture")
    elif change == "missing_payload":
        path.unlink()
    before = path.with_suffix(".json").read_bytes()
    with pytest.raises(ValueError, match="fragment"):
        verified_fragment(path, expected)
    assert path.with_suffix(".json").read_bytes() == before


def compact_fixture(out: Path) -> Path:
    folder = out / "garl/features/fold0/inner_oof"
    atomic_json(folder / "BINDING.json", {"fixture": "TRAIN only"})
    history = np.tile(np.arange(8, dtype=np.int64), (4, 1))
    features = np.arange(24, dtype=np.float32).reshape(8, 3) / 100
    npz(
        folder / "SOURCE.npz",
        features=features,
        history=history,
        times=np.zeros((4, 8, 4), np.float32),
        truth=np.full(4, 0.1, np.float32),
        mass=np.full(4, 0.25, np.float32),
    )
    (folder / "METADATA.csv").write_text("sample_token\nfixture0\nfixture1\nfixture2\nfixture3\n")
    atomic_json(
        folder / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "binding_sha256": digest(folder / "BINDING.json"),
            "source_sha256": digest(folder / "SOURCE.npz"),
            "metadata_sha256": digest(folder / "METADATA.csv"),
            "queries": 4,
            "optimizer_updates": 0,
        },
    )
    npz(
        out / "garl_heads/normalizers/fold0.npz",
        mean=np.zeros(3),
        scale=np.ones(3),
        ids_hash=np.asarray("a" * 64),
    )
    return folder


@pytest.mark.parametrize("change", ["none", "source", "metadata", "binding", "population"])
def test_compact_source_completion_checks_every_bound_output(tmp_path, change):
    folder = compact_fixture(tmp_path)
    binding = digest(folder / "BINDING.json")
    expected_queries = 4
    if change in {"source", "metadata"}:
        path = folder / ("SOURCE.npz" if change == "source" else "METADATA.csv")
        path.write_bytes(b"damaged fixture")
    elif change == "binding":
        binding = "b" * 64
    elif change == "population":
        expected_queries = 5
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    if change == "none":
        assert completed_source(folder, binding, expected_queries)["queries"] == 4
    else:
        with pytest.raises(ValueError, match="source"):
            completed_source(folder, binding, expected_queries)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


@pytest.mark.parametrize("length", [1, 8])
def test_actual_native_head_engine_zero_update_pause_and_resume_preserves_full_state(
    tmp_path, length
):
    from operational.efficient_context.garl_heads import numeric_binding, source

    folder = compact_fixture(tmp_path)
    before = digest(folder / "SOURCE.npz")
    native = source(SimpleNamespace(out=tmp_path), 0, "inner_oof", length)
    budget = WorkBudget(tmp_path / "fixture_work.json", {"fixture": 2500}, 0)
    checkpoint = tmp_path / "fixture_fit/checkpoint_last.pt"
    hashes = []
    for resume in (False, True):
        with numeric_binding():
            result = fit(
                native,
                cast(TemporalConfig, NativeHeadConfig()),
                checkpoint.parent,
                seed=7,
                freeze_sha256="f" * 64,
                resource_ok=lambda: False,
                resume=resume,
                journal=EngineWorkJournal(budget, "fixture"),
                device="cpu",
            )
        state = load_checkpoint(checkpoint)
        assert result["status"] == "PAUSED_RESOURCE" and result["completed_updates"] == 0
        assert state["optimizer"]["state"] == {} and not state["losses"]
        assert state["identity"]["config"]["feature_count"] == 3
        hashes.append(state_digest(state))
    assert hashes[0] == hashes[1] and digest(folder / "SOURCE.npz") == before
    assert not budget.path.exists()


@pytest.mark.parametrize("owner_kind", ["dead", "live", "other_host", "malformed", "wrong_ledger"])
def test_head_journal_lock_recovery_preserves_live_or_unprovable_owners(
    tmp_path, monkeypatch, owner_kind
):
    c = SimpleNamespace(out=tmp_path)
    path = tmp_path / "garl_heads/PHYSICAL_WORK.lock"
    owner = {"owner": "fixture-owner", "pid": 123456789, "host": socket.gethostname()}
    if owner_kind == "other_host":
        owner["host"] = "another-host"
    elif owner_kind == "malformed":
        owner["pid"] = "unprovable"
    atomic_json(path, owner)
    ledger = path.with_suffix(".json")
    atomic_json(
        ledger,
        {"schema": "wrong" if owner_kind == "wrong_ledger" else "simplex_t_physical_work_v1"},
    )
    ledger_before, lock_before = ledger.read_bytes(), path.read_bytes()
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: owner_kind == "live")
    with Lease(tmp_path):
        if owner_kind == "dead":
            recover_head_transaction(c)
            assert not path.exists()
            receipts = list((tmp_path / "garl_heads/recovery").glob("*.json"))
            assert len(receipts) == 1 and read(receipts[0])["owner"] == owner
        else:
            with pytest.raises((RuntimeError, ValueError)):
                recover_head_transaction(c)
            assert path.read_bytes() == lock_before
    assert ledger.read_bytes() == ledger_before


def test_head_journal_recovery_requires_own_live_campaign_lease(tmp_path):
    atomic_json(
        tmp_path / "garl_heads/PHYSICAL_WORK.lock",
        {"pid": 123456789, "host": socket.gethostname()},
    )
    with pytest.raises(FileNotFoundError):
        recover_head_transaction(SimpleNamespace(out=tmp_path))
    assert (tmp_path / "garl_heads/PHYSICAL_WORK.lock").exists()


def test_head_transaction_recovery_can_resume_after_archive_before_unlink_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "garl_heads/PHYSICAL_WORK.lock"
    atomic_json(path, {"owner": "fixture", "pid": 123456789, "host": socket.gethostname()})
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: False)
    unlink = Path.unlink

    def fail_lock_unlink(self, *args, **kwargs):
        if self == path:
            raise OSError("injected crash after recovery archive publication")
        return unlink(self, *args, **kwargs)

    with Lease(tmp_path):
        with monkeypatch.context() as partial:
            partial.setattr(Path, "unlink", fail_lock_unlink)
            with pytest.raises(OSError, match="injected crash"):
                recover_head_transaction(SimpleNamespace(out=tmp_path))
    archive = next((tmp_path / "garl_heads/recovery").glob("*.json"))
    value = read(archive)
    # Fixture represents the first observer from a previous process lifetime.
    value["writer_owner"] = {"pid": 987654321, "create_time": 1.0}
    atomic_json(archive, value)
    archived_before = archive.read_bytes()
    with Lease(tmp_path):
        recover_head_transaction(SimpleNamespace(out=tmp_path))
    assert not path.exists() and archive.read_bytes() == archived_before


def test_native_head_crash_recovery_reconciles_pending_work_and_restores_full_zero_state(
    tmp_path, monkeypatch
):
    from operational.efficient_context.garl_heads import numeric_binding, source

    compact_fixture(tmp_path)
    c = SimpleNamespace(out=tmp_path)
    native = source(c, 0, "inner_oof", 8)
    ledger = tmp_path / "garl_heads/PHYSICAL_WORK.json"
    budget = WorkBudget(ledger, {"fixture": 2500}, 0)
    kwargs = dict(seed=7, freeze_sha256="f" * 64, resource_ok=lambda: False, device="cpu")
    checkpoint = tmp_path / "fixture_fit/checkpoint_last.pt"
    with numeric_binding():
        fit(
            native,
            cast(TemporalConfig, NativeHeadConfig()),
            checkpoint.parent,
            journal=EngineWorkJournal(budget, "fixture"),
            **kwargs,
        )
    before = state_digest(load_checkpoint(checkpoint))
    # Fixture: process dies after reserving a chunk, before any optimizer execution.
    budget.transition("begin", "fixture", 0)
    atomic_json(
        ledger.with_suffix(".lock"),
        {"owner": "dead-fixture", "pid": 123456789, "host": socket.gethostname()},
    )
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: False)
    with Lease(tmp_path):
        recover_head_transaction(c)
        with numeric_binding():
            result = fit(
                native,
                cast(TemporalConfig, NativeHeadConfig()),
                checkpoint.parent,
                resume=True,
                journal=EngineWorkJournal(budget, "fixture"),
                **kwargs,
            )
    assert result["completed_updates"] == 0
    assert state_digest(load_checkpoint(checkpoint)) == before
    accounted = read(ledger)
    assert accounted["fits"]["fixture"]["pending"] is None
    assert accounted["accounting"]["scientific_saved_updates"] == 0
    assert accounted["accounting"]["scientific_uncertain_lost_upper"] == 100
    assert not load_checkpoint(checkpoint)["optimizer"]["state"]


@pytest.mark.parametrize("label", ["GARL_NATIVE", "GARL_H1", "GARL_H8"])
@pytest.mark.parametrize(
    "change", ["none", "query", "protocol", "truncated", "cost", "parity", "checksum"]
)
def test_native_runtime_resume_checks_binding_completeness_cost_and_parity(tmp_path, label, change):
    expected = {
        "runtime_protocol_sha256": "r" * 64,
        "producer_sha256": "p" * 64,
        "query": "train-fixture",
        "label": label,
        "block": "warm_block1",
    }
    regimes = ["R0", "R2_PREPARED_PRODUCER_AND_HEAD"]
    if label != "GARL_NATIVE":
        regimes.append("R2_HEAD_ONLY_CPU_FP32")
    rows = [
        {
            "query": expected["query"],
            "label": label,
            "block": expected["block"],
            "regime": r,
            "milliseconds": 1.0,
        }
        for r in regimes
    ]
    value = {
        **expected,
        "status": "PASSED",
        "measurements": rows,
        "feature_max_abs": 0,
        "phase_max_abs": 0,
        "repeated_output_max_abs": 0,
    }
    if change == "query":
        rows[0]["query"] = "wrong-fixture"
    elif change == "protocol":
        value["runtime_protocol_sha256"] = "changed"
    elif change == "truncated":
        rows.pop()
    elif change == "cost":
        rows[0]["milliseconds"] = -1
    elif change == "parity":
        value["phase_max_abs"] = 1e-3
    value["measurements_sha256"] = hashlib.sha256(
        json.dumps(rows, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if change == "checksum":
        rows[0]["milliseconds"] = 2.0
    receipt = tmp_path / "runtime.json"
    atomic_json(receipt, value)
    before = receipt.read_bytes()
    if change == "none":
        assert runtime_measurements(receipt, expected) == rows
    else:
        with pytest.raises(ValueError):
            runtime_measurements(receipt, expected)
    assert receipt.read_bytes() == before
