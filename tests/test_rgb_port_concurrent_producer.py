from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any, cast

import pytest

import operational.rgb_port.train_producers as training
from operational.rgb_port_concurrent.producer import Runtime, installed


class _Inputs:
    def __init__(self) -> None:
        self.limit = 2 * 1024**3
        self.lock = Lock()
        self.reads = 7
        self.hits = 3
        self.cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.cache_bytes = 0


class _Source:
    population_size: int
    identity: Mapping[str, Any]
    frame_counts: Sequence[int]
    input_span_us: Sequence[int]

    def __init__(self) -> None:
        self._inputs = _Inputs()
        self.population_size = 1
        self.identity = {}
        self.frame_counts = (3,)
        self.input_span_us = (300_000,)

    def batch(self, indices: list[int], modality: str) -> None:
        del indices, modality


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime(tmp_path: Path, fit_id: str = "E_A5_MATCHED") -> tuple[Runtime, Any]:
    run = tmp_path / "run"
    fit = run / "fits" / fit_id
    fit.mkdir(parents=True)
    freeze_path = run / "CONCURRENT_FREEZE.json"
    _write(freeze_path, {"fixture": True})
    checkpoint = fit / "checkpoint_last.pt"
    checkpoint.write_bytes(b"immutable checkpoint")
    state = SimpleNamespace(
        completed=120,
        durable=100,
        identity_sha256="identity",
        path=checkpoint,
        pointer=fit / "CHECKPOINT_POINTER.json",
    )
    _write(
        state.pointer,
        {
            "schema": "rgb_port_checkpoint_pointer_v1",
            "completed_updates": 120,
            "version": "checkpoint_000120.pt",
            "checkpoint_sha256": _sha(checkpoint),
            "identity_sha256": "identity",
        },
    )
    freeze = {
        "identity_sha256": "freeze-identity",
        "source_sha256": {"producer.py": "digest"},
        "pipeline_freeze_path": str(run / "PIPELINE_FREEZE.json"),
        "origins": {
            "E_A5_MATCHED": {
                "completed_updates": 120,
                "checkpoint_sha256": _sha(checkpoint),
                "identity_sha256": "identity",
            },
            "E_C2F_MATCHED": {
                "completed_updates": 120,
                "checkpoint_sha256": _sha(checkpoint),
                "identity_sha256": "identity",
            },
        },
        "cache_policy": {
            "per_fit_max_bytes": {
                "E_A5_MATCHED": 512 * 1024**2,
                "E_C2F_MATCHED": 1024**3,
            },
            "graphs": {"E_A5_MATCHED": "V1_ONLY", "E_C2F_MATCHED": "DISABLED"},
        },
    }
    source = _Source()
    runtime = Runtime(
        cast(training.ProducerSource, source),
        SimpleNamespace(fit_id=fit_id),
        run,
        freeze_path,
        freeze,
    )
    return runtime, state


@pytest.mark.parametrize(
    ("fit_id", "limit", "mode"),
    [
        ("E_A5_MATCHED", 512 * 1024**2, "V1_ONLY"),
        ("E_C2F_MATCHED", 1024**3, "DISABLED"),
    ],
)
def test_runtime_applies_only_frozen_cache_and_graph_policy(
    tmp_path: Path, fit_id: str, limit: int, mode: str
) -> None:
    runtime, _ = _runtime(tmp_path, fit_id)
    assert runtime.inputs.limit == limit
    assert runtime.cache_limit == limit
    assert runtime._mode() == mode
    assert runtime.prior_cache_limit == 2 * 1024**3


def test_checkpoint_transaction_has_pending_and_exact_receipt(tmp_path: Path) -> None:
    runtime, state = _runtime(tmp_path)
    runtime.before_save(state)
    pending = json.loads(runtime.pending.read_text(encoding="utf-8"))
    assert pending["start_update"] == 100
    assert pending["end_update"] == 120
    runtime.after_save(state)
    assert not runtime.pending.exists()
    receipt = json.loads(
        (runtime.receipt_dir / "checkpoint_000120.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "COMPLETE"
    assert receipt["checkpoint_sha256"] == _sha(state.path)
    runtime.after_restore(state)


def test_pending_repair_rejects_changed_execution_binding(tmp_path: Path) -> None:
    runtime, state = _runtime(tmp_path)
    runtime.before_save(state)
    pending = json.loads(runtime.pending.read_text(encoding="utf-8"))
    pending["cache_limit_bytes"] += 1
    _write(runtime.pending, pending)
    with pytest.raises(RuntimeError, match="pending proof"):
        runtime.after_restore(state)


def test_later_c2f_checkpoint_adopts_only_validated_v2_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import operational.rgb_port_pipeline_v2.contracts as pipeline_contracts
    import operational.rgb_port_pipeline_v2.receipts as pipeline_receipts

    runtime, state = _runtime(tmp_path, "E_C2F_MATCHED")
    Path(runtime.freeze["pipeline_freeze_path"]).write_text("{}", encoding="utf-8")
    state.completed = 121
    _write(
        state.pointer,
        {
            "schema": "rgb_port_checkpoint_pointer_v1",
            "completed_updates": 121,
            "version": "checkpoint_000121.pt",
            "checkpoint_sha256": _sha(state.path),
            "identity_sha256": "identity",
        },
    )
    timing = runtime.fit / "pipeline_checkpoints" / "timings_000121.json"
    _write(timing, {"schema": "rgb_port_pipeline_timings_v2"})
    _write(
        runtime.fit / "pipeline_checkpoints" / "checkpoint_000121.json",
        {
            "schema": "rgb_port_pipeline_checkpoint_receipt_v2",
            "status": "COMPLETE",
            "fit_id": "E_C2F_MATCHED",
            "completed_updates": 121,
            "checkpoint_version": "checkpoint_000121.pt",
            "checkpoint_sha256": _sha(state.path),
            "checkpoint_identity_sha256": "identity",
            "timings_snapshot": timing.name,
            "timings_sha256": _sha(timing),
        },
    )
    monkeypatch.setattr(pipeline_contracts, "validate_pipeline_freeze", lambda _path: {})
    monkeypatch.setattr(
        pipeline_receipts,
        "validate_fit_lineage",
        lambda *_args, **_kwargs: None,
    )
    runtime.after_restore(state)
    adopted = json.loads(
        (runtime.receipt_dir / "checkpoint_000121.json").read_text(encoding="utf-8")
    )
    assert adopted["status"] == "ADOPTED_V2_LINEAGE"
    assert adopted["adopted_pipeline_v2_receipt_sha256"] == _sha(
        runtime.fit / "pipeline_checkpoints" / "checkpoint_000121.json"
    )


def test_live_c2f_gap_adopts_validated_v1_metrics_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import operational.rgb_port_acceleration.queue as acceleration_queue

    runtime, state = _runtime(tmp_path, "E_C2F_MATCHED")
    state.completed = 121
    _write(
        state.pointer,
        {
            "schema": "rgb_port_checkpoint_pointer_v1",
            "completed_updates": 121,
            "version": "checkpoint_000121.pt",
            "checkpoint_sha256": _sha(state.path),
            "identity_sha256": "identity",
        },
    )
    root = runtime.fit / "acceleration_checkpoints"
    snapshot = root / "runtime_000121.json"
    _write(snapshot, {"schema": "rgb_port_acceleration_runtime_v1"})
    native_receipt = root / "checkpoint_000121.json"
    _write(
        native_receipt,
        {
            "schema": "rgb_port_acceleration_checkpoint_receipt_v1",
            "status": "COMPLETE",
            "fit_id": "E_C2F_MATCHED",
            "completed_updates": 121,
            "checkpoint_version": "checkpoint_000121.pt",
            "checkpoint_sha256": _sha(state.path),
            "checkpoint_identity_sha256": "identity",
            "runtime_snapshot": snapshot.name,
            "runtime_receipt_sha256": _sha(snapshot),
            "mode": "batched_metrics_only",
            "backend": None,
            "shape": None,
        },
    )
    validated: list[str] = []
    monkeypatch.setattr(
        acceleration_queue,
        "_validate_runtime_receipts",
        lambda _run, *, require_endpoint, fit_id: validated.append(
            f"{fit_id}:{require_endpoint}"
        ),
    )
    runtime.after_restore(state)
    adopted = json.loads(
        (runtime.receipt_dir / "checkpoint_000121.json").read_text(encoding="utf-8")
    )
    assert validated == ["E_C2F_MATCHED:False"]
    assert adopted["status"] == "ADOPTED_V1_LINEAGE"
    assert adopted["adopted_acceleration_receipt_sha256"] == _sha(native_receipt)


def test_installed_wraps_restore_and_save_without_hot_path_patches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, state = _runtime(tmp_path)
    calls: list[str] = []
    original_loss = training._producer_loss

    def restore(_state: Any, *_args: Any, **_kwargs: Any) -> dict[str, int]:
        calls.append("native_restore")
        return {"position": 120}

    def save(_state: Any, *_args: Any, **_kwargs: Any) -> None:
        calls.append("native_save")

    monkeypatch.setattr(training.ProducerCheckpoint, "restore", restore)
    monkeypatch.setattr(training.ProducerCheckpoint, "save", save)
    with installed(runtime):
        assert training._producer_loss is original_loss
        checkpoint_api = cast(Any, training.ProducerCheckpoint)
        assert checkpoint_api.restore(state) == {"position": 120}
        checkpoint_api.save(state)
    assert calls == ["native_restore", "native_save"]
    assert (runtime.receipt_dir / "checkpoint_000120.json").is_file()


def test_cli_help_is_executable() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "operational.rgb_port_concurrent.producer", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--concurrent-freeze" in result.stdout
