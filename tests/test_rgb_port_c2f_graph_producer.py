from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import operational.rgb_port.train_producers as training
import operational.rgb_port_c2f_graph.contracts as contracts
from operational.rgb_port_c2f_graph.producer import installed


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lineage(tmp_path: Path, update: int) -> tuple[Path, Path, dict[str, Any]]:
    run = tmp_path / "run"
    fit = run / "fits" / contracts.FIT_ID
    version = fit / "checkpoint_versions" / f"checkpoint_{update:06d}.pt"
    version.parent.mkdir(parents=True)
    version.write_bytes(f"checkpoint-{update}".encode())
    _write(
        fit / "CHECKPOINT_POINTER.json",
        {
            "version": version.name,
            "completed_updates": update,
            "checkpoint_sha256": _sha(version),
            "identity_sha256": "model-identity",
        },
    )
    freeze_path = run / "C2F_GRAPH_FREEZE.json"
    freeze_path.write_text("{}", encoding="utf-8")
    freeze = {
        "identity_sha256": "graph-freeze",
        "source_sha256": {"producer.py": "digest"},
        "torch_backend_source_sha256": {"backend.py": "digest"},
        "origin": {
            "completed_updates": 100,
            "checkpoint_sha256": _sha(version) if update == 100 else "origin-sha",
            "identity_sha256": "model-identity",
        },
    }
    return run, freeze_path, freeze


def test_lineage_accepts_exact_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, freeze_path, freeze = _lineage(tmp_path, 100)
    monkeypatch.setattr(contracts, "validate_c2f_graph_freeze", lambda _path: freeze)
    contracts.validate_c2f_graph_lineage(run, freeze_path)


def test_lineage_requires_typed_proof_after_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, freeze_path, freeze = _lineage(tmp_path, 200)
    monkeypatch.setattr(contracts, "validate_c2f_graph_freeze", lambda _path: freeze)
    with pytest.raises(FileNotFoundError, match="receipt or pending"):
        contracts.validate_c2f_graph_lineage(run, freeze_path)
    fit = run / "fits" / contracts.FIT_ID
    pointer = json.loads((fit / "CHECKPOINT_POINTER.json").read_text(encoding="utf-8"))
    common = {
        "fit_id": contracts.FIT_ID,
        "c2f_graph_freeze_sha256": "graph-freeze",
        "c2f_graph_freeze_file_sha256": _sha(freeze_path),
        "source_sha256": freeze["source_sha256"],
        "torch_backend_source_sha256": freeze["torch_backend_source_sha256"],
        "backend": "cudagraphs",
        "shape": {"batch_size": 32, "frames": 3},
        "optimizer_updates": 0,
    }
    _write(
        fit / "c2f_graph_checkpoints/PENDING_EXECUTION_RECEIPT.json",
        {
            "schema": "rgb_port_c2f_graph_pending_v1",
            "status": "PENDING",
            **common,
            "start_update": 100,
            "end_update": 200,
            "checkpoint_version": pointer["version"],
            "checkpoint_identity_sha256": pointer["identity_sha256"],
        },
    )
    contracts.validate_c2f_graph_lineage(run, freeze_path)


def test_installed_preserves_native_transaction_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeRuntime:
        def after_restore(self, *_args: Any) -> dict[str, int]:
            calls.append("graph_restore")
            return {"position": 1}

        def before_save(self, _state: Any) -> None:
            calls.append("graph_pending")

        def after_save(self, _state: Any) -> None:
            calls.append("graph_receipt")

    def restore(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        calls.append("native_restore")
        return {"position": 0}

    def save(*_args: Any, **_kwargs: Any) -> None:
        calls.append("native_save")

    monkeypatch.setattr(training.ProducerCheckpoint, "restore", restore)
    monkeypatch.setattr(training.ProducerCheckpoint, "save", save)
    api = cast(Any, training.ProducerCheckpoint)
    with installed(cast(Any, FakeRuntime())):
        api.restore(SimpleNamespace(), None, None, None, None)
        api.save(SimpleNamespace())
    assert calls == [
        "native_restore",
        "graph_restore",
        "graph_pending",
        "native_save",
        "graph_receipt",
    ]


@pytest.mark.parametrize(
    "module",
    [
        "operational.rgb_port_c2f_graph.contracts",
        "operational.rgb_port_c2f_graph.producer",
    ],
)
def test_cli_help(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

