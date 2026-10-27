"""RGB recovery V3 records guard attempts before a guard can raise."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runpod_rgb_recovery_v3 import _write_sidecar, digest, install_queue_heartbeat


def test_first_external_guard_failure_keeps_current_truthful_receipt(
    tmp_path: Path,
) -> None:
    receipts: list[tuple[tuple, dict]] = []
    guards = 0

    def queue(*_args, **_kwargs):
        pytest.fail("queue must not run after the entry guard fails")

    def guard() -> None:
        nonlocal guards
        guards += 1
        raise RuntimeError("first resource guard failed")

    module = SimpleNamespace(
        run_rgb_downstream=queue,
        _receipt=lambda *args, **kwargs: receipts.append((args, kwargs)),
    )
    install_queue_heartbeat(module, clock=lambda: 10.0)

    with pytest.raises(RuntimeError, match="first resource guard failed"):
        module.run_rgb_downstream(tmp_path, 2, external_guard=guard)

    assert guards == 1
    assert len(receipts) == 1
    args, details = receipts[0]
    assert args[1:3] == ("RGB_STAGE_ACTIVE", "RUNNING")
    assert details["heartbeat_event"] == "ENTRY_OR_GUARD_ATTEMPT"
    assert details["receipt_precedes_optional_external_guard"] is True
    assert details["external_guard_success_inferred"] is False
    assert details["optimizer_updates_inferred"] is False


def test_heartbeat_frequency_guard_calls_and_result_are_preserved(tmp_path: Path) -> None:
    times = iter((0.0, 10.0, 31.0))
    receipts: list[tuple[tuple, dict]] = []
    guards: list[bool] = []

    def queue(run, rotation, external_guard):
        external_guard()
        external_guard()
        return {"stage": "PAIR_R", "status": "COMPLETE"}

    module = SimpleNamespace(
        run_rgb_downstream=queue,
        _receipt=lambda *args, **kwargs: receipts.append((args, kwargs)),
    )
    install_queue_heartbeat(module, clock=lambda: next(times))
    result = module.run_rgb_downstream(
        tmp_path, 3, external_guard=lambda: guards.append(True)
    )

    assert len(receipts) == 2
    assert len(guards) == 3
    assert all(item[1]["optimizer_updates_inferred"] is False for item in receipts)
    assert result == {"stage": "PAIR_R", "status": "COMPLETE"}


def test_v3_sidecar_binds_v3_v2_and_migration_bytes(tmp_path: Path) -> None:
    source, v2 = tmp_path / "v3.py", tmp_path / "v2.py"
    source.write_bytes(b"v3")
    v2.write_bytes(b"v2")
    migration = tmp_path / "MIGRATION_BINDING.json"
    migration.write_bytes(b"migration")

    _write_sidecar(tmp_path, source, v2)

    import json

    value = json.loads((tmp_path / "RGB_OPERATIONAL_RECOVERY_V3.json").read_text())
    assert value["source_sha256"] == digest(source)
    assert value["frozen_v2_sha256"] == digest(v2)
    assert value["migration_binding_sha256"] == digest(migration)
    assert value["optimizer_updates_inferred"] is False
