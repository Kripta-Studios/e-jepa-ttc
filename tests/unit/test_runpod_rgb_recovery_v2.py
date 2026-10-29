"""Recovery heartbeat is emitted by actual queue work, with unchanged exceptions."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.runpod_rgb_recovery_v2 import install_queue_heartbeat


def test_cache_resource_pause_keeps_current_worker_entry_receipt(tmp_path: Path) -> None:
    receipts = []

    def queue(run, rotation, external_guard):
        external_guard()
        raise RuntimeError("resource reserve unavailable")

    module = SimpleNamespace(
        run_rgb_downstream=queue, _receipt=lambda *a, **k: receipts.append((a, k))
    )
    install_queue_heartbeat(module, clock=lambda: 10)
    assert not receipts
    with pytest.raises(RuntimeError, match="resource reserve"):
        module.run_rgb_downstream(tmp_path, 4)
    assert len(receipts) == 1
    assert receipts[0][0][1:3] == ("RGB_STAGE_ACTIVE", "RUNNING")
    assert receipts[0][1]["optimizer_updates_inferred"] is False


def test_head_guard_heartbeat_preserves_external_guard_and_result(tmp_path: Path) -> None:
    times = iter((0, 10, 31))
    receipts, guards = [], []

    def queue(run, rotation, external_guard):
        external_guard()
        external_guard()
        return {"stage": "PAIR_R", "status": "COMPLETE"}

    module = SimpleNamespace(
        run_rgb_downstream=queue, _receipt=lambda *a, **k: receipts.append((a, k))
    )
    install_queue_heartbeat(module, clock=lambda: next(times))
    result = module.run_rgb_downstream(tmp_path, 3, external_guard=lambda: guards.append(True))
    assert len(receipts) == 2
    assert len(guards) == 3
    assert result == {"stage": "PAIR_R", "status": "COMPLETE"}
