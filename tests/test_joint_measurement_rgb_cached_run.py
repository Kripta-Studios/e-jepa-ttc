"""The resident-input overlay must witness restore before any optimizer updates."""

import pytest
import torch

from operational.joint_measurement import rgb_cached_run as runtime
from operational.range_transfer.common import read


@pytest.mark.parametrize("corrupt", [False, True])
def test_restore_witness_detects_changed_state(tmp_path, monkeypatch, corrupt):
    monkeypatch.setattr(runtime, "OUT", tmp_path / "runtime")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    generator = torch.Generator().manual_seed(7)
    identity = {"fit_id": "test", "identity_sha256": "test"}
    state = runtime.ProducerCheckpoint(tmp_path / "fit", identity, 10)
    state.save(model, optimizer, scheduler, generator, {"curve": []}, status="PAUSED")
    original = runtime.ProducerCheckpoint.restore

    def altered_restore(self, *args):
        cursor = original(self, *args)
        if corrupt:
            with torch.no_grad():
                args[0].weight.add_(1)
        return cursor

    monkeypatch.setattr(runtime.ProducerCheckpoint, "restore", altered_restore)
    verifier = runtime.VerifiedCheckpoint(tmp_path / "fit", identity, 10)
    if corrupt:
        with pytest.raises(ValueError, match="full-state restore failed"):
            verifier.restore(model, optimizer, scheduler, generator)
    else:
        assert verifier.restore(model, optimizer, scheduler, generator) == {"curve": []}
    result = read(tmp_path / "runtime/restores/test_0.json")
    assert result["status"] == ("FAIL" if corrupt else "PASS")
    assert result["optimizer_updates_in_check"] == 0
    assert read(tmp_path / "fit/UPDATE_JOURNAL.json")["completed_updates"] == 0
