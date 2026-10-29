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


def test_completed_previous_arm_cannot_release_active_arms_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "OUT", tmp_path)
    (tmp_path / "REGISTRATION.json").write_text("{}")
    monkeypatch.setattr(runtime, "admission", lambda: {"cache_limit_bytes": 6 * 1024**3})
    monkeypatch.setattr(runtime.rgb_run, "require_authorization", lambda: None)
    instances = []

    class Dataset:
        def __init__(self, role, *, limit_bytes):
            assert role == "P" and limit_bytes == 6 * 1024**3
            self.releases = 0
            instances.append(self)

        def counters(self):
            return {"loaded_shards": 100}

        def shutdown(self):
            self.releases += 1

    monkeypatch.setattr(runtime, "CachedRGBPairDataset", Dataset)
    calls = 0

    def original_fit(fit_id):
        nonlocal calls
        if fit_id == "FT":
            return {"status": "COMPLETE", "completed_updates": 12000}
        runtime.rgb_train.RGBPairDataset("P")
        calls += 1
        return {
            "status": "COMPLETE" if calls == 3 else "PAUSED",
            "completed_updates": 12000 if calls == 3 else calls * 100,
        }

    monkeypatch.setattr(runtime.rgb_run, "fit", original_fit)

    def driver(*, once):
        assert not once
        for _ in range(3):
            runtime.rgb_run.fit("FT")
            if instances:
                assert len(instances) == 1 and instances[0].releases == 0
            runtime.rgb_run.fit("FROZEN")

    monkeypatch.setattr(runtime.rgb_run, "run", driver)
    runtime.run()
    assert len(instances) == 1 and instances[0].releases == 1
    assert read(tmp_path / "CACHE_STATE.json")["fit_id"] == "FROZEN"
