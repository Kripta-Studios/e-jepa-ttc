"""Engine/journal integration without any optimizer steps or raw data reads."""

import pytest
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig
from e_jepa_ttc.simplex_t.training import atomic_checkpoint, fit, load_checkpoint
from e_jepa_ttc.simplex_t.work_budget import EngineWorkJournal, WorkBudget


class NeverRead:
    population = 1
    identity_sha256 = "a" * 64

    def gather(self, query_ids):
        raise AssertionError("resource pause must precede input gathering")


def test_engine_publishes_zero_checkpoint_before_resource_pause_and_can_resume(tmp_path):
    budget = WorkBudget(tmp_path / "budget.json", {"fit": 2500}, 605)
    for resume in (False, True):
        result = fit(
            NeverRead(),
            TemporalConfig(),
            tmp_path / "fit",
            seed=7,
            freeze_sha256="b" * 64,
            resource_ok=lambda: False,
            resume=resume,
            journal=EngineWorkJournal(budget, "fit"),
        )
        assert result["status"] == "PAUSED_RESOURCE"
        assert result["completed_updates"] == 0
        assert not result["scientific_endpoint"]
        state = load_checkpoint(tmp_path / "fit/checkpoint_last.pt")
        assert state["optimizer"]["state"] == {}
    assert not budget.path.exists()  # No block was reserved or executed.


def test_reservation_failure_precedes_source_and_optimizer_work(tmp_path, monkeypatch):
    budget = WorkBudget(tmp_path / "budget.json", {"fit": 2500}, 605)

    def fail(*args, **kwargs):
        raise OSError("injected journal publication failure")

    monkeypatch.setattr(budget, "transition", fail)
    with pytest.raises(OSError, match="journal publication"):
        fit(
            NeverRead(),
            TemporalConfig(),
            tmp_path / "fit",
            seed=7,
            freeze_sha256="b" * 64,
            resource_ok=lambda: True,
            journal=EngineWorkJournal(budget, "fit"),
        )
    state = load_checkpoint(tmp_path / "fit/checkpoint_last.pt")
    assert state["completed_updates"] == 0
    assert state["optimizer"]["state"] == {}


@pytest.mark.parametrize("change", ["none", "status", "model", "optimizer", "rng", "loss"])
def test_resume_binds_actual_state_not_just_update_count(tmp_path, change):
    # Synthetic checkpoint transitions, not evidence of an executed optimizer update.
    path = tmp_path / "checkpoint.pt"
    state = dict(
        completed_updates=0,
        status="IN_PROGRESS",
        losses=[],
        sampler_hashes=[],
        model={"weight": torch.ones(1)},
        optimizer={"state": {}},
        sampler_rng=torch.tensor([1]),
        torch_rng=torch.tensor([2]),
        identity={"seed": 7},
    )
    atomic_checkpoint(path, state)
    budget = WorkBudget(tmp_path / "budget.json", {"fit": 2500}, 605)
    journal = EngineWorkJournal(budget, "fit")
    journal.start(0, path)
    journal.before_update(0)
    state.update(completed_updates=1, losses=[1.0], sampler_hashes=["synthetic"])
    atomic_checkpoint(path, state)
    journal.checkpoint_saved(1, path)
    if change == "status":
        state["status"] = "PAUSED_RESOURCE"
    elif change == "model":
        state["model"]["weight"] = torch.zeros(1)
    elif change == "optimizer":
        state["optimizer"]["state"] = {"different": 1}
    elif change == "rng":
        state["sampler_rng"] = torch.tensor([9])
    elif change == "loss":
        state["losses"] = [2.0]
    if change != "none":
        atomic_checkpoint(path, state)
    before = budget.path.read_bytes()
    restarted = EngineWorkJournal(budget, "fit")
    if change in {"none", "status"}:
        restarted.start(1, path)
        assert restarted.saved == 1
    else:
        with pytest.raises(ValueError, match="training state differs"):
            restarted.start(1, path)
    assert budget.path.read_bytes() == before
