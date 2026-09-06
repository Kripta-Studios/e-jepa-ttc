"""Engine/journal integration without any optimizer steps or raw data reads."""

import pytest

from e_jepa_ttc.simplex_t.model import TemporalConfig
from e_jepa_ttc.simplex_t.training import fit, load_checkpoint
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
