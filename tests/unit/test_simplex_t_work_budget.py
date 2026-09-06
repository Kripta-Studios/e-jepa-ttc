"""Crash accounting state transitions only; no optimizer is created or updated."""

import json

import pytest

from e_jepa_ttc.simplex_t.work_budget import WorkBudget


def test_safe_partial_checkpoint_does_not_charge_unexecuted_work(tmp_path):
    budget = WorkBudget(tmp_path / "work.json", {"fit": 2500}, 605)
    budget.transition("begin", "fit", 0)
    state = budget.transition("checkpoint", "fit", 37, checkpoint_sha256="a" * 64)
    assert state["accounting"]["scientific_saved_updates"] == 37
    assert state["accounting"]["scientific_uncertain_lost_upper"] == 0
    state = budget.transition("begin", "fit", 37)
    assert state["fits"]["fit"]["pending"] == [37, 100]


def test_crash_retains_possible_work_and_does_not_claim_it_as_executed(tmp_path):
    path = tmp_path / "work.json"
    WorkBudget(path, {"fit": 2500}, 605).transition("begin", "fit", 0)
    restarted = WorkBudget(path, {"fit": 2500}, 605)
    with pytest.raises(ValueError, match="explicit checkpoint recovery"):
        restarted.transition("begin", "fit", 0)
    state = restarted.transition("recover", "fit", 0, checkpoint_sha256="a" * 64)
    assert state["accounting"]["scientific_saved_updates"] == 0
    assert state["accounting"]["scientific_uncertain_lost_upper"] == 100
    assert state["accounting"]["full_graph_physical_work_upper"] == 3205
    restarted.transition("begin", "fit", 0)


def test_crash_after_checkpoint_before_journal_settlement(tmp_path):
    budget = WorkBudget(tmp_path / "work.json", {"fit": 2500}, 605)
    budget.transition("begin", "fit", 0)
    state = budget.transition("recover", "fit", 100, checkpoint_sha256="a" * 64)
    assert state["accounting"]["scientific_saved_updates"] == 100
    assert state["accounting"]["scientific_uncertain_lost_upper"] == 0


def test_refuse_changed_budget_and_out_of_chunk_checkpoint(tmp_path):
    path = tmp_path / "work.json"
    budget = WorkBudget(path, {"fit": 2500}, 605)
    budget.transition("begin", "fit", 0)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="contract changed"):
        WorkBudget(path, {"fit": 2500}, 606).transition("begin", "fit", 0)
    with pytest.raises(ValueError, match="outside reserved chunk"):
        budget.transition("checkpoint", "fit", 101, checkpoint_sha256="a" * 64)
    assert path.read_bytes() == original


def test_absolute_cap_refuses_further_crash_recovery_without_erasing_pending_work(tmp_path):
    path = tmp_path / "boundary_fixture.json"
    budget = WorkBudget(path, {f"fit{i}": 2500 for i in range(84)}, 1000)
    state = budget.transition("begin", "fit0", 0)
    # Boundary fixture, not evidence that any real updates were executed.
    state["fits"]["fit0"]["uncertain_lost_upper"] = 39000
    path.write_text(json.dumps(state), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="exceeds250000"):
        budget.transition("recover", "fit0", 0, checkpoint_sha256="a" * 64)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="explicit checkpoint recovery"):
        budget.transition("begin", "fit0", 0)
