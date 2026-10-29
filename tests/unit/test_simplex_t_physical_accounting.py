"""Real journal transitions only; no optimizer or scientific fit is executed."""

import copy

import pytest

from e_jepa_ttc.simplex_t.physical_accounting import audit_physical_work
from e_jepa_ttc.simplex_t.work_budget import WorkBudget


@pytest.mark.parametrize("mode", ["safe", "crash", "pending"])
def test_physical_bounds_preserve_journal(tmp_path, mode):
    path = tmp_path / "journal.json"
    budget = WorkBudget(path, {"fit": 2500}, 745)
    state = budget.transition("begin", "fit", 0)
    if mode != "pending":
        state = budget.transition(
            "checkpoint" if mode == "safe" else "recover", "fit", 37, checkpoint_sha256="a" * 64
        )
    original = path.read_bytes()
    result = audit_physical_work(
        state, expected_graph={"fit": 2500}, technical_reserved=745, resource_ok=lambda: True
    )
    assert result["saved_updates"] == (0 if mode == "pending" else 37)
    assert result["recorded_work_upper"] == (37 if mode == "safe" else 100)
    assert result["possible_lost_updates_upper"] == (63 if mode == "crash" else 0)
    assert result["pending_updates_upper"] == (100 if mode == "pending" else 0)
    assert result["journal_endpoint_fits"] == 0
    assert path.read_bytes() == original


@pytest.mark.parametrize("fault", ["summary", "fit", "event", "graph", "pause"])
def test_reject_tampered_or_unavailable_accounting(tmp_path, fault):
    state = WorkBudget(tmp_path / "journal.json", {"fit": 2500}, 745).transition("begin", "fit", 0)
    changed = copy.deepcopy(state)
    if fault == "summary":
        changed["accounting"]["scientific_saved_updates"] = 100
    elif fault == "fit":
        changed["fits"]["fit"]["completed"] = 100
    elif fault == "event":
        changed["events"].append(changed["events"][0])
    elif fault == "graph":
        changed["graph"] = {"other": 2500}
    with pytest.raises((ValueError, InterruptedError)):
        audit_physical_work(
            changed,
            expected_graph={"fit": 2500},
            technical_reserved=745,
            resource_ok=lambda: fault != "pause",
        )
