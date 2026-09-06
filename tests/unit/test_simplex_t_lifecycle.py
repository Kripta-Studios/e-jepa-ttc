"""Owner conflict and finite graph budgeting without training."""

import json

import pytest

from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, UpdateLedger


def test_lease_refuses_other_owner(tmp_path):
    path = tmp_path / "lease"
    with ExclusiveLease(path):
        with pytest.raises(FileExistsError), ExclusiveLease(path):
            pass
    assert not path.exists()


def test_no_stale_lease_removal(tmp_path):
    path = tmp_path / "lease"
    path.write_text(json.dumps(dict(owner="old", pid=-1)))
    with pytest.raises(FileExistsError), ExclusiveLease(path):
        pass
    assert path.exists()


def test_changed_owner_not_removed(tmp_path):
    path = tmp_path / "lease"
    with pytest.raises(RuntimeError, match="owner"), ExclusiveLease(path):
        path.write_text(json.dumps(dict(owner="different")))
    assert path.exists()


def test_ledger_refuses_new_arms_and_partial_completion(tmp_path):
    ledger = UpdateLedger(tmp_path / "ledger.json", {"canonical_fold0_seed7": 2500})
    ledger.transaction("reserve", "canonical_fold0_seed7")
    state = ledger.transaction("progress", "canonical_fold0_seed7", 100)
    assert state["fits"]["canonical_fold0_seed7"]["status"] == "PARTIAL"
    with pytest.raises(ValueError, match="unknown"):
        ledger.transaction("reserve", "invented")
    with pytest.raises(ValueError, match="nonmonotonic"):
        ledger.transaction("progress", "canonical_fold0_seed7", 99)
    ledger.transaction("technical", "qa", 1000)
    with pytest.raises(ValueError, match="budget"):
        ledger.transaction("technical", "qa", 1)
