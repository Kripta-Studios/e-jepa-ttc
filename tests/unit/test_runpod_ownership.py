"""Prove future-fit reservation is durable, fail-closed and isolated."""

import json

import pytest

from operational.runpod_migration.ownership import reserve, verify_claim


def fixture(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "SCIENTIFIC_CONFIG.json").write_text("authority")
    active = run / "rotations/2/fits/E_A5/PROGRESS.json"
    active.parent.mkdir(parents=True)
    active.write_text('{"completed_updates":1234}')
    return run, tmp_path / "claim.json", active


def test_reservation_is_real_and_does_not_touch_active_fit(tmp_path):
    run, receipt, active = fixture(tmp_path)
    before = active.read_bytes()
    first = reserve(run, receipt)
    assert (run / "rotations/4/fits/R_A5/STOP_REQUEST").exists()
    assert verify_claim(run, receipt) == first
    assert reserve(run, receipt) == first
    assert active.read_bytes() == before


def test_marker_removed_or_changed_rejects_claim(tmp_path):
    run, receipt, _ = fixture(tmp_path)
    reserve(run, receipt)
    (run / "rotations/4/fits/R_A5/STOP_REQUEST").write_text("other")
    with pytest.raises(ValueError):
        verify_claim(run, receipt)


def test_preexisting_fit_cannot_be_claimed(tmp_path):
    run, receipt, _ = fixture(tmp_path)
    folder = run / "rotations/4/fits/R_A5"
    folder.mkdir(parents=True)
    (folder / "PROGRESS.json").write_text('{"completed_updates":1}')
    with pytest.raises(ValueError):
        reserve(run, receipt)
    assert not receipt.exists()


def test_local_zero_update_pause_allowed_but_updates_rejected(tmp_path):
    run, receipt, _ = fixture(tmp_path)
    reserve(run, receipt)
    path = run / "rotations/4/fits/R_A5/CHECKPOINT_RECEIPT.json"
    path.write_text(json.dumps({"completed_updates": 0}))
    verify_claim(run, receipt)
    path.write_text(json.dumps({"completed_updates": 1}))
    with pytest.raises(ValueError):
        verify_claim(run, receipt)
