"""R3 is an independent assignment; it cannot reuse R4's ownership or budget scope."""

from datetime import UTC, datetime, timedelta

import pytest

from operational.runpod_migration.ownership_multi import reserve as reserve_multi
from operational.runpod_migration.ownership_multi import verify_claim as verify_multi
from operational.runpod_migration.ownership_r3 import reserve, verify_claim
from scripts.runpod_scientific_overlay_v5 import validate_budget


def test_r3_reserves_its_own_fit_without_touching_r4(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "SCIENTIFIC_CONFIG.json").write_text("authority")
    r4 = run / "rotations/4/fits/R_A5/STOP_REQUEST"
    r4.parent.mkdir(parents=True)
    r4.write_text("existing R4 claim")
    receipt = tmp_path / "r3.json"
    reserve(run, receipt)
    assert verify_claim(run, receipt)["tasks"] == [{"rotation": 3, "fit": "R_A5"}]
    assert (run / "rotations/3/fits/R_A5/STOP_REQUEST").exists()
    assert r4.read_text() == "existing R4 claim"


def test_r3_rejects_r4_assignment():
    deadline = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    budget = {
        "schema": "ejepa_cloud_budget_authorization_v1",
        "maximum_total_usd": 24,
        "execution_deadline_utc": deadline,
    }
    assignment = {
        "maximum_compute_usd": 24,
        "execution_authorized": True,
        "execution_deadline_utc": deadline,
        "tasks": [{"rotation": 3, "fit": "R_A5"}],
    }
    validate_budget(assignment, budget)
    with pytest.raises(ValueError):
        validate_budget({**assignment, "tasks": [{"rotation": 4, "fit": "R_A5"}]}, budget)


@pytest.mark.parametrize("rotation", [2, 3])
def test_multi_claim_preserves_active_event_fit(tmp_path, rotation):
    run = tmp_path / "run"
    run.mkdir()
    (run / "SCIENTIFIC_CONFIG.json").write_text("authority")
    active = run / "rotations/2/fits/E_A5/PROGRESS.json"
    active.parent.mkdir(parents=True)
    active.write_text('{"completed_updates":1234}')
    receipt = tmp_path / "claim.json"
    reserve_multi(run, receipt, rotation)
    assert verify_multi(run, receipt, rotation)["tasks"] == [{"rotation": rotation, "fit": "R_A5"}]
    assert active.read_text() == '{"completed_updates":1234}'
