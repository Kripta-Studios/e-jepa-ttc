from pathlib import Path

import pytest

from e_jepa_ttc.training.campaign_budget import CampaignBudget


def test_replicas_and_resume_share_original_deadline(tmp_path: Path) -> None:
    path = tmp_path / "replications.json"
    seed13 = CampaignBudget(path, hours=48, clock=lambda: 100.0)
    seed23 = CampaignBudget(path, hours=48, clock=lambda: 100.0 + 47 * 3600)
    assert seed13.deadline == seed23.deadline
    seed23.check()
    resumed = CampaignBudget(path, hours=48, clock=lambda: 100.0 + 48 * 3600)
    with pytest.raises(TimeoutError):
        resumed.check()
    with pytest.raises(ValueError, match="identity"):
        CampaignBudget(path, hours=96)


def test_clock_rollback_is_integrity_failure(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    CampaignBudget(path, hours=1, clock=lambda: 100.0)
    resumed = CampaignBudget(path, hours=1, clock=lambda: 99.0)
    with pytest.raises(ValueError, match="clock"):
        resumed.check()
