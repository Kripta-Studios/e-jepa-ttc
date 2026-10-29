"""Removing time caps cannot remove scientific, resource or identity safeguards."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from e_jepa_ttc.artifacts import time_cap_amendment as amendment
from e_jepa_ttc.artifacts.hashing import compute_file_hash, sign_artifact
from e_jepa_ttc.artifacts.training_authorization import validate_training_authorization
from e_jepa_ttc.training import campaign_budget
from e_jepa_ttc.training.campaign_budget import CampaignBudget


def _signed(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sign_artifact(value)), encoding="utf-8")


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    _signed(root / "TRAINING_LOCK.json", {"training_commit": "old"})
    log = root / "qa/log.txt"
    log.parent.mkdir(exist_ok=True)
    log.write_text("passed", encoding="utf-8")
    lock_sha = compute_file_hash(str(root / "TRAINING_LOCK.json"))
    qa_path = root / "qa/TIME_AMENDMENT_QA.json"
    _signed(
        qa_path,
        {
            "status": "passed",
            "execution_commit": "new",
            "training_lock_sha256": lock_sha,
            "evidence": {"log": {"path": str(log), "sha256": compute_file_hash(str(log))}},
        },
    )
    changed = "src/e_jepa_ttc/training/campaign_budget.py"
    value = {
        "execution_commit": "new",
        "original_training_commit": "old",
        "training_lock_sha256": lock_sha,
        "policy": "disable_wall_time_caps_only",
        "scope": ["stage63", "stage64_seed7", "stage64_replications", "stage65"],
        "fixed_endpoint_updates": 3000,
        "resource_margins_unchanged": True,
        "user_request": "remove wall-time limits",
        "changed_files": [changed],
        "implementation_diff_sha256": hashlib.sha256(b"reviewed diff").hexdigest(),
        "qa_sha256": compute_file_hash(str(qa_path)),
    }
    _signed(root / amendment.AMENDMENT_NAME, value)

    def git(command: list[str], **kwargs: object) -> str | bytes:
        if "rev-parse" in command:
            return "new\n"
        if "status" in command:
            return ""
        if "--name-only" in command:
            return changed + "\n"
        return b"reviewed diff"

    monkeypatch.setattr(amendment.subprocess, "check_output", git)
    return value


@pytest.mark.parametrize("hours", [1.0, 12.0, 24.0, 48.0])
def test_explicit_amendment_disables_only_time_without_resetting_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hours: float
) -> None:
    path = tmp_path / "budgets/stage.json"
    CampaignBudget(path, hours=hours, clock=lambda: 100.0)
    original = path.read_bytes()
    _fixture(tmp_path, monkeypatch)
    resumed = CampaignBudget(path, hours=hours, clock=lambda: 100.0 + hours * 3600 + 1)
    resumed.check()
    assert resumed.wall_time_unlimited
    assert path.read_bytes() == original
    assert amendment.training_identity_commit(tmp_path, tmp_path) == "old"
    (tmp_path / amendment.AMENDMENT_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        resumed.check()


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_commit", "foreign"),
        ("training_lock_sha256", "wrong"),
        ("policy", "skip_all_gates"),
        ("scope", ["stage64_seed7"]),
        ("fixed_endpoint_updates", 100),
        ("resource_margins_unchanged", False),
        ("user_request", ""),
        ("implementation_diff_sha256", "other"),
        ("qa_sha256", "wrong"),
        ("changed_files", ["src/e_jepa_ttc/models/raw_time_residual.py"]),
    ],
)
def test_time_amendment_rejects_changed_authority_or_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    payload = _fixture(tmp_path, monkeypatch)
    payload[field] = value
    _signed(tmp_path / amendment.AMENDMENT_NAME, payload)
    with pytest.raises(ValueError):
        amendment.validate_time_amendment(tmp_path, tmp_path)


def test_time_amendment_rejects_changed_qa_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    (tmp_path / "qa/log.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence"):
        amendment.validate_time_amendment(tmp_path, tmp_path)


def test_unlimited_wall_time_does_not_disable_ram_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    budget = CampaignBudget(tmp_path / "budgets/stage.json", hours=1, clock=lambda: 100.0)
    assert budget.wall_time_unlimited
    monkeypatch.setattr(
        campaign_budget.psutil, "virtual_memory", lambda: SimpleNamespace(available=19, total=100)
    )
    monkeypatch.setattr(
        campaign_budget.psutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )
    with pytest.raises(TimeoutError, match="RAM"):
        campaign_budget.check_resource_margins(tmp_path)


def test_foreign_execution_commit_without_amendment_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path, monkeypatch)
    (tmp_path / amendment.AMENDMENT_NAME).unlink()
    with pytest.raises(ValueError, match="exact clean training commit"):
        validate_training_authorization(tmp_path, tmp_path, stage=64, invocation={})
