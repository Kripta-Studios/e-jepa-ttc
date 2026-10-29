"""Final reporting must not promote positive gates over a conformance block."""

import hashlib
from pathlib import Path

import torch

from scripts.package_scientific_recovery_v9_stage63_65 import _report, _snapshot_index


def test_integrity_block_is_distinct_from_observed_runner_candidate(tmp_path: Path) -> None:
    text = _report(
        {
            "status": "INTEGRITY_BLOCKED",
            "execution_status": "RISK_ROUTER_DEV_CANDIDATE",
            "final_integrity_audit": {"acceptance_blockers": ["CE17 replay was late"]},
        },
        {"automatic_next_action": "none", "analysis_commit": "analysis"},
        tmp_path,
    )
    assert "Final protocol status: `INTEGRITY_BLOCKED`" in text
    assert "Observed runner endpoint: `RISK_ROUTER_DEV_CANDIDATE`" in text
    assert "candidate is not accepted for promotion" in text
    assert "CE17 replay was late" in text


def test_snapshot_index_verifies_bytes_and_never_calls_partial_final(tmp_path: Path) -> None:
    path = tmp_path / "resume_snapshots/attempt/checkpoint_last.pt"
    path.parent.mkdir(parents=True)
    torch.save(
        {"identity": {"seed": 7, "outer_fold": 2, "arm": "S64-RAW-L1"}, "completed_updates": 1700},
        path,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".pt.sha256").write_text(digest + "\n", encoding="ascii")
    record = _snapshot_index(tmp_path)[0]
    assert record["sha256"] == digest
    assert record["updates"] == 1700
    assert record["status"] == "snapshot_bytes_verified_not_final"
    assert Path(record["path"]).is_absolute()
