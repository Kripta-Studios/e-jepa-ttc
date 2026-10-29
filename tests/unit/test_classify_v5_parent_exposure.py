"""Tests for fail-closed classification of parent-exposed V5 runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.classify_v5_parent_exposure import build_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parent_exposure_blocks_all_affected_runs(tmp_path: Path) -> None:
    parent = tmp_path / "global_parent.pt"
    parent.write_bytes(b"globally exposed parent")
    parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    runs = tmp_path / "runs"
    _write_json(
        runs / "fold0" / "summary.json",
        {
            "initialization": {"checkpoint_sha256": parent_sha},
            "selection": {"sequence_macro_MiD": 119.50065707115164},
            "artifact_sha256": "signed-summary",
        },
    )
    _write_json(
        runs / "fold1" / "state" / "progress.json",
        {"status": "running", "epoch": 10, "best_selection": 142.11},
    )

    report = build_report(
        runs,
        global_parent=parent,
        run_names=("fold0", "fold1"),
        repository_root=tmp_path,
    )

    assert report["status"] == "completed_promotion_blocked"
    assert report["contracts"]["private_test_opened"] is False
    assert report["contracts"]["affected_summaries_modified"] is False
    assert len(report["affected_runs"]) == 2
    assert all(row["promotion_eligible"] is False for row in report["affected_runs"])
    fold0, fold1 = report["affected_runs"]
    assert fold0["selection"]["sequence_macro_MiD"] == 119.50065707115164
    assert fold0["summary_left_unmodified"] is True
    assert fold1["interrupted"] is True
    assert fold1["termination_reason"] == "parent_exposure_detected"
