"""Recovery-ledger checks reject replay double counting and calendar drift."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_detection_rare_result import audit_calendar


def make_history(tmp_path: Path):
    runs = []
    tokens = ["a", "b", "c", "d"]
    for name, start, end, logged_end in (
        ("smoke", 1, 8, 8),
        ("interrupted", 9, 3840, 4096),
        ("resumed", 3841, 8192, 8192),
    ):
        run = tmp_path / name
        run.mkdir()
        rows = [
            {"update": update, "sample_tokens": tokens}
            for update in range(start, logged_end + 1)
        ]
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        runs.append((run, start, end))
    return runs, {"indices": [0, 1, 2, 3] * 8192}, tokens


def test_recovery_counts_committed_updates_once(tmp_path):
    receipt = audit_calendar(*make_history(tmp_path))
    assert receipt["updates"] == 8192
    assert receipt["exposures"] == 32768


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "wrong_token"])
def test_recovery_rejects_corrupted_committed_history(tmp_path, corruption):
    runs, plan, tokens = make_history(tmp_path)
    path = runs[-1][0] / "metrics.jsonl"
    rows = path.read_text().splitlines()
    if corruption == "missing":
        rows.pop(10)
    elif corruption == "duplicate":
        rows.insert(10, rows[10])
    else:
        value = json.loads(rows[10])
        value["sample_tokens"][0] = "other"
        rows[10] = json.dumps(value)
    path.write_text("\n".join(rows))
    with pytest.raises(ValueError):
        audit_calendar(runs, plan, tokens)
