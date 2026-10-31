"""Missing observations cannot win a screen by an easier paired intersection."""

import numpy as np
import pytest

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_timewarp.screen_metrics import compare_screen
from scripts import evaluate_phase_canonical_screen as evaluator


def test_coverage_loss_rejects_apparent_accuracy_win():
    result = compare_screen(
        np.ones(3),
        np.array([2.0, 5.0, 2.0]),
        np.array([1.0, np.nan, 1.0]),
        np.array(["a", "a", "b"]),
    )
    assert result["canonical"]["mae_seconds"] == 0
    assert result["newly_missing"] == 1
    assert not result["nonworsening"]


def test_recordings_are_equally_weighted_and_missing_original_is_retained():
    result = compare_screen(
        np.ones(4),
        np.array([2.0, 2.0, 5.0, np.nan]),
        np.array([1.0, 1.0, 3.0, np.nan]),
        np.array(["a", "a", "b", "b"]),
    )
    assert result["original"]["mae_seconds"] == 2.5
    assert result["canonical"]["mae_seconds"] == 1
    assert result["population"] == 4 and result["paired"] == 3
    assert result["nonworsening"]


def test_missing_target_or_entire_recording_prevents_passing():
    for target, values in (([1.0, np.nan], [2.0, 2.0]), ([1.0, 1.0], [2.0, np.nan])):
        result = compare_screen(
            np.array(target), np.array(values), np.array(values), np.array(["a", "b"])
        )
        assert not result["nonworsening"]


@pytest.mark.parametrize("tampered", [False, True])
def test_incomplete_or_tampered_predictions_stop_before_target_access(
    tmp_path, monkeypatch, tampered
):
    screen = tmp_path / "screen"
    atomic_receipt(screen / "FREEZE.json", {})
    atomic_receipt(screen / "PREDICTIONS.json", {"records": []})
    atomic_receipt(
        screen / "COMPLETE.json",
        {
            "status": "COMPLETE" if tampered else "INCOMPLETE",
            "baseline_exact": True,
            "targets_read": False,
            "optimizer_updates": 0,
            "queries": 149,
            "freeze_sha256": file_sha(screen / "FREEZE.json"),
            "artifacts": {"PREDICTIONS.json": "bad-digest"},
        },
    )
    opened = []
    monkeypatch.setattr(np, "load", lambda *a, **k: opened.append(a))
    monkeypatch.setattr(
        "sys.argv", ["evaluate", "--screen", str(screen), "--output", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError):
        evaluator.main()
    assert opened == []
