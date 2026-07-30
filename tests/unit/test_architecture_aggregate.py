import json
import sys
from pathlib import Path

import pytest

from scripts.aggregate_evttc_architecture_selection import main


def _write_summary(root: Path, *, fold: int, seed: int, score: float) -> None:
    path = root / f"fold-{fold}" / "A0_MATCHED_GLOBAL" / f"seed-{seed}" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "validation": {
                    "sequence_macro_selection_score": score,
                    "sequence_macro_mean_relative_error": score / 2,
                    "sequence_macro_mae_s": score * 2,
                    "worst_sequence_selection_score": score + 1,
                    "worst_sequence_mae_s": score + 2,
                    "milliseconds_per_window": 1.0,
                },
                "peak_vram_bytes": 1,
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_uses_only_predeclared_fold_seed_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runs"
    for fold in (0, 1):
        for seed in (7, 13):
            _write_summary(root, fold=fold, seed=seed, score=float(fold + seed))
    _write_summary(root, fold=0, seed=99, score=-1000.0)
    output = tmp_path / "aggregate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate",
            "--root",
            str(root),
            "--output",
            str(output),
            "--expected-folds",
            "2",
            "--expected-seeds",
            "2",
            "--folds",
            "0",
            "1",
            "--seeds",
            "7",
            "13",
        ],
    )

    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    row = payload["ranking"][0]
    assert row["complete_for_final_selection"] is True
    assert row["run_count"] == 4
    assert row["seeds"] == [7, 13]
    assert payload["expected_fold_ids"] == [0, 1]
    assert payload["expected_seed_ids"] == [7, 13]
