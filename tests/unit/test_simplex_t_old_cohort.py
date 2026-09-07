"""Metadata-only cohort assembly; producer exclusion loader is tested separately."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import old_cohort as module


@pytest.mark.parametrize(
    "failure", ["", "index", "resource", "target", "duplicate", "fold", "role"]
)
def test_old_cohort(tmp_path: Path, monkeypatch, failure: str):
    (tmp_path / "tables").mkdir()
    index = tmp_path / "FROZEN_EXPERT_TABLE_INDEX.json"
    ancestry = tmp_path / "NESTED_ANCESTRY_AUDIT.json"
    index.write_text("fixture", encoding="utf-8")
    ancestry.write_text("fixture", encoding="utf-8")
    rows = pd.DataFrame(
        {
            "sample_token": [f"q{i:05d}" for i in range(8192)],
            "sequence_id": [f"s{i % 9}" for i in range(8192)],
            "track_id": [f"t{i % 20}" for i in range(8192)],
            "target_ttc": np.full(8192, 2.0),
            "outer_fold": [i % 3 for i in range(8192)],
        }
    )
    if failure == "duplicate":
        rows.loc[1, "sample_token"] = rows.loc[0, "sample_token"]
    if failure == "role":
        rows.loc[0, "sequence_id"] = "closed"
    tables, calls = {}, []
    for fold in range(3):
        frame = rows.loc[rows.outer_fold == fold].reset_index(drop=True)
        if failure == "fold" and fold == 1:
            frame["outer_fold"] = 0
        path = tmp_path / "tables" / f"outer{fold}_outer_dev.csv"
        frame.to_csv(path, index=False)
        tables[fold] = {
            "metadata": frame.copy(),
            "reference": {"metadata_sha256": sha256(path), "arrays_sha256": "a" * 64},
        }
    if failure == "target":
        tables[0]["metadata"].loc[0, "target_ttc"] = 9.0

    def load(root, fold, role, **kwargs):
        calls.append((fold, role))
        assert role == "outer_dev"
        return tables[fold]

    monkeypatch.setattr(module, "load_current_inputs", load)
    options: dict[str, Any] = dict(
        ancestry_sha256=sha256(ancestry),
        table_index_sha256="0" * 64 if failure == "index" else sha256(index),
        allowed_sequences={f"s{i}" for i in range(9)},
        resource_ok=lambda: failure != "resource",
    )
    if failure:
        with pytest.raises(InterruptedError if failure == "resource" else ValueError):
            module.load_old_evaluation_cohort(tmp_path, **options)
        if failure in {"index", "resource"}:
            assert not calls
    else:
        frame, receipt = module.load_old_evaluation_cohort(tmp_path, **options)
        pd.testing.assert_frame_equal(frame, rows.loc[:, frame.columns])
        assert calls == [(0, "outer_dev"), (1, "outer_dev"), (2, "outer_dev")]
        assert receipt["queries"] == 8192
        assert receipt["history_constructed"] is False
        assert receipt["prediction_scores_read"] is False
