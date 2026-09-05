"""The routing point must originate from the same signed nested producer."""

from pathlib import Path

import pandas as pd
import pytest

from scripts import run_scientific_recovery_v9_stage65 as runner


@pytest.mark.parametrize("corruption", [None, "checkpoint", "csv", "path", "token"])
def test_bound_point_rejects_foreign_or_corrupt_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str | None
) -> None:
    train = tmp_path / "train"
    train.mkdir()
    point = train / "dev_predictions.csv"
    pd.DataFrame({"sample_token": ["a"], "point_prediction_ttc_s": [2.0]}).to_csv(
        point, index=False
    )
    summary = {
        "checkpoint": {"sha256": "foreign" if corruption == "checkpoint" else "same"},
        "predictions": {"path": point.name, "sha256": runner._sha(point)},
    }
    if corruption == "csv":
        point.write_text("corrupt", encoding="utf-8")
    if corruption == "path":
        summary["predictions"]["path"] = "../outside.csv"
    monkeypatch.setattr(
        runner,
        "read_signed",
        lambda path: summary if path.name == "summary.json" else {"checkpoint": {"sha256": "same"}},
    )
    frame = pd.DataFrame(
        {"token_id": ["other" if corruption == "token" else "a"], "prediction_ttc": [float("nan")]}
    )
    if corruption is not None:
        with pytest.raises(ValueError):
            runner._bound_routing_points(frame, tmp_path)
    else:
        bound, sources = runner._bound_routing_points(frame, tmp_path)
        assert bound.prediction_ttc.tolist() == [2.0]
        assert len(sources) == 2
