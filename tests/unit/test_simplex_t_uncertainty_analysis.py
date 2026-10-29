"""Paired uncertainty plumbing with synthetic data and a mocked draw engine."""

import numpy as np
import pandas as pd
import pytest

from e_jepa_ttc.simplex_t import uncertainty_analysis


@pytest.fixture
def frames():
    index = np.arange(8192)
    base = pd.DataFrame(
        dict(
            sample_token=[f"q{i:05d}" for i in index],
            sequence_id=[f"s{i % 9}" for i in index],
            track_id=[f"t{i % 4}" for i in index],
            outer_fold=index % 3,
            target_ttc=np.asarray([-2.0, 1.0, 4.0, 8.0])[index % 4],
            loss=np.full(8192, 10.0),
        )
    )
    return {"reference": base, "candidate": base.assign(loss=8.0).iloc[::-1]}


def test_shared_draws_and_sequence_diagnostics_keep_all_queries(frames, tmp_path, monkeypatch):
    calls = []

    def draw(frame, losses, output, check):
        assert len(frame) == 8192
        assert "target_ttc_s" in frame
        calls.append(losses.copy())
        check()
        return np.tile([8.0, 10.0], (8192, 1)), {"fixture": True}

    monkeypatch.setattr(uncertainty_analysis, "hierarchical_losses", draw)
    output = tmp_path / "new"
    report = uncertainty_analysis.paired_uncertainty(
        frames,
        reference="reference",
        output=output,
        resource_check=lambda: None,
    )
    assert len(calls) == 1
    assert report["column_order"] == ["candidate", "reference"]
    candidate = report["comparisons"]["candidate"]
    assert candidate["point_delta"] == pytest.approx(-2)
    assert candidate["hierarchical_ci95"] == [-2.0, -2.0]
    assert candidate["sequence_only"]["sequence_wins"] == 9
    assert not candidate["sequence_only"]["retraining_performed"]
    assert (output / "PAIRED_UNCERTAINTY.json").exists()
    with pytest.raises(FileExistsError):
        uncertainty_analysis.paired_uncertainty(
            frames,
            reference="reference",
            output=output,
            resource_check=lambda: None,
        )


@pytest.mark.parametrize("failure", ["drop", "target", "track", "loss"])
def test_rejects_incomplete_or_unpaired_population_before_writing(frames, tmp_path, failure):
    candidate = frames["candidate"].copy()
    if failure == "drop":
        candidate = candidate.iloc[1:]
    elif failure == "loss":
        candidate.loc[0, "loss"] = np.inf
    elif failure == "target":
        candidate.loc[0, "target_ttc"] = 2.0
    else:
        candidate.loc[0, "track_id"] = "other"
    frames["candidate"] = candidate
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        uncertainty_analysis.paired_uncertainty(
            frames,
            reference="reference",
            output=output,
            resource_check=lambda: None,
        )
    assert not output.exists()


def test_real_historical_draw_engine_runs_through_adapter(frames, tmp_path):
    # One complete synthetic track per sequence guarantees all bucket support.
    frames = {name: frame.assign(track_id="complete-track") for name, frame in frames.items()}
    output = tmp_path / "real_draw_engine_synthetic_losses"
    report = uncertainty_analysis.paired_uncertainty(
        frames,
        reference="reference",
        output=output,
        resource_check=lambda: None,
    )
    assert report["bootstrap"]["valid_draws"] == 8192
    assert report["bootstrap"]["attempts"] == 8192
    assert report["comparisons"]["candidate"]["hierarchical_ci95"] == [-2.0, -2.0]
    assert (output / "HIERARCHICAL_DRAWS.jsonl").exists()
    assert np.load(output / "BOOTSTRAP_LOSSES.npy").shape == (8192, 2)


def test_interrupted_draws_never_publish_complete_report(frames, tmp_path, monkeypatch):
    def interrupted(*args):
        raise InterruptedError("synthetic resource pause")

    monkeypatch.setattr(uncertainty_analysis, "hierarchical_losses", interrupted)
    output = tmp_path / "new"
    with pytest.raises(InterruptedError):
        uncertainty_analysis.paired_uncertainty(
            frames,
            reference="reference",
            output=output,
            resource_check=lambda: None,
        )
    assert output.exists()
    assert not (output / "PAIRED_UNCERTAINTY.json").exists()
