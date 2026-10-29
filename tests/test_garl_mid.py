"""Integration checks against the pinned official scorer with hand-computed cases."""

from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from operational.garl_comparison.mid import load_scorer, paired_mid, scores


@pytest.fixture(scope="module")
def scorer() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/garlttc_submission_20261010/official_score.py"
    )
    if not path.exists():
        pytest.skip("Official scorer integration requires the audited local evidence bundle")
    return load_scorer(path)


def test_perfect_predictions_cover_all_four_official_bands(scorer: ModuleType) -> None:
    truth = np.array([1.0, 4.0, 8.0, -2.0])
    result = scores(scorer, truth, truth.copy())
    assert result["overall_MiD"] == 0
    assert result["overall_available"]
    assert result["failed_rate"] == 0
    assert [result[f"n_{s}"] for s in "csln"] == [1, 1, 1, 1]


def test_known_crucial_error_has_half_weight_in_ranking(scorer: ModuleType) -> None:
    result = scores(scorer, np.array([1.0, 4.0, 8.0, -2.0]), np.array([2.0, 4.0, 8.0, -2.0]))
    assert result["MiDc"] == pytest.approx(540.6722127028)
    assert result["overall_MiD"] == pytest.approx(270.3361063514)
    assert result["mean_MiD"] == pytest.approx(135.1680531757)


def test_missing_negative_band_does_not_renormalize_ranking(scorer: ModuleType) -> None:
    truth = np.array([1.0, 4.0, 8.0])
    result = scores(scorer, truth, truth.copy())
    assert result["n_n"] == 0
    assert result["mean_MiD"] == 0
    assert not result["overall_available"]
    assert np.isnan(result["overall_MiD"])


def test_invalid_height_ratio_is_not_hidden_by_conditional_mean(scorer: ModuleType) -> None:
    result = scores(scorer, np.array([1.0, 2.0]), np.array([0.05, 2.0]))
    assert result["mean_MiD"] == 0
    assert result["failed_rate"] == 50
    assert result["invalid_mid_total"] == result["invalid_mid_c"] == 1
    assert result["invalid_mid_n"] == 0
    assert result["strict_mean_MiD"] is None


def test_scorer_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "score.py"
    path.write_text("raise AssertionError('must not execute')", encoding="utf-8")
    with pytest.raises(ValueError, match="audited release"):
        load_scorer(path)


def test_bootstrap_keeps_sequences_together_and_sample_weighted_point() -> None:
    ours = np.array([2.0] * 100 + [0.0])
    comparator = np.ones(101)
    sequence = np.array(["a"] * 100 + ["b"])
    result = paired_mid(ours, comparator, sequence)
    assert result["delta_mean_MiD"] == pytest.approx(99 / 101)
    assert result["ci95_low"] == -1
    assert result["ci95_high"] == 1
    assert result == paired_mid(ours, comparator, sequence)


def test_bootstrap_does_not_drop_invalid_mid() -> None:
    result = paired_mid(np.array([0.0, np.nan]), np.zeros(2), np.array(["a", "b"]))
    assert result["status"] == "UNAVAILABLE_INVALID_MID_RETAINED"
    assert "delta_mean_MiD" not in result
