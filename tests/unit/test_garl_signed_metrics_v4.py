from __future__ import annotations

import numpy as np
import pytest

from e_jepa_ttc.evaluation.garl_ttc_protocol import (
    select_checkpoint,
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)


def test_signed_metrics_cover_all_domains_and_keep_zero_failure_explicit() -> None:
    target = np.asarray([-1.0, 1.0, 4.0, 8.0])
    metrics = signed_garl_metrics(target, target)
    assert metrics["failure_count"] == 0
    assert metrics["paper_MiD_overall"] == pytest.approx(0.0)
    assert metrics["bins"]["negative"]["count"] == 1
    zero_metrics = signed_garl_metrics(np.asarray([0.0]), np.asarray([0.0]))
    assert zero_metrics["bins"]["negative"]["count"] == 1
    assert zero_metrics["failure_count"] == 1


def test_signed_metrics_match_official_right_closed_bucket_boundaries() -> None:
    target = np.asarray([-10.0, -10.0 + 1e-6, 0.0, 3.0, 3.0 + 1e-6, 6.0, 10.0])
    metrics = signed_garl_metrics(target, target)
    assert metrics["bins"]["negative"]["count"] == 2
    assert metrics["bins"]["crucial"]["count"] == 1
    assert metrics["bins"]["small"]["count"] == 2
    assert metrics["bins"]["large"]["count"] == 1


def test_signed_metrics_keep_mid_failure_and_filter_only_rte() -> None:
    metrics = signed_garl_metrics(np.asarray([1.0, 2.0]), np.asarray([np.nan, 2.0]))
    assert metrics["bins"]["crucial"]["failure_count"] == 1
    assert metrics["bins"]["crucial"]["mid"] == pytest.approx(0.0)
    assert metrics["bins"]["crucial"]["rte_pct"] == pytest.approx(0.0)


def test_sequence_macro_and_checkpoint_selection_are_validation_only() -> None:
    macro = sequence_macro_signed_metrics(
        np.asarray([-1.0, 1.0, 4.0, 8.0, -1.0, 1.0, 4.0, 8.0]),
        np.asarray([-1.0, 1.0, 4.0, 8.0, -1.0, 1.0, 4.0, 8.0]),
        ["a", "a", "a", "a", "b", "b", "b", "b"],
    )
    assert macro["sequence_macro_paper_MiD_overall"] == pytest.approx(0.0)
    selected = select_checkpoint(
        [
            {"protocol": "garl_signed_v1", "paper_MiD_overall": 2.0, "failure_rate_pct": 0.0},
            {"protocol": "garl_signed_v1", "paper_MiD_overall": 1.0, "failure_rate_pct": 0.0},
        ]
    )
    assert selected["selected_index"] == 1


def test_checkpoint_selection_excludes_nan_metrics() -> None:
    selected = select_checkpoint(
        [
            {
                "protocol": "garl_signed_v1",
                "paper_MiD_overall": float("nan"),
                "failure_rate_pct": 0.0,
            },
            {
                "protocol": "garl_signed_v1",
                "paper_MiD_overall": 2.0,
                "failure_rate_pct": 1.0,
            },
        ]
    )
    assert selected["selected_index"] == 1
    assert selected["excluded_non_finite_indices"] == [0]

    with pytest.raises(ValueError, match="no finite validation metrics"):
        select_checkpoint(
            [
                {
                    "protocol": "garl_signed_v1",
                    "paper_MiD_overall": float("nan"),
                    "failure_rate_pct": 0.0,
                }
            ]
        )
