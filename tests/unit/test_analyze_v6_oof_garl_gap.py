"""Tests for V6 OOF gap decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd

from e_jepa_ttc.evaluation.garl_ttc_protocol import signed_garl_metrics
from scripts.analyze_v6_oof_garl_gap import (
    _arm_columns,
    _official_bucket_decomposition,
)


def test_official_bucket_contributions_reconstruct_mid_for_one_sequence() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["s"] * 4,
            "target_ttc_s": [2.0, 4.0, 8.0, -2.0],
            "v6_1_prediction": [2.2, 3.8, 8.5, -2.2],
        }
    )

    result = _official_bucket_decomposition(frame, "v6_1")
    reconstructed = sum(
        bucket["weighted_contribution_to_sequence_macro_MiD"] for bucket in result.values()
    )
    expected = signed_garl_metrics(frame["target_ttc_s"], frame["v6_1_prediction"])[
        "paper_MiD_overall"
    ]

    assert np.isclose(reconstructed, expected)


def test_arm_columns_preserves_failures() -> None:
    frame = pd.DataFrame(
        {
            "target_ttc_s": [2.0, -2.0],
            **{
                f"{arm}_prediction": [2.0, np.nan if arm == "a5_causal" else -2.0]
                for arm in ("a5_causal", "v6_1", "a8_0", "a6", "garl")
            },
        }
    )

    result = _arm_columns(frame)

    assert bool(result.loc[1, "a5_causal_failure"]) is True
    assert np.isnan(result.loc[1, "a5_causal_raw_mid"])
    assert bool(result.loc[1, "garl_failure"]) is False
