from __future__ import annotations

import numpy as np
import pytest

from e_jepa_ttc.data.garl_input_contract import (
    EVENT_CHANNEL_NAMES,
    INPUT_SCHEMA_VERSION,
    validate_input_schema,
)
from e_jepa_ttc.data.garlttc_calibration import OFFICIAL_FY
from e_jepa_ttc.data.garlttc_lhr_cache import (
    _official_ttc_at_endpoint,
    _visible_heights,
    select_temporal_indices,
)
from e_jepa_ttc.data.garlttc_sampling import select_balanced_cache_rows, signed_ttc_bucket
from e_jepa_ttc.evaluation.garl_ttc_protocol import signed_garl_metrics


def test_input_schema_is_explicit_and_rejects_incompatible_cache() -> None:
    schema = {
        "version": INPUT_SCHEMA_VERSION,
        "event_roi_shape": [2, 20, 128, 128],
        "channel_names": list(EVENT_CHANNEL_NAMES),
        "normalization": "official_timevolume20_grid_sample_v1",
    }
    validate_input_schema(schema)
    schema["version"] = "garlttc_input_v3"
    with pytest.raises(ValueError, match="Unsupported Garl input schema"):
        validate_input_schema(schema)


def test_signed_protocol_keeps_negative_bucket_and_has_no_positive_only_selection() -> None:
    target = np.asarray([-1.0, 2.0, 4.0, 8.0])
    metrics = signed_garl_metrics(target, target)
    assert metrics["failure_count"] == 0
    assert metrics["paper_MiD_overall"] == pytest.approx(0.0)
    assert metrics["weighted_RTE_pct"] == pytest.approx(0.0)
    assert metrics["bins"]["negative"]["count"] == 1
    assert signed_ttc_bucket(-0.5) == "negative"
    assert signed_ttc_bucket(-10.0) == "out_of_protocol"
    assert signed_ttc_bucket(0.0) == "negative"


def test_temporal_pair_checks_both_endpoint_errors_and_ttc_comes_from_frame_t2() -> None:
    first, second, context = select_temporal_indices(
        [0, 90_000, 100_000],
        anchor_timestamp_us=100_000,
        target_delta_t_s=0.1,
        tolerance_s=0.025,
        context_delta_t_s=0.1,
        context_tolerance_s=0.01,
    )
    assert (first, second, context) == (0, 2, None)
    with pytest.raises(ValueError, match="endpoint pair"):
        select_temporal_indices(
            [0, 90_000, 100_000],
            anchor_timestamp_us=130_000,
            target_delta_t_s=0.1,
            tolerance_s=0.025,
            context_delta_t_s=0.1,
            context_tolerance_s=0.01,
        )

    row = {"frame_ttc": [2.0, 1.5], "ttc": 9.0}
    assert _official_ttc_at_endpoint(row, 1) == pytest.approx(1.5)
    with pytest.raises(ValueError, match=r"frame_ttc\[t2\]"):
        _official_ttc_at_endpoint({"frame_ttc": [2.0, float("nan")], "ttc": 9.0}, 1)
    assert _official_ttc_at_endpoint(
        {"frame_ttc": [2.0, float("nan")], "ttc": 9.0},
        1,
        allow_row_ttc_compatibility=True,
    ) == pytest.approx(9.0)


def test_official_visible_height_does_not_require_row_k_event() -> None:
    row = {
        "box3d_h": 1.8,
        "box3d_Fcam": [
            [[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [1.0, 1.0, 10.0]],
            [[0.0, 0.0, 8.0], [1.0, 0.0, 8.0], [1.0, 1.0, 8.0]],
        ],
    }
    values = _visible_heights(row, (0, 1), [(0.0, 0.0, 64.0, 64.0)] * 2, 128)
    assert values[0] == pytest.approx(OFFICIAL_FY * 1.8 / 10.0 * 2.0)


def test_balanced_selection_is_before_materialization_and_sequence_aware() -> None:
    import pandas as pd

    rows = pd.DataFrame(
        [
            {"sequence_id": sequence, "track_id": track, "ttc": ttc, "sampling_group": "g"}
            for sequence in ("s1", "s2")
            for track in ("a", "b")
            for ttc in (-1.0, 2.0, 4.0, 8.0)
        ]
    )
    selected, report = select_balanced_cache_rows(
        rows,
        {"s1": "train", "s2": "validation"},
        seed=7,
        max_samples_per_split=4,
    )
    assert len(selected) == 8
    assert selected["sequence_id"].value_counts().to_dict() == {"s1": 4, "s2": 4}
    assert report["discard_count"] == len(rows) - len(selected)
