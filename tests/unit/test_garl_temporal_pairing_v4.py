from __future__ import annotations

import pytest

from e_jepa_ttc.data.garlttc_lhr_cache import _official_ttc_at_endpoint, select_temporal_indices


def test_temporal_pair_requires_both_endpoints_and_uses_frame_t2() -> None:
    pair = select_temporal_indices(
        [0, 90_000, 100_000],
        anchor_timestamp_us=100_000,
        target_delta_t_s=0.1,
        tolerance_s=0.025,
        context_delta_t_s=0.1,
        context_tolerance_s=0.01,
    )
    assert pair == (0, 2, None)
    assert _official_ttc_at_endpoint({"frame_ttc": [2.0, 1.5], "ttc": 99.0}, 1) == pytest.approx(
        1.5
    )
    with pytest.raises(ValueError, match=r"frame_ttc\[t2\]"):
        _official_ttc_at_endpoint({"frame_ttc": [2.0, float("nan")], "ttc": 99.0}, 1)


def test_temporal_pair_rejects_anchor_without_a_valid_endpoint_pair() -> None:
    with pytest.raises(ValueError, match="endpoint pair"):
        select_temporal_indices(
            [0, 90_000, 100_000],
            anchor_timestamp_us=130_000,
            target_delta_t_s=0.1,
            tolerance_s=0.025,
            context_delta_t_s=0.1,
            context_tolerance_s=0.01,
        )
