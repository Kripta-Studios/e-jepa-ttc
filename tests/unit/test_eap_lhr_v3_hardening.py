from __future__ import annotations

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.garlttc_lhr_cache import (
    _box_features,
    _geometry_target,
    select_temporal_indices,
)
from e_jepa_ttc.models.eap_lhr_jepa_ttc import EAPLHRJEPATTC, EAPLHRJEPATTCConfig


def test_temporal_pair_is_near_100ms_and_context_is_prior() -> None:
    timestamps = [700_000, 800_000, 900_000, 1_000_000]
    first, second, context = select_temporal_indices(
        timestamps,
        anchor_timestamp_us=1_000_000,
        target_delta_t_s=0.1,
        tolerance_s=0.01,
        context_delta_t_s=0.1,
        context_tolerance_s=0.01,
    )
    assert (first, second, context) == (2, 3, 1)


def test_temporal_pair_rejects_wrong_horizon() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        select_temporal_indices(
            [0, 40_000, 200_000],
            anchor_timestamp_us=200_000,
            target_delta_t_s=0.1,
            tolerance_s=0.01,
            context_delta_t_s=0.1,
            context_tolerance_s=0.01,
        )


def test_visibility_uses_area_not_only_horizontal_width() -> None:
    values, _ = _box_features((100, 100, 200, 300), (100, -100, 200, 100), 0.1)
    assert 0.49 < float(values[10]) < 0.51


def test_track_age_is_not_delta_t() -> None:
    observable = np.zeros(18, dtype=np.float32)
    observable[17] = 0.2
    metadata = {"log_area_rate_raw": 0.0}
    values, valid = _geometry_target(
        observable, metadata, depths=None, delta_t_s=0.1, track_age_s=1.0
    )
    assert valid[13]
    assert float(values[13]) == pytest.approx(0.5)
    assert float(values[13]) != pytest.approx(float(observable[17]))


def test_jepa_prediction_does_not_use_second_endpoint() -> None:
    torch.manual_seed(7)
    model = EAPLHRJEPATTC(EAPLHRJEPATTCConfig(dim=32)).eval()
    full = torch.randn(2, 2, 21, 32, 32)
    roi = torch.randn(2, 40, 32, 32)
    motion = torch.randn(2, 18)
    context_motion = torch.randn(2, 18)
    elapsed = torch.full((2,), 0.1)
    with torch.no_grad():
        first = model(
            full_frame_events=full,
            event_roi_pair=roi,
            delta_t_s=elapsed,
            observable_motion=motion,
            jepa_context_motion=context_motion,
        ).jepa_prediction
        changed_full = full.clone()
        changed_full[:, 1] = torch.randn_like(changed_full[:, 1]) * 10
        changed_roi = roi.clone()
        changed_roi[:, 20:] = torch.randn_like(changed_roi[:, 20:]) * 10
        second = model(
            full_frame_events=changed_full,
            event_roi_pair=changed_roi,
            delta_t_s=elapsed,
            observable_motion=motion,
            jepa_context_motion=context_motion,
        ).jepa_prediction
    torch.testing.assert_close(first, second)
