from __future__ import annotations

import pytest

from e_jepa_ttc.data.annotations import LabelMeasurement
from e_jepa_ttc.data.evttc_object_cache import (
    EvTTCObjectCacheConfig,
    _event_box,
    _EvTTCState,
    _windows,
)


def _measurement(index: int) -> LabelMeasurement:
    return LabelMeasurement(
        sequence_id="sequence",
        frame_index=index,
        timestamp_us=1_000_000 + index * 50_000,
        category="car",
        bbox_xyxy=(192.0, 120.0, 960.0, 600.0),
        bbox_area=768.0 * 480.0,
        bbox_scale=(768.0 * 480.0) ** 0.5,
        ttc_seconds=8.0 - index * 0.05,
        image_width=1920,
        image_height=1200,
    )


def test_evttc_object_windows_and_cross_sensor_scaling() -> None:
    states = [
        _EvTTCState(
            measurement=_measurement(index),
            bbox_event_xyxy=(128.0, 72.0, 640.0, 360.0),
            depth_m=20.0 - index * 0.1,
        )
        for index in range(16)
    ]
    windows = _windows(states, EvTTCObjectCacheConfig())

    assert windows
    history, future = windows[0]
    assert len(history) == 3
    assert 100 in future
    assert _event_box(_measurement(0)) == (128.0, 72.0, 640.0, 360.0)


def test_evttc_action_contract_and_disjoint_horizons_are_validated() -> None:
    assert EvTTCObjectCacheConfig().action_dim == 8
    with pytest.raises(ValueError, match="must not overlap"):
        EvTTCObjectCacheConfig(prediction_horizons_ms=(50, 100))
    with pytest.raises(ValueError, match="eight physical"):
        EvTTCObjectCacheConfig(action_dim=4)
