from __future__ import annotations

import numpy as np

from e_jepa_ttc.data.eap import EAPObjectState
from e_jepa_ttc.data.eap_geometry_v2 import EAP_GEOMETRY_V2_DIM, geometry_v2_targets


def _state(
    *, timestamp_us: int, box: tuple[float, float, float, float], height: float
) -> EAPObjectState:
    return EAPObjectState(
        sample_token=str(timestamp_us),
        sequence_id="sequence",
        track_id="track",
        category="pedestrian.adult",
        timestamp_us=timestamp_us,
        bbox_xyxy=box,
        bbox_3d_ego=(0.0, 0.0, 10.0, 4.0, 2.0, 1.5, 0.0),
        nearest_depth_m=10.0,
        visible_height_px=height,
        depth_velocity_mps=-2.0,
        ttc_s=5.0,
        unclipped_bbox_xyxy=box,
        visibility_fraction=1.0,
        first_seen_timestamp_us=0,
    )


def test_geometry_v2_detects_transverse_motion() -> None:
    previous = _state(timestamp_us=0, box=(100.0, 100.0, 200.0, 300.0), height=200.0)
    current = _state(timestamp_us=100_000, box=(228.0, 100.0, 328.0, 302.0), height=202.0)
    target = geometry_v2_targets(current, previous)
    assert target.values.shape == (EAP_GEOMETRY_V2_DIM,)
    assert target.valid.shape == target.values.shape
    assert target.values[8] > 0.1
    assert target.values[9] < 0.5
    assert ":transverse:" in target.sampling_group


def test_geometry_v2_detects_radial_looming() -> None:
    previous = _state(timestamp_us=0, box=(500.0, 200.0, 700.0, 400.0), height=200.0)
    current = _state(timestamp_us=100_000, box=(490.0, 180.0, 710.0, 420.0), height=240.0)
    target = geometry_v2_targets(current, previous)
    assert np.isfinite(target.values).all()
    assert target.values[9] > 0.5
    assert ":longitudinal:" in target.sampling_group


def test_geometry_v2_marks_partial_visibility() -> None:
    state = _state(timestamp_us=100_000, box=(0.0, 100.0, 80.0, 300.0), height=200.0)
    state = EAPObjectState(
        **{
            **state.__dict__,
            "unclipped_bbox_xyxy": (-40.0, 100.0, 80.0, 300.0),
            "visibility_fraction": 2.0 / 3.0,
        }
    )
    target = geometry_v2_targets(state, None)
    assert target.values[10] == np.float32(2.0 / 3.0)
    assert target.values[11] == 1.0
    assert ":partial:" in target.sampling_group
