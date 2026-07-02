import numpy as np

from e_jepa_ttc.baselines.roi_events import (
    ROI_EVENT_FEATURE_NAMES,
    _scale_bbox_to_event_plane,
    extract_roi_event_features,
)
from e_jepa_ttc.data.annotations import LabelMeasurement
from e_jepa_ttc.data.types import EventBatch


def test_scale_bbox_to_event_plane_uses_label_image_size() -> None:
    measurement = LabelMeasurement(
        sequence_id="seq",
        frame_index=0,
        timestamp_us=1_000,
        category="car",
        bbox_xyxy=(960.0, 600.0, 1920.0, 1200.0),
        bbox_area=960.0 * 600.0,
        bbox_scale=float(np.sqrt(960.0 * 600.0)),
        ttc_seconds=1.0,
        image_width=1920,
        image_height=1200,
    )

    bbox = _scale_bbox_to_event_plane(measurement, event_width=1280, event_height=720)

    assert bbox == (640.0, 360.0, 1280.0, 720.0)


def test_roi_event_features_ignore_future_events() -> None:
    events = EventBatch(
        x=np.array([10, 20, 30], dtype=np.int32),
        y=np.array([10, 15, 10], dtype=np.int32),
        t_us=np.array([0, 50, 150], dtype=np.int64),
        polarity=np.array([1, -1, 1], dtype=np.int8),
        width=100,
        height=100,
        sequence_id="seq",
        t_start_us=0,
        t_end_us=150,
    )

    features = extract_roi_event_features(
        events,
        bbox_xyxy_event=(0.0, 0.0, 25.0, 25.0),
        reference_time_us=100,
    )
    values = dict(zip(ROI_EVENT_FEATURE_NAMES, features.values, strict=True))

    assert features.total_event_count == 2
    assert features.roi_event_count == 2
    assert np.isclose(values["log_total_event_count"], np.log1p(2))
    assert np.isclose(values["log_roi_event_count"], np.log1p(2))
    assert values["roi_positive_fraction"] == 0.5
    assert values["roi_polarity_balance"] == 0.0
