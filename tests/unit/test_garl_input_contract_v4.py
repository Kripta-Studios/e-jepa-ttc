from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.data.garl_input_contract import (
    EVENT_CHANNEL_NAMES,
    INPUT_SCHEMA_VERSION,
    GarlTTCModelInput,
    validate_input_schema,
)


def _valid_input() -> GarlTTCModelInput:
    return GarlTTCModelInput(
        event_roi_endpoints=torch.zeros(2, 2, 20, 128, 128),
        endpoint_timestamps_us=torch.zeros(2, 2, dtype=torch.int64),
        delta_t_s=torch.full((2,), 0.1),
        boxes_xyxy=None,
        full_event_context=None,
        rgb_endpoints=None,
        input_valid=torch.ones(2, dtype=torch.bool),
        protocol_id="garlttc_official_v1",
    )


def test_input_contract_accepts_official_shape_and_rejects_channel_changes() -> None:
    value = _valid_input()
    value.validate()
    with pytest.raises(ValueError, match="event_roi_endpoints"):
        GarlTTCModelInput(
            event_roi_endpoints=torch.zeros(2, 2, 19, 128, 128),
            endpoint_timestamps_us=value.endpoint_timestamps_us,
            delta_t_s=value.delta_t_s,
            boxes_xyxy=None,
            full_event_context=None,
            rgb_endpoints=None,
            input_valid=value.input_valid,
            protocol_id=value.protocol_id,
        ).validate()


def test_input_schema_checks_version_channel_order_and_normalization() -> None:
    schema = {
        "version": INPUT_SCHEMA_VERSION,
        "event_roi_shape": [2, 20, 128, 128],
        "channel_names": list(EVENT_CHANNEL_NAMES),
        "normalization": "official_timevolume20_grid_sample_v1",
    }
    validate_input_schema(schema)
    schema["channel_names"] = list(reversed(schema["channel_names"]))
    with pytest.raises(ValueError, match="channel_names"):
        validate_input_schema(schema)
