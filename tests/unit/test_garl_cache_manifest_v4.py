from __future__ import annotations

import pytest

from e_jepa_ttc.data.garl_input_contract import (
    EVENT_CHANNEL_NAMES,
    validate_cache_manifest_input_schema,
)


def _manifest() -> dict[str, object]:
    return {
        "input_schema": {
            "version": "garlttc_input_v4",
            "event_roi_shape": [2, 20, 128, 128],
            "channel_names": list(EVENT_CHANNEL_NAMES),
            "normalization": "official_timevolume20_grid_sample_v1",
        }
    }


def test_cache_manifest_input_schema_is_validated_before_model_use() -> None:
    validate_cache_manifest_input_schema(_manifest())
    broken = _manifest()
    broken["input_schema"]["event_roi_shape"] = [2, 20, 128, 127]  # type: ignore[index]
    with pytest.raises(ValueError, match="event_roi_shape"):
        validate_cache_manifest_input_schema(broken)
