from __future__ import annotations

import numpy as np

from e_jepa_ttc.data.object_event_v4 import collate_object_event_v4


def _record(token: str) -> dict[str, object]:
    return {
        "event_v4_common_roi": np.random.default_rng(7).normal(
            size=(3, 12, 16, 16)
        ).astype(np.float32),
        "garl_delta_t_s": np.float32(0.1),
        "observable_motion": np.zeros(18, dtype=np.float32),
        "garl_visible_heights_px": np.asarray([20.0, 25.0], dtype=np.float32),
        "ttc_s": np.float32(0.5),
        "event_v4_boxes_xyxy": np.asarray(
            [[4, 4, 8, 8], [3, 3, 9, 9], [2, 2, 10, 10]], dtype=np.float32
        ),
        "event_v4_common_square_xyxy": np.asarray([0, 0, 16, 16], dtype=np.float32),
        "event_v4_precontext_valid": True,
        "sequence_id": "sequence",
        "sample_token": token,
        "track_id": "track",
    }


def test_collate_exposes_only_observable_model_inputs() -> None:
    batch = collate_object_event_v4([_record("a"), _record("b")])
    assert batch.events.shape == (2, 3, 12, 16, 16)
    assert set(batch.event_inputs()) == {"events", "delta_t_s"}
    assert set(batch.model_inputs()) == {"events", "delta_t_s", "observable_motion"}
    assert "target_ttc_s" not in batch.model_inputs()
    assert "boxes_xyxy" not in batch.model_inputs()
