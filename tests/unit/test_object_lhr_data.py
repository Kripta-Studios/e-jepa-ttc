from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.data.garlttc_object_lhr import collate_object_lhr


def _record(ttc: float = 2.0) -> dict[str, object]:
    return {
        "jepa_event_roi": torch.zeros(2, 21, 16, 16),
        "garl_delta_t_s": 0.1,
        "garl_visible_heights_px": torch.tensor([90.0, 94.736842]),
        "ttc_s": ttc,
        "sequence_id": "sequence",
        "sample_token": "sample",
        "track_id": "track",
    }


def test_collate_exposes_only_events_and_delta_as_model_inputs() -> None:
    batch = collate_object_lhr([_record()])
    assert set(batch.model_inputs()) == {"events", "delta_t_s"}
    assert batch.events.shape == (1, 2, 21, 16, 16)
    assert batch.visible_heights_px.shape == (1, 2)
    assert not batch.mask_valid.any()


def test_collate_rejects_nonpositive_lhr_ratio() -> None:
    record = _record(ttc=0.05)
    with pytest.raises(ValueError, match="non-positive height ratio"):
        collate_object_lhr([record])


def test_collate_accepts_optional_mask_supervision() -> None:
    record = _record()
    record["garl_mask_pair"] = torch.ones(2, 1, 16, 16)
    record["garl_mask_valid"] = torch.tensor([True, False])
    batch = collate_object_lhr([record])
    assert batch.masks.shape == (1, 2, 1, 16, 16)
    assert batch.mask_valid.tolist() == [[True, False]]
