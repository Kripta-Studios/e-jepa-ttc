"""Numerical diagnostics must preserve actual historical batches and valid history order."""

import numpy as np
import pytest
import torch

from operational.joint_measurement.fusion_head_parity import variants


def cache():
    values = {"sample_token": np.asarray([str(i) for i in range(130)])}
    for modality in ("event", "rgb"):
        values[f"{modality}_features"] = np.arange(130 * 4 * 3, dtype=np.float32).reshape(130, 4, 3)
        values[f"{modality}_timing"] = np.zeros((130, 4, 2), dtype=np.float32)
        values[f"{modality}_valid"] = np.tile([False, False, True, True], (130, 1))
    values["event_expert_phase"] = np.arange(130 * 3, dtype=np.float32).reshape(130, 3)
    return values


def test_historical_tail_and_repeat_preserve_selected_query_and_experts():
    values = cache()
    result = variants(values, 129)
    batch, offset = result["historical_batch128"]
    single, _ = result["padded_batch1"]
    repeated, _ = result["repeat128"]
    trimmed, _ = result["trimmed_batch1"]
    assert offset == 1 and len(batch[0]) == 2
    for full, one, repeat in zip(batch, single, repeated, strict=True):
        assert torch.equal(full[offset], one[0])
        assert torch.equal(repeat[0], one[0]) and torch.equal(repeat[127], one[0])
    for feature, timing, valid in ((0, 1, 2), (4, 5, 6)):
        for key in (feature, timing, valid):
            assert torch.equal(trimmed[key], single[key][:, -2:])
    assert torch.equal(trimmed[3], single[3])


@pytest.mark.parametrize("mask", [[True, False, True, True], [False, False, False, False]])
def test_holes_and_absent_modality_are_not_silently_dropped(mask):
    values = cache()
    values["rgb_valid"][0] = mask
    with pytest.raises(ValueError, match="suffix"):
        variants(values, 0)


def test_missing_row_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        variants(cache(), 130)
