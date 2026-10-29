from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.data.evttc_garl_adapter import (
    make_garl_model_input,
    reject_labels_from_predict_payload,
)


def test_adapter_constructs_model_input_without_supervision_fields() -> None:
    value = make_garl_model_input(
        torch.zeros(1, 2, 20, 128, 128),
        torch.tensor([[0, 100_000]], dtype=torch.int64),
        torch.tensor([0.1]),
    )
    assert value.protocol_id == "evttc_garl_p0_zero_shot_v1"
    assert value.input_valid.tolist() == [True]


def test_adapter_rejects_ttc_and_future_labels_at_predict_boundary() -> None:
    with pytest.raises(ValueError, match="forbidden labels"):
        reject_labels_from_predict_payload({"events": [], "ttc_s": 1.0})
