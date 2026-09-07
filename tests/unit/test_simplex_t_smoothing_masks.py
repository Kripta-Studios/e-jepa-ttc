"""Label-free EWMA must ignore absent payloads and retain actual current input."""

import pytest
import torch

from e_jepa_ttc.simplex_t.controls import ewma_phase
from e_jepa_ttc.simplex_t.phase import emitted_phase


def test_absent_nan_payload_and_lag_do_not_contaminate_current_only_smoothing():
    experts = torch.tensor([[[float("nan")] * 3, [0.02] * 3]])
    lags = torch.tensor([[float("nan"), 0.0]])
    valid = torch.tensor([[False, True]])
    actual = ewma_phase(experts, lags, valid)
    assert torch.equal(actual, emitted_phase(torch.tensor([0.02])))


@pytest.mark.parametrize("corruption", ["current_missing", "current_lag", "lag_nan", "phase_nan"])
def test_invalid_observed_input_rejected(corruption):
    experts = torch.full((1, 2, 3), 0.02)
    lags = torch.tensor([[0.1, 0.0]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    if corruption == "current_missing":
        valid[0, -1] = False
    elif corruption == "current_lag":
        lags[0, -1] = 0.01
    elif corruption == "lag_nan":
        lags[0, 0] = float("nan")
    else:
        experts[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        ewma_phase(experts, lags, valid)


def test_valid_finite_recipe_is_bit_identical_to_original_weighting():
    generator = torch.Generator().manual_seed(7)
    experts = torch.randn((8, 16, 3), generator=generator) * 0.03
    lags = torch.arange(15, -1, -1).repeat(8, 1) * 0.05
    valid = torch.rand((8, 16), generator=generator) > 0.2
    valid[:, -1] = True
    phases = torch.where(valid[..., None], experts, torch.zeros_like(experts)).median(-1).values
    weights = torch.exp(-lags / 0.3) * valid
    original = emitted_phase((weights * phases).sum(-1) / weights.sum(-1))
    assert torch.equal(ewma_phase(experts, lags, valid), original)
