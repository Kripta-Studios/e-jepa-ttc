"""Canonical PAIR tests preserve batch grouping, exact identity and unclipped phase math."""

import pytest
import torch
from torch import nn

from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.rgb_port.features import pair133
from operational.phase_timewarp.canonical_pair import CanonicalPair


class RecordingPair(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, batch):
        self.calls.append(batch.features.clone())
        return batch.features[:, 0].clone()


def batch(phases, dt):
    tokens = torch.zeros(len(dt), 128, dtype=torch.float32)
    tokens[:, 0] = torch.tensor(phases, dtype=torch.float32)
    supports = torch.tensor([[0.7, 0.6]] * len(dt), dtype=torch.float32)
    return PairFeatureBatch(pair133(tokens, torch.tensor(dt, dtype=torch.float32), supports))


def test_identity_preserves_bits_near_zero_and_saturation_without_second_call():
    source = RecordingPair()
    wrapper = CanonicalPair(source)
    values = batch([-0.0, 0.0, 1e-30, -100.0, 100.0], [0.1] * 5)
    output = wrapper(values)
    assert len(source.calls) == 1 and torch.equal(source.calls[0], values.features)
    assert torch.equal(output.view(torch.int32), values.features[:, 0].view(torch.int32))
    assert wrapper.last_trace["identity_mask"].all()


def test_canonicalization_rebuilds_all_time_columns_and_preserves_whole_mixed_batch():
    source = RecordingPair()
    wrapper = CanonicalPair(source)
    values = batch([0.02, -0.03, 0.0], [0.1, 0.2, 0.05])
    output = wrapper(values)
    assert [v.shape for v in source.calls] == [torch.Size([3, 133])] * 2
    torch.testing.assert_close(source.calls[0], values.features, rtol=0, atol=0)
    canonical = pair133(values.features[:, :128], torch.full((3,), 0.1), values.features[:, 131:])
    torch.testing.assert_close(source.calls[1], canonical, rtol=0, atol=0)
    assert output[0].view(torch.int32) == values.features[0, 0].view(torch.int32)
    assert output[1] < 0 and output[2] == 0
    torch.testing.assert_close(wrapper.last_trace["raw_corrected_phase"], output, rtol=0, atol=0)


@pytest.mark.parametrize("phase", [0.0, 0.01, -0.01, 1e-8, -1.0])
def test_fixed_token_inverse_rate_uses_measured_interval_without_floor(phase):
    wrapper = CanonicalPair(RecordingPair())
    values = batch([phase] * 3, [0.05, 0.1, 0.2])
    output = wrapper(values)
    reconstructed = -torch.expm1(-output) / torch.tensor(0.1)
    expected = -torch.expm1(-torch.tensor(phase)) / torch.tensor([0.05, 0.1, 0.2])
    torch.testing.assert_close(reconstructed, expected, rtol=3e-7, atol=1e-10)
    if phase == 1e-8:
        assert reconstructed.max() < 1 / 60  # No emission floor before correction.


def test_domain_rejection_is_whole_batch_and_records_failed_math():
    from operational.phase_timewarp.canonical_pair import CanonicalPhaseDomainError

    source = RecordingPair()
    wrapper = CanonicalPair(source)
    with pytest.raises(CanonicalPhaseDomainError, match="domain"):
        wrapper(batch([0.01, 2.0], [0.1, 0.05]))
    assert len(source.calls) == 2 and all(len(x) == 2 for x in source.calls)
    assert wrapper.last_trace["domain_valid"].tolist() == [True, False]
    assert wrapper.last_trace["corrected_q"][1] > 10


def test_input_validation_is_not_a_physical_phase_domain_rejection():
    from operational.phase_timewarp.canonical_pair import CanonicalPhaseDomainError

    wrapper = CanonicalPair(RecordingPair())
    values = batch([.1], [.1])
    values.features[0, 128] = -1
    with pytest.raises(ValueError) as error:
        wrapper(values)
    assert not isinstance(error.value, CanonicalPhaseDomainError)


@pytest.mark.parametrize("column,value", [(128, 0.0), (128, -0.1), (131, -0.1), (132, 1.1)])
def test_invalid_delta_and_support_rejected_before_forward(column, value):
    source = RecordingPair()
    wrapper = CanonicalPair(source)
    values = batch([0.01], [0.1])
    values.features[0, column] = value
    with pytest.raises(ValueError):
        wrapper(values)
    assert source.calls == [] and wrapper.last_trace == {}


def test_fp64_and_nonfinite_inputs_are_not_silently_coerced():
    wrapper = CanonicalPair(RecordingPair())
    values = batch([0.01], [0.1])
    with pytest.raises(ValueError, match="FP32"):
        wrapper(PairFeatureBatch(values.features.double()))
    values.features[0, 0] = float("nan")
    with pytest.raises(ValueError):
        wrapper(values)


def test_stale_trace_is_cleared_before_rejected_next_call():
    wrapper = CanonicalPair(RecordingPair())
    wrapper(batch([0.01], [0.1]))
    values = batch([0.01], [0.1])
    values.features[0, 128] = -1
    with pytest.raises(ValueError):
        wrapper(values)
    assert wrapper.last_trace == {}
