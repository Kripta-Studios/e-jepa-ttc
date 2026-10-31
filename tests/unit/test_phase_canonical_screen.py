"""Compact PAIR replay preserves identity and propagates physical changes."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from e_jepa_ttc.rgb_port.features import event_phase17, pair133
from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.phase_timewarp.canonical_pair import CanonicalPair
from operational.phase_timewarp.screen import replay_pair_batches


class Pair(nn.Module):
    def forward(self, batch):
        return batch.features[:, 0]


def fixture():
    tokens = torch.zeros(2, 128)
    tokens[:, 0] = torch.tensor([0.02, 0.04])
    features = pair133(tokens, torch.tensor([0.1, 0.2]), torch.ones(2, 2))
    experts = torch.tensor([[0.03, 0.05, 0.02], [0.06, 0.07, 0.04]])
    experts[:, 2] = emitted_phase(tokens[:, 0])
    raw = event_phase17(torch.ones(2, 2), torch.ones(2, 3), torch.ones(2, 3), experts)
    arrays = {
        "call0.PAIR_E_MATCHED.input133": features.numpy(),
        "call0.delta": np.array([[0.1], [0.2]], np.float32),
        "call0.raw17": raw.numpy(),
        "call0.expert3": experts.numpy(),
    }
    calls = [
        {
            "call_id": 0,
            "prefix": "E",
            "status": "accepted",
            "observations": [
                {
                    "batch_index": i,
                    "source_accepted": True,
                    "anchor_us": (i + 1) * 100000,
                    "observation_id": f"event:{i}",
                }
                for i in range(2)
            ],
        }
    ]
    return calls, arrays


def test_original_exact_and_corrected_rebuilds_signed_and_absolute_differences():
    calls, arrays = fixture()
    original = replay_pair_batches({"PAIR_E_MATCHED": Pair()}, calls, arrays, "cpu")
    for field in ("raw17", "expert3"):
        np.testing.assert_array_equal(
            original["arrays"]["call0." + field], arrays["call0." + field]
        )
    changed = replay_pair_batches({"PAIR_E_MATCHED": CanonicalPair(Pair())}, calls, arrays, "cpu")
    raw = changed["arrays"]["call0.raw17"]
    np.testing.assert_array_equal(raw[0], arrays["call0.raw17"][0])
    np.testing.assert_array_equal(raw[:, :10], arrays["call0.raw17"][:, :10])
    assert raw[1, 10] != arrays["call0.raw17"][1, 10]
    np.testing.assert_array_equal(raw[:, 11], raw[:, 10] - raw[:, 8])
    np.testing.assert_array_equal(raw[:, 12], raw[:, 10] - raw[:, 9])
    np.testing.assert_array_equal(raw[:, 14:17], np.abs(raw[:, 11:14]))
    assert [r["anchor_us"] for r in changed["observations"]["event"]] == [100000, 200000]


def test_rejects_clock_or_batch_mapping_corruption():
    calls, arrays = fixture()
    bad = copy.deepcopy(calls)
    bad[0]["observations"].reverse()
    with pytest.raises(ValueError, match="identity"):
        replay_pair_batches({"PAIR_E_MATCHED": Pair()}, bad, arrays, "cpu")
    arrays["call0.delta"][1, 0] = 0.3
    with pytest.raises(AssertionError):
        replay_pair_batches({"PAIR_E_MATCHED": Pair()}, calls, arrays, "cpu")


def test_canonical_domain_failure_rejects_entire_batch_and_keeps_trace():
    calls, arrays = fixture()
    arrays["call0.PAIR_E_MATCHED.input133"][1, 0] = 2
    arrays["call0.PAIR_E_MATCHED.input133"][1, 128] = 0.01
    arrays["call0.delta"][1, 0] = 0.01
    result = replay_pair_batches({"PAIR_E_MATCHED": CanonicalPair(Pair())}, calls, arrays, "cpu")
    assert not result["observations"]["event"]
    assert result["calls"][0]["status"] == "rejected"
    assert not result["arrays"]["call0.trace.domain_valid"].all()


def test_empty_query_does_not_invent_native_observations():
    assert replay_pair_batches({}, [], {}, "cpu")["observations"] == {"event": [], "rgb": []}
