from __future__ import annotations

import numpy as np
import torch

from e_jepa_ttc.data.stage61_pair_feature_cache import LocalTemporalFieldBatch
from e_jepa_ttc.models.local_temporal_phase_field import LocalTemporalPhaseField
from e_jepa_ttc.training.stage62_local_field import (
    derange_cross_track,
    global_pool_field,
    time_swap_field,
)


def test_local_field_topology_and_zero_proposal_replay() -> None:
    model = LocalTemporalPhaseField().eval()
    batch = LocalTemporalFieldBatch(
        torch.randn(5, 16, 34), torch.ones(5, 16, dtype=torch.bool), torch.full((5,), 0.2)
    )
    with torch.no_grad():
        output = model(batch)
    torch.testing.assert_close(output.benchmark_phase, batch.a5_phase)
    assert output.patch_weights.shape == (5, 16)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3382


def test_matched_interventions_preserve_a5_state() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(size=(12, 16, 34)).astype(np.float32)
    pooled = global_pool_field(values)
    swapped = time_swap_field(values)
    np.testing.assert_array_equal(pooled[:, :, 31:], values[:, :, 31:])
    np.testing.assert_array_equal(swapped[:, :, 31:], values[:, :, 31:])
    np.testing.assert_array_equal(swapped[:, :, 20:29], -values[:, :, 20:29])
    shuffled, permutation = derange_cross_track(
        values,
        sequence_ids=["s"] * 12,
        track_ids=[f"track-{index}" for index in range(12)],
        seed=7,
    )
    assert np.all(permutation != np.arange(12))
    np.testing.assert_array_equal(shuffled[:, :, 31:], values[:, :, 31:])
