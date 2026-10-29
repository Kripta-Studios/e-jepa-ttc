from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.models.object_lhr import (
    ObjectCentricLHR,
    ObjectCentricLHRConfig,
    stable_ttc_from_log_ratio,
)


def test_stable_ttc_preserves_approaching_and_receding_signs() -> None:
    delta = torch.tensor([0.1, 0.1])
    # h1/h2 < 1 -> approaching -> positive TTC.
    # h1/h2 > 1 -> receding -> negative TTC.
    log_ratio = torch.log(torch.tensor([0.9, 1.1]))
    result = stable_ttc_from_log_ratio(
        log_ratio,
        delta,
        denominator_epsilon=1e-3,
        clip_seconds=60.0,
    )
    assert result[0].item() == pytest.approx(1.0, rel=1e-5)
    assert result[1].item() == pytest.approx(-1.0, rel=1e-5)


def test_stable_ttc_clamps_near_singular_ratio() -> None:
    result = stable_ttc_from_log_ratio(
        torch.tensor([0.0]),
        torch.tensor([0.1]),
        denominator_epsilon=1e-3,
        clip_seconds=60.0,
    )
    assert result.item() == pytest.approx(60.0)


def test_object_lhr_emits_two_positive_heights_and_masks() -> None:
    torch.manual_seed(5)
    model = ObjectCentricLHR(
        ObjectCentricLHRConfig(
            in_channels=4,
            embed_dim=16,
            patch_size=8,
            spatial_window=2,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            query_count=2,
            head_hidden_dim=16,
            mask_size=32,
        )
    ).eval()
    events = torch.randn(2, 2, 4, 32, 32)
    with torch.inference_mode():
        output = model(events, torch.tensor([0.1, 0.1]))
    assert output.visible_heights_px.shape == (2, 2)
    assert bool((output.visible_heights_px > 0).all())
    assert output.log_height_ratio.shape == (2,)
    assert output.ttc_mean_seconds.shape == (2,)
    assert output.endpoint_embeddings.shape == (2, 2, 16)
    assert output.mask_logits is not None
    assert output.mask_logits.shape == (2, 2, 1, 32, 32)


def test_temporal_endpoint_pooling_keeps_endpoint_identity() -> None:
    torch.manual_seed(17)
    model = ObjectCentricLHR(
        ObjectCentricLHRConfig(
            in_channels=2,
            embed_dim=8,
            patch_size=4,
            spatial_window=2,
            heads=2,
            spatial_depth=1,
            temporal_depth=1,
            query_count=2,
            head_hidden_dim=8,
            mask_decoder=False,
            mask_size=16,
        )
    ).eval()
    events = torch.zeros(1, 2, 2, 16, 16)
    events[:, 1] = 3.0
    with torch.inference_mode():
        output = model(events, torch.tensor([0.1]))
    assert not torch.equal(output.endpoint_embeddings[:, 0], output.endpoint_embeddings[:, 1])
