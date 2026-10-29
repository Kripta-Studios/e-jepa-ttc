from __future__ import annotations

import torch

from e_jepa_ttc.models.object_signed_expansion import (
    ObjectCentricSignedExpansionTTC,
    ObjectSignedExpansionConfig,
    antisymmetric_pair_score,
    expansion_to_log_ratio,
    geometry_prior_from_motion,
    reverse_observable_motion,
    safe_ttc_from_expansion,
)


def _config() -> ObjectSignedExpansionConfig:
    return ObjectSignedExpansionConfig(
        in_channels=3,
        embed_dim=24,
        patch_size=8,
        spatial_window=2,
        heads=4,
        spatial_depth=1,
        temporal_depth=1,
        query_count=2,
        adapter_hidden_dim=24,
        motion_hidden_dim=16,
        predictor_hidden_dim=32,
        pair_hidden_dim=32,
    )


def _inputs(batch: int = 3) -> dict[str, torch.Tensor]:
    motion = torch.zeros(batch, 18)
    motion[:, 7] = torch.tensor([0.1, -0.1, 0.04])[:batch]
    context = torch.zeros_like(motion)
    return {
        "events": torch.randn(batch, 2, 3, 32, 32),
        "delta_t_s": torch.full((batch,), 0.1),
        "observable_motion": motion,
        "jepa_context_motion": context,
        "precontext_motion_valid": torch.tensor([True, False, True])[:batch],
    }


def test_geometry_prior_matches_height_rate_identity() -> None:
    motion = torch.zeros(2, 18)
    motion[:, 7] = torch.tensor([0.1, -0.1])
    dt = torch.full((2,), 0.1)
    expansion, ratio = geometry_prior_from_motion(
        motion,
        dt,
        log_rate_scale=5.0,
        max_abs_expansion=0.25,
    )
    expected_ratio = torch.tensor([-0.05, 0.05])
    assert torch.allclose(ratio, expected_ratio)
    assert torch.allclose(expansion, 1.0 - torch.exp(expected_ratio))


def test_reverse_motion_is_involutive() -> None:
    motion = torch.randn(4, 18)
    assert torch.equal(reverse_observable_motion(reverse_observable_motion(motion)), motion)


def test_safe_ttc_and_ratio_keep_continuous_sign() -> None:
    expansion = torch.tensor([0.05, -0.05])
    dt = torch.tensor([0.1, 0.1])
    ttc = safe_ttc_from_expansion(
        expansion,
        dt,
        minimum_abs_expansion=1.0e-4,
        clip_seconds=60.0,
    )
    assert torch.allclose(ttc, torch.tensor([2.0, -2.0]))
    ratio = expansion_to_log_ratio(expansion, epsilon=1.0e-6)
    assert torch.allclose(ratio, torch.log1p(-expansion))


def test_model_starts_from_observable_geometry_prior() -> None:
    model = ObjectCentricSignedExpansionTTC(_config()).eval()
    inputs = _inputs()
    with torch.no_grad():
        output = model(**inputs)
        prior, _ = geometry_prior_from_motion(
            inputs["observable_motion"],
            inputs["delta_t_s"],
            log_rate_scale=model.config.box_log_rate_scale,
            max_abs_expansion=model.config.max_abs_expansion,
        )
    assert output.signed_expansion.shape == (3,)
    assert output.pair_embedding.shape == (3, 32)
    assert torch.allclose(output.signed_expansion, prior, atol=1.0e-6)
    assert torch.equal(output.ttc_mean_seconds < 0, prior < 0)


def test_ordered_scores_are_projected_to_odd_component() -> None:
    forward = torch.tensor([0.4, -0.2])
    reverse = torch.tensor([-0.1, 0.3])
    odd = antisymmetric_pair_score(forward, reverse)
    swapped = antisymmetric_pair_score(reverse, forward)
    assert torch.equal(odd, -swapped)


def test_target_encoder_is_frozen_and_ema_updated() -> None:
    model = ObjectCentricSignedExpansionTTC(_config())
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    online = next(model.encoder.parameters())
    target = next(model.target_encoder.parameters())
    before = target.detach().clone()
    with torch.no_grad():
        online.add_(1.0)
    model.update_target_encoder(0.5)
    assert torch.allclose(target, before + 0.5)
    model.train()
    assert not model.target_encoder.training
