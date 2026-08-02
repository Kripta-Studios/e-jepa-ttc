from __future__ import annotations

import pytest
import torch

from e_jepa_ttc.models.highres_factorized import (
    EJEPATubeletLHR,
    EJEPATubeletLHRConfig,
    TemporalOnlyAttention,
    TheoreticalOOMError,
    make_patch_geometry,
    normalized_patch_coordinates,
    pad_to_patch_grid,
    space_to_depth_2x2,
)


def test_padding_keeps_right_bottom_pixels_and_marks_border_patches() -> None:
    inputs = torch.zeros(1, 2, 1, 17, 19)
    inputs[:, :, :, -1, -1] = 7.0
    padded, valid, geometry = pad_to_patch_grid(inputs, 8)

    assert geometry.padded_height == 24
    assert geometry.padded_width == 24
    assert padded[0, 0, 0, 16, 18].item() == 7.0
    assert padded[0, 1, 0, 16, 18].item() == 7.0
    assert valid.shape == (1, 2, 3, 3)
    assert bool(valid[:, :, -1, -1].all())
    assert geometry.padded_pixels == 253


def test_space_to_depth_propagates_any_valid_child_patch() -> None:
    tokens = torch.arange(1 * 1 * 3 * 3 * 2, dtype=torch.float32).reshape(1, 1, 3, 3, 2)
    valid = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    valid[:, :, 0, 0] = True
    merged, merged_mask = space_to_depth_2x2(tokens, valid)

    assert merged.shape == (1, 1, 2, 2, 8)
    assert merged_mask.shape == (1, 1, 2, 2)
    assert bool(merged_mask[:, :, 0, 0].item())
    assert int(merged_mask.sum()) == 1


def test_space_to_depth_zeroes_invalid_child_values_before_projection() -> None:
    tokens = torch.full((1, 1, 2, 2, 1), 9.0)
    valid = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    valid[:, :, 0, 0] = True
    merged, merged_mask = space_to_depth_2x2(tokens, valid)

    assert torch.equal(merged[0, 0, 0, 0], torch.tensor([9.0, 0.0, 0.0, 0.0]))
    assert bool(merged_mask[0, 0, 0, 0])


def test_post_merge_coordinates_follow_emitted_odd_grid_axis() -> None:
    model = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=2,
            embed_dim=16,
            patch_size=8,
            spatial_window=2,
            heads=4,
            temporal_depth=1,
            merge_2x2=True,
        )
    ).eval()
    with torch.inference_mode():
        features = model.forward_features(torch.randn(1, 2, 2, 17, 19))

    assert features.tokens.shape[2] == 4
    assert (features.encoded_grid_height, features.encoded_grid_width) == (2, 2)
    assert features.post_merge_patch_coordinates.shape == (4, 2)
    assert torch.allclose(
        features.post_merge_patch_coordinates,
        torch.tensor([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]),
    )
    assert torch.equal(features.patch_coordinates, features.post_merge_patch_coordinates)
    assert torch.allclose(
        normalized_patch_coordinates(2, 2, device=torch.device("cpu"), dtype=torch.float32),
        features.post_merge_patch_coordinates,
    )


def test_r4_has_960_spatial_patches_and_no_global_attention() -> None:
    geometry = make_patch_geometry(192, 320, 8)
    assert (geometry.grid_height, geometry.grid_width, geometry.patch_count) == (24, 40, 960)
    model = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=4,
            embed_dim=32,
            patch_size=8,
            spatial_window=8,
            heads=4,
            spatial_depth=1,
            temporal_depth=1,
            merge_2x2=True,
        )
    ).eval()
    with torch.inference_mode():
        output = model.forward_features(torch.randn(1, 5, 4, 192, 320))
    assert output.tokens.shape == (1, 5, 240, 32)
    assert output.valid_patch_mask.all()
    assert int(output.diagnostics["tokens_before_merge"]) == 4800
    assert int(output.diagnostics["tokens_after_merge"]) == 1200


def test_global_r4_oom_guard_fails_before_global_attention() -> None:
    model = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=4,
            embed_dim=32,
            patch_size=8,
            heads=4,
            temporal_depth=1,
            merge_2x2=False,
            global_attention=True,
        )
    ).eval()
    with pytest.raises(TheoreticalOOMError, match="Global attention is forbidden"):
        model.forward_features(torch.randn(1, 5, 4, 192, 320))


def test_global_r4_oom_guard_runs_before_patch_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    model = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=4,
            embed_dim=32,
            patch_size=8,
            heads=4,
            temporal_depth=1,
            global_attention=True,
        )
    ).eval()

    def fail_if_called(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("patch embedding must not run before the OOM guard")

    monkeypatch.setattr(model.patch_embed, "forward", fail_if_called)
    with pytest.raises(TheoreticalOOMError, match="Global attention is forbidden"):
        model.forward_features(torch.randn(1, 5, 4, 192, 320))


def test_kda_mixes_time_only_and_preserves_past_outputs() -> None:
    torch.manual_seed(11)
    model = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=4,
            embed_dim=32,
            patch_size=8,
            heads=4,
            temporal_depth=1,
            temporal_mixer="kda",
            merge_2x2=True,
        )
    ).eval()
    inputs = torch.randn(1, 3, 4, 32, 40)
    changed = inputs.clone()
    changed[:, 2] += 100.0
    with torch.inference_mode():
        first = model.forward_features(inputs).tokens
        second = model.forward_features(changed).tokens
    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-5)


def test_native_cache_resolution_is_not_called_high_resolution() -> None:
    model = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(in_channels=4, embed_dim=32, heads=4, temporal_depth=1)
    ).eval()
    with torch.inference_mode():
        output = model.forward_features(torch.randn(1, 2, 4, 90, 160))
    assert output.geometry.source_height == 90
    assert output.geometry.source_width == 160
    assert (output.geometry.grid_height, output.geometry.grid_width) == (6, 10)


def test_temporal_mixer_is_invariant_to_patch_permutation() -> None:
    torch.manual_seed(19)
    mixer = TemporalOnlyAttention(32, heads=4, depth=1).eval()
    tokens = torch.randn(2, 5, 7, 32)
    valid = torch.ones(2, 5, 7, dtype=torch.bool)
    permutation = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    inverse = torch.argsort(permutation)
    with torch.inference_mode():
        reference = mixer(tokens, valid)
        permuted = mixer(tokens[:, :, permutation], valid[:, :, permutation])
    assert torch.allclose(reference, permuted[:, :, inverse], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("temporal_mixer", ["block_causal", "kda"])
def test_temporal_padding_holes_do_not_affect_features_or_query_readout(
    temporal_mixer: str,
) -> None:
    torch.manual_seed(61)
    model = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=4,
            embed_dim=16,
            patch_size=8,
            spatial_window=2,
            heads=4,
            temporal_depth=1,
            temporal_mixer=temporal_mixer,
            merge_2x2=True,
            pooling="query",
            query_count=2,
        )
    ).eval()
    inputs = torch.randn(1, 4, 4, 16, 16)
    valid_temporal = torch.tensor([[True, False, True, True]])
    changed = inputs.clone()
    changed[:, 1] = 1.0e6
    with torch.inference_mode():
        reference = model(inputs, valid_temporal_mask=valid_temporal)
        result = model(changed, valid_temporal_mask=valid_temporal)
    assert not reference.valid_patch_mask[:, 1].any()
    assert torch.allclose(reference.tokens, result.tokens, atol=1e-6, rtol=1e-6)
    assert torch.allclose(reference.embedding, result.embedding, atol=1e-6, rtol=1e-6)


def test_query_is_default_highres_readout() -> None:
    model = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(in_channels=4, embed_dim=16, heads=4, temporal_depth=1)
    )
    assert model.config.pooling == "query"
    assert model.query_tokens is not None


def test_highres_ttc_head_supports_negative_signed_predictions() -> None:
    model = EJEPATubeletLHR(
        EJEPATubeletLHRConfig(
            in_channels=4,
            embed_dim=16,
            heads=4,
            temporal_depth=1,
        )
    ).eval()
    with torch.no_grad():
        for parameter in model.ttc_head.parameters():
            parameter.zero_()
        final_layer = model.ttc_head[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        final_layer.bias.fill_(-2.0)
    with torch.inference_mode():
        output = model(torch.randn(1, 2, 4, 32, 40))
    assert output.ttc_mean_seconds.item() == pytest.approx(-2.0)
