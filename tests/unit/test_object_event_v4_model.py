from __future__ import annotations

import torch

from e_jepa_ttc.models.highres_factorized import EJEPATubeletLHR, EJEPATubeletLHRConfig
from e_jepa_ttc.models.object_event_v4 import ObjectEventTTCV4, ObjectEventV4Config


def _config() -> ObjectEventV4Config:
    return ObjectEventV4Config(
        embed_dim=24,
        patch_size=4,
        spatial_window=2,
        heads=4,
        spatial_depth=1,
        temporal_depth=1,
        query_count=2,
        predictor_hidden_dim=32,
        event_head_hidden_dim=32,
        motion_hidden_dim=16,
        fusion_hidden_dim=16,
        dropout=0.0,
    )


def test_event_branch_is_independent_of_motion_and_reversal_is_exact() -> None:
    torch.manual_seed(7)
    model = ObjectEventTTCV4(_config()).eval()
    events = torch.randn(2, 3, 12, 16, 16)
    dt = torch.full((2,), 0.1)
    first = model(events, dt, torch.zeros(2, 18))
    second = model(events, dt, torch.randn(2, 18))
    reversed_output = model(torch.flip(events, dims=(1,)), dt, torch.zeros(2, 18))
    assert torch.allclose(first.event_expansion, second.event_expansion, atol=1e-7)
    assert torch.allclose(
        first.event_expansion,
        -reversed_output.event_expansion,
        atol=1e-6,
    )
    assert float(first.reversal_error.detach().max()) <= 1.0e-7
    assert bool((first.event_gate >= model.config.minimum_event_gate).all())


def test_adapted_level_transfer_slices_only_patch_channels() -> None:
    config = _config()
    model = ObjectEventTTCV4(config)
    source_config = EJEPATubeletLHRConfig(
        in_channels=21,
        embed_dim=config.embed_dim,
        patch_size=config.patch_size,
        spatial_window=config.spatial_window,
        heads=config.heads,
        spatial_depth=config.spatial_depth,
        temporal_depth=config.temporal_depth,
        temporal_mixer=config.temporal_mixer,
        merge_2x2=config.merge_2x2,
        global_attention=config.global_attention,
        memory_budget_gb=config.memory_budget_gb,
        pooling=config.pooling,
        query_count=config.query_count,
    )
    source = EJEPATubeletLHR(source_config)
    state = source.backbone_state_dict()
    report = model.load_adapted_pretrained_backbone(
        state, source.backbone_structural_config()
    )
    assert report["source_in_channels"] == 21
    assert report["target_in_channels"] == 12
    assert torch.equal(
        model.encoder.patch_embed.weight,
        source.patch_embed.weight[:, :12],
    )
