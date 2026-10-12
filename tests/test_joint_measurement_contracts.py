"""Prospective V18 integration regressions; no optimizer updates."""

import copy
from dataclasses import fields

import numpy as np
import pytest
import torch

from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch
from e_jepa_ttc.joint_measurement.loss import training_mass
from e_jepa_ttc.joint_measurement.model import bidirectional_multiscale
from operational.joint_measurement.native_loss import effective_native_loss, p2_geometry
from operational.joint_measurement.train import initialize, native_config, setup


def test_multiscale_valid_constant_cosines_are_not_diluted_by_padding():
    maps = torch.ones(1, 64, 8, 8)
    values, support = bidirectional_multiscale(maps, maps)
    torch.testing.assert_close(values[support], torch.ones_like(values[support]))


def fixture_batch(n=4, size=32):
    boxes = torch.tensor([4.0, 4.0, 24.0, 28.0]).repeat(n, 2, 1)
    target = torch.tensor([0.8, 2.0, 7.0, -3.0])[:n]
    return ObjectEventV4Batch(
        events=torch.rand(n, 2, 12, size, size),
        delta_t_s=torch.full((n,), 0.1),
        observable_motion=torch.zeros(n, 18),
        visible_heights_px=torch.full((n, 2), 24.0),
        target_ttc_s=target,
        boxes_xyxy=boxes,
        common_square_xyxy=torch.zeros(n, 4),
        sequence_ids=["synthetic"] * n,
        sample_tokens=[f"synthetic-{i}" for i in range(n)],
        track_ids=["synthetic"] * n,
        dinov3_relation_targets=torch.rand(n, 2, 6, size // 4, size // 4),
        dinov3_relation_valid=torch.ones(n, 2, 6, size // 4, size // 4, dtype=torch.bool),
    )


def split_batch(batch, count=2):
    return [
        ObjectEventV4Batch(
            **{
                f.name: (
                    getattr(batch, f.name)[i : i + count]
                    if getattr(batch, f.name) is not None
                    else None
                )
                for f in fields(batch)
            }
        )
        for i in range(0, len(batch.events), count)
    ]


def test_p2_preserves_both_real_geometry_annotations():
    batch = fixture_batch()
    assert p2_geometry(batch).valid.all()


def test_mass_preserves_equal_groups_and_registered_band_ratios():
    t = np.array([1.0, 2.0, 4.0, 8.0, -1.0, 1.0, -2.0])
    g = np.array(["a"] * 5 + ["b"] * 2)
    mass = training_mass(t, g, role="P")
    assert mass[g == "a"].sum() == pytest.approx(0.5)
    assert mass[:2].sum() == pytest.approx(0.25)
    assert mass[5] == pytest.approx(0.5 * 0.5 / 0.6)
    with pytest.raises(ValueError):
        training_mass(t, g, role="H")


def test_native_global_topk_and_denominators_match_full_batch():
    setup(7)
    first = initialize("NATIVE-P2-FT-S7", random_parent=True).eval()
    second = copy.deepcopy(first)
    batch = fixture_batch()
    _, cfg = native_config()
    one, comp1 = effective_native_loss(first, [batch], cfg, recompute=False)
    two, comp2 = effective_native_loss(second, split_batch(batch), cfg, recompute=False)
    torch.testing.assert_close(one, two, rtol=1e-5, atol=1e-5)
    for key in comp1:
        torch.testing.assert_close(comp1[key], comp2[key], rtol=2e-5, atol=2e-5)
    one.backward()
    two.backward()
    for p, q in zip(first.parameters(), second.parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-4)


def test_native_checkpoint_preserves_dropout_rng_and_gradients():
    setup(7)
    first = initialize("NATIVE-P2-FT-S7", random_parent=True).train()
    second = copy.deepcopy(first)
    batches = split_batch(fixture_batch())
    _, cfg = native_config()
    rng = torch.get_rng_state()
    one, _ = effective_native_loss(first, batches, cfg, recompute=False)
    one.backward()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    two, _ = effective_native_loss(second, batches, cfg, recompute=True)
    two.backward()
    assert torch.equal(after, torch.get_rng_state())
    torch.testing.assert_close(one, two, atol=0, rtol=0)
    for p, q in zip(first.parameters(), second.parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
