from __future__ import annotations

import torch

from e_jepa_ttc.models.object_event_v4_19 import dense_flow_scores
from e_jepa_ttc.models.object_event_v4_20 import BoxPseudoFlowRefiner, ObjectEventV420Config
from e_jepa_ttc.training.object_event_v4_20 import (
    ObjectEventV420LossConfig,
    box_affine_pseudoflow,
    pseudoflow_loss,
)


def test_box_affine_pseudoflow_recovers_translation() -> None:
    boxes = torch.tensor([[[0, 0, 1, 1], [20, 20, 60, 60], [24, 18, 64, 58]]], dtype=torch.float32)
    flow, mask = box_affine_pseudoflow(
        boxes, source_height=100, source_width=100, target_height=20, target_width=20,
        first_index=1, second_index=2,
    )
    selected = flow[0, :, mask[0].bool()]
    assert torch.allclose(selected[0].mean(), torch.tensor(0.8), atol=1.0e-5)
    assert torch.allclose(selected[1].mean(), torch.tensor(-0.4), atol=1.0e-5)


def test_box_affine_pseudoflow_expansion_has_positive_divergence() -> None:
    boxes = torch.tensor([[[0, 0, 1, 1], [30, 30, 70, 70], [20, 20, 80, 80]]], dtype=torch.float32)
    flow, mask = box_affine_pseudoflow(
        boxes, source_height=100, source_width=100, target_height=32, target_width=32,
        first_index=1, second_index=2,
    )
    div, _, _ = dense_flow_scores(
        flow[:, 0], flow[:, 1], mask, torch.ones_like(mask),
        foreground_floor=1.0e-6, confidence_floor=1.0e-6,
    )
    assert float(div[0]) > 0.5


def test_reverse_pseudoflow_has_opposite_divergence_orientation() -> None:
    boxes = torch.tensor([[[0, 0, 1, 1], [30, 30, 70, 70], [20, 20, 80, 80]]], dtype=torch.float32)
    forward, mf = box_affine_pseudoflow(
        boxes, source_height=100, source_width=100, target_height=32, target_width=32,
        first_index=1, second_index=2,
    )
    reverse, mr = box_affine_pseudoflow(
        boxes, source_height=100, source_width=100, target_height=32, target_width=32,
        first_index=2, second_index=1,
    )
    df, _, _ = dense_flow_scores(
        forward[:, 0], forward[:, 1], mf, torch.ones_like(mf),
        foreground_floor=1.0e-6, confidence_floor=1.0e-6,
    )
    dr, _, _ = dense_flow_scores(
        reverse[:, 0], reverse[:, 1], mr, torch.ones_like(mr),
        foreground_floor=1.0e-6, confidence_floor=1.0e-6,
    )
    assert float(df[0]) > 0.0
    assert float(dr[0]) < 0.0


def test_refiner_is_raw_flow_identity_at_initialisation() -> None:
    torch.manual_seed(1)
    model = BoxPseudoFlowRefiner(ObjectEventV420Config(hidden_dim=8))
    inputs = torch.randn(3, 5, 9, 9)
    refined, residual = model(inputs)
    assert torch.allclose(residual, torch.zeros_like(residual))
    assert torch.allclose(refined, inputs[:, :2])


def test_refiner_residual_is_bounded() -> None:
    model = BoxPseudoFlowRefiner(ObjectEventV420Config(hidden_dim=8, residual_limit=0.75))
    with torch.no_grad():
        model.net[-1].weight.fill_(10.0)
    inputs = torch.ones(2, 5, 8, 8)
    _, residual = model(inputs)
    assert float(residual.detach().abs().max()) <= 0.750001


def test_pseudoflow_loss_is_finite_and_backpropagates() -> None:
    torch.manual_seed(3)
    model = BoxPseudoFlowRefiner(ObjectEventV420Config(hidden_dim=8))
    f = torch.randn(2, 5, 12, 12)
    r = torch.randn(2, 5, 12, 12)
    boxes = torch.tensor(
        [
            [[0, 0, 1, 1], [20, 20, 60, 60], [18, 18, 64, 64]],
            [[0, 0, 1, 1], [25, 20, 55, 65], [27, 22, 57, 67]],
        ],
        dtype=torch.float32,
    )
    tf, mf = box_affine_pseudoflow(
        boxes, source_height=100, source_width=100, target_height=12, target_width=12,
        first_index=1, second_index=2,
    )
    tr, mr = box_affine_pseudoflow(
        boxes, source_height=100, source_width=100, target_height=12, target_width=12,
        first_index=2, second_index=1,
    )
    pf, rf = model(f)
    pr, rr = model(r)
    loss, components = pseudoflow_loss(
        pf, rf, pr, rr, tf, mf, tr, mr, config=ObjectEventV420LossConfig()
    )
    assert torch.isfinite(loss)
    assert set(components) == {"flow", "divergence", "residual"}
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
