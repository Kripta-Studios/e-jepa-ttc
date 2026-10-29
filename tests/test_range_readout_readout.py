"""Synthetic raw/cache contracts, negative targets and explicit operator loading."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from e_jepa_ttc.distillation.dinov3_relational import local_cosine_relation_maps
from e_jepa_ttc.losses.causal_scale_ttc import CausalScaleTTCLossConfig
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.range_readout.operators import ResidualSpec
from e_jepa_ttc.range_readout.readout import VectorReadout, wrap
from operational.range_readout.cache_readout import capture, loss
from operational.range_readout.train_readout import load_endpoint
from operational.rgb_port.train_producers import _producer_loss


@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize("bound", [0.05, 0.20])
def test_raw_cached_loss_and_gradient_with_negatives(steps, bound):
    torch.manual_seed(9)
    model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=12,
            hidden_dim=8,
            geometry_dim=8,
            residual_depth=1,
            transport_enabled=True,
            transport_radius=1,
        )
    )
    model = wrap(model, ResidualSpec(bound), reset=False)
    final = model.residual.scorer[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.fill_(0.01)
    events = torch.rand(4, steps, 12, 32, 32)
    delta = torch.full((4,), 0.1)
    with torch.no_grad():
        raw = model(events, delta[:, None].expand(-1, steps - 1), return_dense_features=True)
        relation = local_cosine_relation_maps(raw.endpoint_dense_features[:, -2:])
    batch = SimpleNamespace(
        events=events,
        delta_t_s=delta,
        target_ttc_s=torch.tensor([0.5, 2.0, -2.0, -0.2]),
        boxes_xyxy=torch.tensor([1.0, 2.0, 25.0, 28.0]).expand(4, steps, 4).clone(),
        dinov3_relation_targets=torch.zeros_like(relation.values),
        dinov3_relation_valid=relation.valid,
    )
    with torch.no_grad():
        values, _ = capture(model, batch, labels=True)
    cfg = CausalScaleTTCLossConfig(
        log_ratio_tail_weight=1.3,
        log_ratio_tail_fraction=0.5,
        foreground_width_weight=0.2,
        foreground_center_weight=0.1,
        foreground_pair_ratio_weight=0.3,
    )
    model.residual.scorer.train()
    torch.manual_seed(12)
    a, _ = _producer_loss(model, cast(Any, batch), cfg)
    ga = torch.autograd.grad(a, tuple(model.residual.scorer.parameters()))
    torch.manual_seed(12)
    b, _ = loss(VectorReadout(model), values, cfg)
    gb = torch.autograd.grad(b, tuple(model.residual.scorer.parameters()))
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    for left, right in zip(ga, gb, strict=True):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match="targets"):
        VectorReadout(model)(values)


def test_checkpoint_operator_mismatch_rejected_before_loading(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save(
        {
            "identity": {
                "operator": {"bound": 0.05, "slope": 0.05, "operator": "equal_slope_tanh_v1"}
            }
        },
        path,
    )
    with pytest.raises(ValueError, match="operator"):
        load_endpoint(path, ResidualSpec(0.20))


def test_real_cache_cannot_use_technical_budget():
    from operational.range_readout.train_readout import fit

    fake = SimpleNamespace(binding={"binding": {"role": "P"}}, population=53890)
    with pytest.raises(ValueError, match="synthetic-only"):
        fit(fake, 0.20, technical="invalid_technical")


def test_incomplete_queue_cannot_open_validation():
    from operational.range_readout.evaluate import SCREEN, freeze_queue

    with pytest.raises(ValueError, match="four"):
        freeze_queue(SCREEN[:2], "screen")
