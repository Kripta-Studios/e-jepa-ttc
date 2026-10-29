"""Fusion engine contracts without optimizer updates (accounted QA is separate)."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from operational.joint_measurement.fusion_engine import FusionRecipe, backward, segment
from operational.joint_measurement.fusion_engine_qa import fixture
from operational.joint_measurement.fusion_model import RGBAnchorComplement


def recipe():
    return FusionRecipe(7, "TRUE", 10, 8, 16, 0.03, 1e-3, 0.01, 1.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", -1),
        ("updates", 0),
        ("batch_size", 1.5),
        ("hidden", True),
        ("mode", "UNKNOWN"),
        ("residual_scale", float("nan")),
        ("learning_rate", 0.0),
        ("weight_decay", -0.1),
        ("clip_norm", float("inf")),
    ],
)
def test_invalid_recipes_rejected_before_updates(field, value):
    with pytest.raises(ValueError):
        replace(recipe(), **{field: value}).validate()


def test_all_missing_batch_has_constant_loss_and_zero_parameter_gradients():
    bank = fixture()
    batch = bank.batch(torch.tensor([1, 4, 6]))
    model = RGBAnchorComplement(6, 6, hidden=16)
    target = torch.tensor([0.2, -0.1, 0.03])
    result = backward(model, batch, target, torch.full((3,), 1 / 8), 8)
    assert result["loss"] > 0 and result["rgb_rows"] == 0
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in model.parameters())
    assert torch.equal(model(**batch)["ttc"], batch["event_ttc"])


def test_mixed_batch_trains_complement_without_learning_from_missing_rgb():
    bank = fixture()
    batch = bank.batch(torch.arange(8))
    model = RGBAnchorComplement(6, 6, hidden=16)
    result = backward(model, batch, torch.full((8,), 0.4), torch.full((8,), 1 / 8), 8)
    assert result["rgb_rows"] == 5
    assert model.network[-1].bias.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("error", ["role", "mass", "target", "binding", "limit"])
def test_invalid_admission_creates_no_journal_or_optimizer_updates(tmp_path, error):
    bank = fixture()
    target = np.ones(8, np.float32)
    masses = np.full(8, 1 / 8, np.float64)
    binding, limit = {"qa": "unexecuted"}, 1
    if error == "role":
        bank.role = "V"
    elif error == "mass":
        masses *= 2
    elif error == "target":
        target[0] = 0
    elif error == "binding":
        binding = {}
    else:
        limit = 11
    with pytest.raises(ValueError):
        segment(
            tmp_path / "fit",
            bank,
            target,
            masses,
            recipe(),
            binding=binding,
            limit=limit,
            deadline=float("inf"),
        )
    assert not (tmp_path / "fit").exists()
