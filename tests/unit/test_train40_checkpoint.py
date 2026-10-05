"""Resume produces bit-identical dropout/Adam/scheduler/sampler trajectories on CPU."""

import copy
import random

import numpy as np
import pytest
import torch

from operational.train40_system.checkpoint import DurableState


def objects():
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 6), torch.nn.Dropout(0.25), torch.nn.Linear(6, 1)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    generator = torch.Generator().manual_seed(11)
    return model, optimizer, scheduler, generator


def updates(state, values, count):
    model, optimizer, scheduler, generator = values
    for _ in range(count):
        x = torch.rand(8, 4, generator=generator)
        target = np.random.rand() + random.random()
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - target).square().mean()
        loss.backward()
        state.begin_update()
        optimizer.step()
        state.commit_update()
        scheduler.step()


def contract():
    return {"updates_limit": 10, "recovery_upper_limit": 100, "data_sha256": "data"}


def test_exact_resume_and_conservative_lost_update_accounting(tmp_path):
    continuous = objects()
    uninterrupted = DurableState(tmp_path / "continuous", contract())
    updates(uninterrupted, continuous, 6)
    expected = copy.deepcopy(continuous[0].state_dict())
    expected_rng = torch.get_rng_state()
    expected_sampler = continuous[3].get_state()
    values = objects()
    first = DurableState(tmp_path / "resumed", contract())
    updates(first, values, 3)
    first.save(*values, {"epoch": 1, "position": 24}, status="RUNNING")
    updates(first, values, 1)  # A hard stop can lose this update, but it remains accounted.
    fresh = objects()
    restored = DurableState(tmp_path / "resumed", contract())
    cursor = restored.restore(*fresh)
    assert cursor == {"epoch": 1, "position": 24}
    assert restored.recovery_upper == 1
    updates(restored, fresh, 3)
    for key, value in expected.items():
        assert torch.equal(fresh[0].state_dict()[key], value)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(fresh[3].get_state(), expected_sampler)
    assert restored.committed == 6


def test_checkpoint_refuses_optimizer_in_flight_and_changed_contract(tmp_path):
    values = objects()
    state = DurableState(tmp_path, contract())
    state.save(*values, {}, status="READY")
    state.begin_update()
    with pytest.raises(RuntimeError, match="incomplete"):
        state.save(*values, {}, status="RUNNING")
    changed = DurableState(tmp_path, {**contract(), "data_sha256": "other"})
    with pytest.raises(ValueError, match="contract"):
        changed.restore(*values)
