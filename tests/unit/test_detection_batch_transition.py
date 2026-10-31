"""Exposure conservation and continuation contracts for batch-four to batch-eight."""

import copy

import pytest
import torch

from e_jepa_ttc.detection.batch_transition import (
    ExposureLRScheduler,
    batch_size_at,
    build_plan,
    exposure_units,
)
from e_jepa_ttc.detection.campaign_training import ResumablePermutationSampler, UpdateLRScheduler


def test_all_exposures_and_milestones_are_conserved():
    plan = build_plan(6144, 70279, [17570, 26355, 35140, 43924, 52709, 61494, 70279])
    batches = [batch_size_at(plan, u) for u in range(6145, plan["total_updates"] + 1)]
    assert set(batches) == {4, 8}
    assert 6144 * 4 + sum(batches) == 281116
    for old, new in plan["selection_update_map"].items():
        assert exposure_units(plan, new) == int(old)


def test_lr_and_optimizer_state_are_preserved_and_new_resume_is_exact():
    plan = build_plan(8, 101, [21, 50, 101])
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    original = UpdateLRScheduler(
        optimizer, total_updates=101, kind="warmup_cosine", warmup_fraction=0.05, minimum_ratio=0.05
    )
    for update in range(1, 9):
        original.apply(update)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    new = ExposureLRScheduler(
        optimizer,
        plan=plan,
        total_updates=plan["total_updates"],
        kind="warmup_cosine",
        warmup_fraction=0.05,
        minimum_ratio=0.05,
    )
    new.load_state_dict(original.state_dict())
    optimizer.load_state_dict(state)
    for key, value in state["state"][0].items():
        assert torch.equal(value, optimizer.state_dict()["state"][0][key])
    for update in range(9, plan["total_updates"] + 1):
        lr = new.apply(update)[0]
        assert lr == 2e-4 * original.ratio_for(exposure_units(plan, update))
        saved = new.state_dict()
        new.load_state_dict(saved)
        assert new.state_dict() == saved


def test_invalid_plan_and_wrong_scheduler_clock_fail():
    with pytest.raises(ValueError):
        build_plan(21, 101, [21, 101])
    plan = build_plan(8, 101, [21, 101])
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.001)
    scheduler = ExposureLRScheduler(
        optimizer,
        plan=plan,
        total_updates=plan["total_updates"],
        kind="constant",
        warmup_fraction=0,
        minimum_ratio=0.05,
    )
    scheduler.apply(9)
    state = scheduler.state_dict()
    state["original"]["update"] -= 1
    with pytest.raises(ValueError, match="clocks"):
        scheduler.load_state_dict(state)


def test_batch_transition_preserves_order_across_epoch_and_partial_batch_boundaries():
    plan = build_plan(4, 23, [6, 11, 23])
    reference = ResumablePermutationSampler(23, 7)
    candidate = ResumablePermutationSampler(23, 7)
    original = [i for _ in range(23) for i in reference.next_batch(4)]
    switched = [
        i
        for u in range(1, plan["total_updates"] + 1)
        for i in candidate.next_batch(batch_size_at(plan, u))
    ]
    assert switched == original
    assert reference.epoch == candidate.epoch == 4
    assert reference.position == candidate.position == 0
    assert torch.equal(reference.generator.get_state(), candidate.generator.get_state())
