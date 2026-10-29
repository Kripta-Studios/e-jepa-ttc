"""Corrupt retained endpoint evidence without executing optimizer updates."""

import copy

import pytest
import torch
from torch import nn

from operational.joint_measurement.verify_rgb_endpoint import inspect_payload


def fixture(frozen=False):
    model = nn.Module()
    model.encoder = nn.Linear(2, 2)
    model.readout = nn.Linear(2, 1)
    model.encoder.requires_grad_(not frozen)
    final = copy.deepcopy(model.state_dict())
    final["readout.weight"] += 0.1
    if not frozen:
        final["encoder.weight"] += 0.1
    generator = torch.Generator().manual_seed(7)
    curves = [
        {
            "update": step,
            "ids": torch.randint(8, (2,), generator=generator).tolist(),
            "loss": 0.1,
            "norm": 1.0,
        }
        for step in (1, 2, 3)
    ]
    parameters = [value for value in model.parameters() if value.requires_grad]
    payload = {
        "completed_updates": 3,
        "accumulation_index": 0,
        "model_state_dict": final,
        "cursor": {"curve": curves, "timings": [0.1, 0.1, 0.1]},
        "sampler_generator_state": generator.get_state(),
        "optimizer_state_dict": {
            "param_groups": [{"params": list(range(len(parameters)))}],
            "state": {
                index: {
                    "step": torch.tensor(3.0),
                    "exp_avg": torch.zeros_like(value),
                    "exp_avg_sq": torch.zeros_like(value),
                }
                for index, value in enumerate(parameters)
            },
        },
        "scheduler_state_dict": {"last_epoch": 3},
        "recovery_upper": 0,
    }
    return model, payload


def inspect(model, payload, frozen=False):
    return inspect_payload(
        model, payload, updates=3, population=8, batch_size=2, seed=7, frozen=frozen
    )


@pytest.mark.parametrize("frozen", [False, True])
def test_admitted_trajectory_checks_all_ids_and_states(frozen):
    result = inspect(*fixture(frozen), frozen=frozen)
    assert result["status"] == "PASS"
    assert result["sample_ids_regenerated"] == 6
    assert result["optimizer_updates_in_verification"] == 0
    assert result["frozen_encoder_byte_identical"] == frozen


def test_frozen_encoder_corruption_rejected():
    model, payload = fixture(True)
    payload["model_state_dict"]["encoder.weight"][0, 0] += 0.01
    with pytest.raises(ValueError, match="encoder/readout change"):
        inspect(model, payload, frozen=True)


def test_one_changed_in_range_sample_id_rejected():
    model, payload = fixture()
    payload["cursor"]["curve"][1]["ids"][0] ^= 1
    with pytest.raises(ValueError, match="sample sequence"):
        inspect(model, payload)


def test_wrong_final_sampler_state_rejected():
    model, payload = fixture()
    payload["sampler_generator_state"] = torch.Generator().manual_seed(8).get_state()
    with pytest.raises(ValueError, match="final sampler state"):
        inspect(model, payload)


@pytest.mark.parametrize("corruption", ["missing_parameter", "moment_shape", "step", "nan"])
def test_optimizer_state_corruption_rejected(corruption):
    model, payload = fixture()
    states = payload["optimizer_state_dict"]["state"]
    if corruption == "missing_parameter":
        del states[0]
    elif corruption == "moment_shape":
        states[0]["exp_avg"] = torch.zeros(1)
    elif corruption == "step":
        states[0]["step"] = torch.tensor(2.0)
    else:
        states[0]["exp_avg_sq"].fill_(float("nan"))
    with pytest.raises(ValueError, match="optimizer parameter|Adam"):
        inspect(model, payload)


def test_wrong_scheduler_step_rejected():
    model, payload = fixture()
    payload["scheduler_state_dict"]["last_epoch"] = 2
    with pytest.raises(ValueError, match="scheduler step"):
        inspect(model, payload)
