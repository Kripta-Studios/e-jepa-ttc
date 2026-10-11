"""External paths must use the same producer mask validation as sealed H/V."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from e_jepa_ttc.rgb_port.features import ProducerObservation
from operational.rgb_port_external_fix.queue import route
from operational.rgb_port_external_fix.runtime import observation_contract


def observation(steps=3):
    return ProducerObservation(
        token128=torch.ones(2, 128),
        diagnostics3=torch.zeros(2, 3),
        prediction_ttc=torch.ones(2),
        point_phase=torch.ones(2) * 0.1,
        support=torch.ones(2, steps),
        known=torch.tensor([True, False]),
        anchor_us=torch.tensor([100, 200], dtype=torch.int64),
        available_us=torch.tensor([100, 201], dtype=torch.int64),
    )


@pytest.mark.parametrize("steps", [2, 3])
def test_current_mask_preserves_input_and_features(steps):
    value = observation(steps)
    with pytest.raises(ValueError, match="known mask"):
        value.validate()
    previous = ProducerObservation.validate
    with observation_contract():
        value.validate()
        native = value.pair_input(torch.full((2,), 0.1))
        legacy = replace(value, known=value.known[:, None].expand_as(value.support))
        legacy.validate()
        torch.testing.assert_close(native, legacy.pair_input(torch.full((2,), 0.1)))
        assert value.known.shape == (2,)
        assert value.known.tolist() == [True, False]
    assert ProducerObservation.validate is previous


@pytest.mark.parametrize(
    "field,value",
    [
        ("known", torch.ones(2)),
        ("known", torch.ones(2, 4, dtype=torch.bool)),
        ("support", torch.ones(2, 3) * 2),
        ("point_phase", torch.tensor([float("nan"), 1.0])),
        ("available_us", torch.tensor([99, 200], dtype=torch.int64)),
    ],
)
def test_invalid_contracts_remain_rejected(field, value):
    with observation_contract(), pytest.raises((ValueError, TypeError, RuntimeError)):
        replace(observation(), **{field: value}).validate()


def test_restores_validator_on_error():
    before = ProducerObservation.validate
    with pytest.raises(RuntimeError), observation_contract():
        raise RuntimeError("test")
    assert ProducerObservation.validate is before


def test_routes_only_external_inference():
    for target in ("transfer", "cost_routes"):
        result = route(
            ["python", "-m", "operational.rgb_port." + target, "--device", "cpu"], Path("run")
        )
        assert result[2] == "operational.rgb_port_external_fix.runtime"
        assert result[-2:] == ["--device", "cpu"]
    original = ["python", "-m", "operational.rgb_port.train_heads"]
    assert route(original, Path("run")) == original


def test_wrapper_retains_fusion_runtime(tmp_path):
    from operational.rgb_port_external_fix import runtime

    argv = ["runtime", str(tmp_path), "operational.rgb_port.transfer", "predict-dev32"]
    original = ProducerObservation.validate

    def delegate():
        assert ProducerObservation.validate is not original
        observation().validate()

    with (
        patch.object(runtime, "validate"),
        patch.object(runtime.parent, "main", delegate),
        patch.object(runtime.sys, "argv", argv),
    ):
        runtime.main()
    assert ProducerObservation.validate is original


def test_recovery_preserves_endpoints_and_rebuilds_report():
    from operational.rgb_port_external_fix.recovery import retry_state

    state = {
        "tasks": {
            "DEV32_PREDICT": {"status": "FAILED", "completed_updates": 0, "attempts": 1},
            "PROFILE_ROUTES": {"status": "FAILED", "completed_updates": 0, "attempts": 1},
            "DEV32_SCORE": {"status": "SKIPPED_DEPENDENCY"},
            "REPORT": {"status": "COMPLETE", "completion_sha256": "old"},
            "F_TRUE": {"status": "COMPLETE", "completed_updates": 2500},
            "FCWD_RGB_STATUS": {"status": "BLOCKED_EXTERNAL"},
        }
    }
    updated, names = retry_state(state, [{"id": "DEV32_SCORE", "depends": ["DEV32_PREDICT"]}])
    assert set(names) == {"DEV32_PREDICT", "PROFILE_ROUTES", "DEV32_SCORE", "REPORT"}
    assert updated["tasks"]["F_TRUE"] == state["tasks"]["F_TRUE"]
    assert updated["tasks"]["FCWD_RGB_STATUS"] == state["tasks"]["FCWD_RGB_STATUS"]
    assert updated["tasks"]["DEV32_PREDICT"]["attempts"] == 1
    assert state["tasks"]["REPORT"]["status"] == "COMPLETE"


def test_cost_priority_preserves_task_contracts():
    from operational.rgb_port_external_fix.queue import prioritize_costs

    tasks = [
        {"id": "DEV32_PREDICT", "depends": ["V_EVALUATION"]},
        {"id": "PROFILE_ROUTES", "depends": ["V_PREDICTIONS"]},
    ]
    reordered = prioritize_costs(tasks)
    assert reordered == list(reversed(tasks))
    assert reordered[0] is tasks[1]
    assert prioritize_costs(reordered) == reordered
