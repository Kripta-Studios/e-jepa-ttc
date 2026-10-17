from __future__ import annotations

import json
import math

import pandas as pd
import pytest
import torch

from operational.decision_recovery import gradient_diagnostics as gradients
from operational.decision_recovery.gradient_diagnostics import (
    _admit_existing_result,
    adamw_delta,
    clip_vectors,
    cosine,
    ht_band_masses,
    parameter_block_indices,
    parameter_blocks,
    select_panel,
    validate_adamw_state,
    validate_paired_rng_restores,
    vector_summary,
)


def test_panel_is_fixed_bounded_and_has_horvitz_thompson_counts() -> None:
    rows = []
    targets = {"crucial": 2.0, "small": 4.0, "long": 8.0, "negative": -2.0}
    for group in ("a", "b"):
        for band, target in targets.items():
            for index in range(7):
                rows.append(
                    {
                        "sample_token": f"{group}-{band}-{index}",
                        "group_id": group,
                        "target_ttc": target,
                    }
                )
    frame = pd.DataFrame(rows)
    first, second = select_panel(frame), select_panel(frame.sample(frac=1, random_state=91))

    def identity(panel: list[dict]) -> list[tuple[str, int, int]]:
        return sorted((x["sample_token"], x["cell_population"], x["cell_sample"]) for x in panel)

    assert identity(first) == identity(second)
    assert len(first) == 2 * 4 * 4
    assert {(x["cell_population"], x["cell_sample"]) for x in first} == {(7, 4)}


def test_panel_rejects_duplicate_or_out_of_contract_target() -> None:
    duplicate = pd.DataFrame(
        {"sample_token": ["x", "x"], "group_id": ["g", "g"], "target_ttc": [2.0, 4.0]}
    )
    with pytest.raises(ValueError, match="unique"):
        select_panel(duplicate)
    outside = pd.DataFrame({"sample_token": ["x"], "group_id": ["g"], "target_ttc": [0.05]})
    with pytest.raises(ValueError, match="outside"):
        select_panel(outside)


def test_parameter_blocks_are_exhaustive_and_strict() -> None:
    assert parameter_blocks(
        ["encoder.0.weight", "spatial.1.bias", "gates.0.weight", "spatial_weights.bias"]
    ) == {
        "encoder.0.weight": "encoder",
        "spatial.1.bias": "spatial_tower",
        "gates.0.weight": "head",
        "spatial_weights.bias": "head",
    }
    with pytest.raises(ValueError, match="Unregistered"):
        parameter_blocks(["other.weight"])
    assert parameter_block_indices(["encoder.0.weight", "spatial.1.bias", "gates.0.weight"]) == {
        "encoder": [0],
        "spatial_tower": [1],
        "head": [2],
    }
    with pytest.raises(ValueError, match="all contain"):
        parameter_block_indices(["encoder.0.weight", "spatial.1.bias"])


def test_vector_metrics_cosine_and_clipping() -> None:
    left = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    right = [torch.tensor([-3.0, -4.0]), torch.tensor([1.0])]
    assert vector_summary(left, [0, 1]) == {
        "l2_norm": 5.0,
        "has_signal": True,
        "finite": True,
        "parameter_tensors": 2,
        "zero_parameter_tensors": 1,
    }
    assert cosine(left, right, [0]) == pytest.approx(-1.0)
    assert cosine(left, right, [1]) is None
    clipped, norm, scale = clip_vectors(left)
    assert norm == pytest.approx(5.0)
    assert scale == pytest.approx(1 / (5.0 + 1e-6))
    assert math.sqrt(sum(float(x.square().sum()) for x in clipped)) < 1.0


def test_adamw_delta_matches_manual_formula_without_mutation() -> None:
    parameter = torch.tensor([2.0, -1.0])
    gradient = torch.tensor([0.5, -0.25])
    state = {
        "step": torch.tensor(4.0),
        "exp_avg": torch.tensor([0.1, -0.2]),
        "exp_avg_sq": torch.tensor([0.3, 0.4]),
    }
    group = {
        "lr": 1e-3,
        "eps": 1e-8,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "amsgrad": False,
        "maximize": False,
    }
    before = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }
    observed = adamw_delta(parameter, gradient, state, group)
    m = 0.9 * state["exp_avg"] + 0.1 * gradient
    v = 0.999 * state["exp_avg_sq"] + 0.001 * gradient.square()
    expected = -(1e-3 / (1 - 0.9**5)) * m / (v.sqrt() / math.sqrt(1 - 0.999**5) + 1e-8)
    expected -= 1e-3 * 0.01 * parameter
    assert torch.equal(observed, expected)
    assert all(torch.equal(state[key], value) for key, value in before.items())


def test_nonfinite_vector_is_visible_and_incomplete_adam_is_rejected() -> None:
    assert vector_summary([torch.tensor([float("nan")])], [0])["finite"] is False
    with pytest.raises(ValueError, match="complete"):
        adamw_delta(torch.ones(1), torch.ones(1), {}, {"amsgrad": False, "maximize": False})


def test_zero_gradient_is_reported_as_no_signal() -> None:
    summary = vector_summary([torch.zeros(2)], [0])
    assert summary["finite"] is True
    assert summary["has_signal"] is False
    assert summary["l2_norm"] == 0
    assert cosine([torch.zeros(2)], [torch.ones(2)], [0]) is None


def test_ht_band_masses_and_strict_adam_state_validation() -> None:
    panel = [
        {
            "band": band,
            "index": index,
            "cell_population": 8,
            "cell_sample": 4,
        }
        for index, band in enumerate(("crucial", "small", "long", "negative"))
    ]
    assert ht_band_masses(panel, [0.25, 0.15, 0.05, 0.05]) == {
        "crucial": 0.5,
        "small": 0.3,
        "long": 0.1,
        "negative": 0.1,
    }
    parameter = torch.ones(2)
    state = {
        "step": torch.tensor(12000.0),
        "exp_avg": torch.zeros(2),
        "exp_avg_sq": torch.ones(2),
    }
    validate_adamw_state(parameter, state, expected_step=12000)
    with pytest.raises(ValueError, match="shape"):
        validate_adamw_state(
            parameter,
            {**state, "exp_avg": torch.zeros(3)},
            expected_step=12000,
        )


def test_rng_restores_and_existing_result_receipt_are_fail_closed(tmp_path, monkeypatch) -> None:
    validate_paired_rng_restores([True, True, True, True])
    with pytest.raises(ValueError, match="RNG"):
        validate_paired_rng_restores([True, False, True, True])

    registration = tmp_path / "REGISTRATION.json"
    registration.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gradients, "REGISTRATION", registration)
    output = tmp_path / "RESULT.json"
    result = {
        "status": "COMPLETE_ZERO_UPDATE_DIAGNOSTIC",
        "registration_sha256": gradients.common.sha(registration),
        "checkpoint_sha256": "checkpoint",
        "optimizer_updates": 0,
    }
    output.write_text(json.dumps(result), encoding="utf-8")
    complete = {
        "status": result["status"],
        "optimizer_updates": 0,
        "result_sha256": gradients.common.sha(output),
    }
    output.with_name("COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    assert _admit_existing_result(result, {"checkpoint_sha256": "checkpoint"}, output) == result
    complete["optimizer_updates"] = 1
    output.with_name("COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        _admit_existing_result(result, {"checkpoint_sha256": "checkpoint"}, output)
