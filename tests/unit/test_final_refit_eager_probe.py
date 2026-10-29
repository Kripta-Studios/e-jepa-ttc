from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from operational.final_refit.eager_probe import (
    CASE_KINDS,
    CONTROL_POLICY,
    CONTROL_POLICY_SHA256,
    FIT_IDS,
    REGIMES,
    SCHEMA,
    _controlled_compare,
    _restore_rng,
    select_observed_cases,
    validate_receipt,
)


def test_observed_case_selection_covers_t2_t3_and_mixed_without_padding() -> None:
    source = SimpleNamespace(
        population_size=14,
        batch_keys=[("P", 2)] * 5 + [("P", 3)] * 5 + [("H", 2)] * 4,
    )

    cases = select_observed_cases(source, batch_size=4)

    assert tuple(item.kind for item in cases) == CASE_KINDS
    assert all(len(item.indices) == 4 for item in cases)
    assert cases[0].observed_keys == (("H", 2),)
    assert cases[1].observed_keys == (("P", 3),)
    assert {frames for _, frames in cases[2].observed_keys} == {2, 3}
    assert set(cases[2].indices) == set(cases[0].indices[:2] + cases[1].indices[:2])


def test_observed_case_selection_rejects_missing_temporal_signature() -> None:
    source = SimpleNamespace(population_size=4, batch_keys=[("P", 3)] * 4)

    with pytest.raises(ValueError, match="T2"):
        select_observed_cases(source, batch_size=4)


def _passing_receipt() -> dict:
    numerical = {
        "passed": True,
        "strict_original_threshold_passed": True,
        "strict_control_threshold_passed": True,
        "control_dominance_passed": True,
        "policy": "strict_candidate_reference_and_strict_reference_reference_control",
        "candidate_reference": {
            "names_equal": True,
            "shapes_equal": True,
            "finite": True,
            "max_abs": 0.0,
            "relative_l2": 0.0,
        },
        "reference_reference_control": {
            "names_equal": True,
            "shapes_equal": True,
            "finite": True,
            "max_abs": 0.0,
            "relative_l2": 0.0,
        },
    }
    comparisons = [
        {
            "update": update,
            "rng_equal": True,
            "sampler_generator_equal": True,
            **{
                field: dict(numerical)
                for field in (
                    "loss",
                    "components",
                    "raw_gradients",
                    "clip_norm",
                    "clipped_gradients",
                    "model_after_step",
                    "optimizer_after_step",
                    "scheduler_after_step",
                )
            },
        }
        for update in (1, 2)
    ]
    cases = [
        {
            "fit_id": fit_id,
            "regime": regime,
            "case_kind": kind,
            "candidate_technical_optimizer_steps": 2,
            "reference_technical_optimizer_steps": 2,
            "control_technical_optimizer_steps": 4,
            "control_policy_sha256": CONTROL_POLICY_SHA256,
            "batch_parity": True,
            "initial_state_equal": True,
            "resume_exact": True,
            "scientific_updates": 0,
            "comparisons": comparisons,
            "passed": True,
        }
        for fit_id in FIT_IDS
        for regime in REGIMES
        for kind in CASE_KINDS
    ]
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "scientific_updates": 0,
        "technical_optimizer_steps": 8 * len(cases),
        "updates_per_case_per_path": 2,
        "control_policy": CONTROL_POLICY,
        "control_policy_sha256": CONTROL_POLICY_SHA256,
        "runtime_admission": True,
        "producer_plan_runtime_admission": "PASS",
        "runtime_route": {
            "passed": True,
            "effective_batch_size": 32,
            "technical_optimizer_steps": 0,
            "scientific_updates": 0,
            "technical_forward_only_passes": 2,
        },
        "cases": cases,
    }


def test_receipt_validation_requires_complete_case_matrix_and_exact_accounting() -> None:
    receipt = _passing_receipt()
    validate_receipt(receipt)

    incomplete = {**receipt, "cases": receipt["cases"][:-1]}
    with pytest.raises(ValueError, match="coverage"):
        validate_receipt(incomplete)

    wrong_count = {**receipt, "technical_optimizer_steps": 1}
    with pytest.raises(ValueError, match="accounting"):
        validate_receipt(wrong_count)


def test_receipt_validation_rejects_scientific_update() -> None:
    receipt = {**_passing_receipt(), "scientific_updates": 1}

    with pytest.raises(ValueError, match="science accounting"):
        validate_receipt(receipt)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("batch_parity",), False),
        (("initial_state_equal",), False),
        (("resume_exact",), False),
        (("comparisons", 0, "rng_equal"), False),
        (("comparisons", 0, "sampler_generator_equal"), False),
        (("comparisons", 0, "raw_gradients", "passed"), False),
        (("comparisons", 0, "raw_gradients", "candidate_reference", "finite"), False),
    ],
)
def test_receipt_validation_recomputes_case_gates(path: tuple[object, ...], value: object) -> None:
    import copy

    receipt = copy.deepcopy(_passing_receipt())
    target = receipt["cases"][0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="gate"):
        validate_receipt(receipt)


def test_controlled_compare_rejects_nonfinite_and_invalid_control() -> None:
    import torch

    nonfinite = _controlled_compare(
        torch.tensor([float("nan")]),
        torch.tensor([float("nan")]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        atol=1.0,
        rtol=1.0,
    )
    mismatched_control = _controlled_compare(
        {"x": torch.tensor([1.0])},
        {"x": torch.tensor([1.1])},
        {"x": torch.tensor([1.0])},
        {"y": torch.tensor([2.0])},
        atol=0.0,
        rtol=0.0,
    )

    assert nonfinite["passed"] is False
    assert nonfinite["candidate_reference"]["finite"] is False
    assert mismatched_control["passed"] is False
    assert mismatched_control["reference_reference_control"]["names_equal"] is False


def test_receipt_validation_recomputes_reported_numerical_decision() -> None:
    import copy

    receipt = copy.deepcopy(_passing_receipt())
    comparison = receipt["cases"][0]["comparisons"][0]["raw_gradients"]
    comparison["strict_original_threshold_passed"] = False
    comparison["candidate_reference"]["max_abs"] = 2.0
    comparison["candidate_reference"]["relative_l2"] = 2.0
    comparison["reference_reference_control"]["max_abs"] = 1.0
    comparison["reference_reference_control"]["relative_l2"] = 1.0
    comparison["control_dominance_passed"] = True
    comparison["passed"] = True

    with pytest.raises(ValueError, match="gate"):
        validate_receipt(receipt)


def test_restore_rng_normalizes_serialized_states_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    import numpy as np
    import torch

    seen: dict[str, object] = {}
    monkeypatch.setattr(torch, "set_rng_state", lambda value: seen.setdefault("cpu", value))
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda value: seen.setdefault("cuda", value),
    )
    _restore_rng(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.zeros(4, dtype=torch.uint8),
            "cuda": [torch.ones(4, dtype=torch.uint8)],
        }
    )

    assert cast(torch.Tensor, seen["cpu"]).device.type == "cpu"
    assert all(item.device.type == "cpu" for item in cast(list[torch.Tensor], seen["cuda"]))


def test_controlled_compare_keeps_control_dominance_informative_only() -> None:
    import torch

    reference = {"weight": torch.tensor([1.0, 2.0])}
    control_left = {"weight": torch.tensor([1.0, 2.0])}
    control_right = {"weight": torch.tensor([1.0, 2.002])}

    bounded = _controlled_compare(
        {"weight": torch.tensor([1.0, 2.001])},
        reference,
        control_left,
        control_right,
        atol=1e-5,
        rtol=1e-5,
    )
    excessive = _controlled_compare(
        {"weight": torch.tensor([1.0, 2.004])},
        reference,
        control_left,
        control_right,
        atol=1e-5,
        rtol=1e-5,
    )

    assert bounded["strict_original_threshold_passed"] is False
    assert bounded["control_dominance_passed"] is True
    assert bounded["passed"] is False
    assert excessive["control_dominance_passed"] is False
    assert excessive["passed"] is False


def test_controlled_compare_keeps_original_strict_pass_independent_of_control() -> None:
    import torch

    exact = _controlled_compare(
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        atol=0.0,
        rtol=0.0,
    )

    assert exact["strict_original_threshold_passed"] is True
    assert exact["strict_control_threshold_passed"] is True
    assert exact["passed"] is True
    assert exact["policy"] == CONTROL_POLICY["numerical_rule"]


def test_eager_probe_source_has_no_compile_or_graph_api() -> None:
    source_path = Path(__file__).parents[2] / "operational/final_refit/eager_probe.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "compile" not in called
    assert "CUDAGraph" not in called
    assert "make_graphed_callables" not in called
