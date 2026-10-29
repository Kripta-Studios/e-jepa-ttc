"""CPU checks of the two-call coordinate intervention and its immutable controls."""

from __future__ import annotations

import ast
import inspect
import sys
from copy import deepcopy
from typing import Any

import pytest
import torch

from e_jepa_ttc.models import local_transport
from scripts.diagnose_train40_linspace import (
    CANDIDATE,
    coordinate_probe,
    diagnostic_gate,
    make_linspace_probe,
    make_native_linspace_physical,
    native_summary_linspace_operator,
    rewrite_summary_linspaces,
    validate_cohort,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("steps", [1, 16, 32])
def test_native_coordinates_preserve_dtype_device_values_and_operator_schema(
    dtype: torch.dtype, steps: int
) -> None:
    like = torch.arange(12, dtype=dtype).reshape(3, 4)[:, ::2]
    before = like.clone()
    operation = native_summary_linspace_operator()
    actual = operation(like, steps)
    expected = torch.linspace(-0.5, 0.5, steps, dtype=dtype, device=like.device)
    assert actual.dtype == dtype and actual.device == like.device
    assert torch.equal(actual, expected)
    assert torch.equal(like, before)
    assert actual.untyped_storage().data_ptr() != like.untyped_storage().data_ptr()
    assert torch.library.opcheck(
        operation, (like, steps), test_utils=("test_schema", "test_faketensor")
    ) == {"test_schema": "SUCCESS", "test_faketensor": "SUCCESS"}


@pytest.mark.parametrize("dtype,steps", [(torch.int32, 16), (torch.float32, 0)])
def test_native_coordinates_reject_invalid_inputs(dtype: torch.dtype, steps: int) -> None:
    with pytest.raises(ValueError, match="floating input and positive steps"):
        native_summary_linspace_operator()(torch.zeros(1, dtype=dtype), steps)


def test_ast_rewrites_exactly_the_two_coordinate_assignments() -> None:
    source = inspect.getsource(local_transport._summarise_one)
    parsed, digest = rewrite_summary_linspaces(source)
    restored = deepcopy(parsed)
    sites = []
    for node in ast.walk(restored):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "_NATIVE_SUMMARY_LINSPACE":
                size = ast.unparse(call.args[1])
                sites.append(size)
                assert ast.unparse(call.args[0]) == "forward.dx"
                node.value = ast.parse(
                    f"torch.linspace(-0.5, 0.5, {size}, device=device, dtype=dtype)",
                    mode="eval",
                ).body
    assert sorted(sites) == ["height", "width"]
    assert ast.dump(restored) == ast.dump(ast.parse(source))
    assert len(digest) == 64


@pytest.mark.parametrize("change", ["endpoint", "dtype", "name", "missing", "extra"])
def test_ast_guard_rejects_coordinate_callsite_drift(change: str) -> None:
    source = inspect.getsource(local_transport._summarise_one)
    if change == "endpoint":
        source = source.replace(
            "torch.linspace(-0.5, 0.5, width", "torch.linspace(-0.4, 0.5, width"
        )
    elif change == "dtype":
        source = source.replace(
            "device=device, dtype=dtype)", "device=device, dtype=torch.float64)"
        )
    elif change == "name":
        source = source.replace("x = torch.linspace", "xx = torch.linspace")
    elif change == "missing":
        source = source.replace(
            "x = torch.linspace(-0.5, 0.5, width, device=device, dtype=dtype)", "x = forward.dx"
        )
    else:
        source += "\n    extra = torch.linspace(-0.5, 0.5, 16)\n"
    with pytest.raises(ValueError, match="exactly the two frozen"):
        rewrite_summary_linspaces(source)


def test_private_candidate_leaves_scientific_globals_and_torch_linspace_intact() -> None:
    original_globals = dict(vars(local_transport))
    original_linspace = torch.linspace
    original_summary = local_transport._summarise_one
    private, binding = make_native_linspace_physical()
    assert private.__module__ == binding["private_module"]
    assert private.__globals__ is vars(sys.modules[private.__module__])
    assert private.__globals__ is not vars(local_transport)
    assert private.__globals__["_summarise_one"] is not original_summary
    assert private.__globals__["_weighted_slope"].__globals__ is private.__globals__
    assert private.__globals__["_NATIVE_SUMMARY_LINSPACE"] is native_summary_linspace_operator()
    assert binding["replacements"] == 2
    assert binding["scientific_globals_modified"] is False
    assert len(binding["original_summary_sha256"]) == 64
    assert len(binding["rewritten_ast_sha256"]) == 64
    assert vars(local_transport) == original_globals
    assert torch.linspace is original_linspace


@pytest.mark.parametrize("size,radius,deterministic", [(16, 1, False), (32, 2, True)])
def test_cpu_partition_preserves_controls_and_candidate_for_changed_and_zero_inputs(
    size: int, radius: int, deterministic: bool
) -> None:
    generator = torch.Generator().manual_seed(263)
    inputs = (
        torch.randn(2, 4, size, size, generator=generator),
        torch.randn(2, 4, size, size, generator=generator),
        torch.rand(2, 1, size, size, generator=generator),
        torch.rand(2, 1, size, size, generator=generator),
    )
    changed = (inputs[0].flip(-1), inputs[1] * 0.6, inputs[2] * 0.7, inputs[3].flip(-2))
    zero = tuple(torch.zeros_like(value) for value in inputs)
    probes = make_linspace_probe(
        radius, 0.02, deterministic_sampling=deterministic, backend="eager"
    )
    assert set(probes) == {"original", "native_weighted_mean", CANDIDATE}
    with torch.inference_mode():
        for values in (inputs, changed, zero):
            expected = probes["original"](*values)
            for name in ("native_weighted_mean", CANDIDATE):
                actual = probes[name](*values)
                assert len(actual) == len(expected) == 2
                assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)), name


def test_standalone_cpu_coordinates_report_exact_values_and_bits() -> None:
    records, tensors = coordinate_probe(device="cpu", backend="eager")
    assert [row["steps"] for row in records] == [16, 32]
    for row in records:
        for name in ("compiled", "opaque"):
            comparison = row["variants"][name]
            assert comparison["exact"]
            assert comparison["different_indices"] == []
            assert comparison["max_abs"] == 0
            assert comparison["bits"] == comparison["original_bits"]
            assert torch.equal(
                tensors[f"{name}_{row['steps']}"], tensors[f"original_{row['steps']}"]
            )


def _cohort() -> list[dict[str, Any]]:
    return [
        {"index": index, "producer": producer, "scale": scale, "status": "COMPARED"}
        for index in (488, 505)
        for producer, scale in (
            ("E_A5_MATCHED", "fine"),
            ("E_C2F_MATCHED", "fine"),
            ("E_C2F_MATCHED", "coarse"),
            ("R_A5", "fine"),
            ("R_C2F", "fine"),
            ("R_C2F", "coarse"),
        )
    ]


def test_cohort_requires_the_entire_fixed_population() -> None:
    rows = _cohort()
    assert validate_cohort(rows) is rows


@pytest.mark.parametrize("damage", ["missing", "duplicate", "unexpected", "failed"])
def test_cohort_rejects_partial_or_changed_evidence(damage: str) -> None:
    rows = _cohort()
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows[-1] = rows[0]
    elif damage == "unexpected":
        rows[-1]["index"] = 506
    else:
        rows[-1]["status"] = "FAILED"
    with pytest.raises(ValueError):
        validate_cohort(rows)


def _passing_evidence() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from scripts.diagnose_train40_transport_summaries import compare_transport_vectors

    vector = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    exact = compare_transport_vectors((vector, vector), (vector, vector))
    rows = _cohort()
    for row in rows:
        row["variants"] = {"original": deepcopy(exact), CANDIDATE: deepcopy(exact)}
        row["control_reproduction"] = {
            "original": deepcopy(exact),
            "native_weighted_mean": deepcopy(exact),
        }
    coordinates = [{"steps": steps, "variants": {"opaque": {"exact": True}}} for steps in (16, 32)]
    return rows, coordinates


def test_gate_requires_all_controls_candidates_and_standalone_coordinates() -> None:
    rows, coordinates = _passing_evidence()
    assert all(diagnostic_gate(rows, coordinates).values())


@pytest.mark.parametrize(
    "damage",
    [
        "missing_control",
        "failed_control",
        "missing_feature",
        "failed_feature",
        "nonfinite",
        "partial_cohort",
        "missing_standalone",
    ],
)
def test_gate_rejects_incomplete_or_negative_evidence_despite_aggregate_flags(damage: str) -> None:
    rows, coordinates = _passing_evidence()
    if damage == "missing_control":
        rows[-1]["control_reproduction"].pop("native_weighted_mean")
    elif damage == "failed_control":
        rows[-1]["control_reproduction"]["native_weighted_mean"]["exact"] = False
    elif damage in {"missing_feature", "failed_feature", "nonfinite"}:
        features = rows[-1]["variants"][CANDIDATE]["outputs"][0]["coordinates"]
        if damage == "missing_feature":
            features.pop()
        elif damage == "failed_feature":
            features[0]["exact"] = False
        else:
            features[0]["nonfinite_pairs"] = 1
    elif damage == "partial_cohort":
        rows.pop()
    else:
        coordinates.pop()
    assert not diagnostic_gate(rows, coordinates)["diagnostic_gate_passed"]
