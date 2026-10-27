from __future__ import annotations

import pytest
import torch

from operational.final_refit.optimization_probe import (
    SCHEMA,
    _canonical_sha256,
    compare_values,
    validate_receipt,
)


def test_compare_values_requires_names_shapes_and_finite_values() -> None:
    valid = compare_values({"x": torch.tensor([1.0])}, {"x": torch.tensor([1.0 + 1e-6])})
    assert valid["passed"] is True
    assert compare_values({"x": torch.ones(1)}, {"y": torch.ones(1)})["passed"] is False
    assert compare_values(torch.ones(1), torch.ones(2))["passed"] is False
    assert compare_values(torch.tensor([float("nan")]), torch.ones(1))["passed"] is False


def test_compare_values_exact_mode_rejects_tolerated_float_delta() -> None:
    left = torch.tensor([1.0])
    right = torch.tensor([1.0 + 1e-6])
    assert compare_values(left, right)["passed"] is True
    assert compare_values(left, right, exact=True)["passed"] is False


def test_compare_values_normalizes_numpy_rng_arrays() -> None:
    state = torch.arange(4).numpy()
    assert compare_values({"numpy": state}, {"numpy": state.copy()}, exact=True)["passed"] is True


def _passing_receipt() -> dict[str, object]:
    approximate = {"passed": True, "finite": True, "exact": False}
    exact = {"passed": True, "finite": True, "exact": True}
    cases = []
    for fit_id in ("R_A5", "R_C2F"):
        for epoch in (2, 4):
            for signature in ("T2", "T3", "MIXED", "MIXED_B32"):
                steps = [
                    {
                        "numerical": {"loss": dict(approximate)},
                        "state": {"model": dict(approximate)},
                        "rng": dict(exact),
                        "passed": True,
                    }
                    for _ in range(2)
                ]
                resume_steps = [
                    {"fields": {"model": dict(exact)}, "passed": True} for _ in range(2)
                ]
                comparison = {
                    "steps": steps,
                    "host_optimization_available": True,
                    "scalar_optimization_available": True,
                    "resume_parity": {"steps": resume_steps, "passed": True},
                    "passed": True,
                }
                cases.append(
                    {
                        "fit_id": fit_id,
                        "epoch": epoch,
                        "signature": signature,
                        "comparison": comparison,
                        "passed": True,
                    }
                )
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "status": "PASS",
        "scientific_optimizer_updates": 0,
        "technical_optimizer_updates": 96,
        "admitted_options": {
            "shared_resource_snapshot": False,
            "host_loss_metadata": True,
            "transport_reuse": True,
            "scalar_transfer": True,
            "merge_parts_by_t": False,
            "cudagraph_local_transport": False,
        },
        "configured_options": {
            "host_loss_metadata": True,
            "transport_reuse": True,
        },
        "checks": {
            "numerical_parity": "PASS",
            "state_parity": "PASS",
            "resume_parity": "PASS",
            "rng_parity": "PASS",
        },
        "population_physical_mask_parity": {
            "population_size": 88_744,
            "mask_exact": True,
            "passed": True,
            "target_comparison": {"finite": True, "passed": True},
        },
        "cases": cases,
        "source_stability": {"passed": True},
        "source_sha256_before_after": {
            "operational/final_refit/producers.py": {
                "before_sha256": "a" * 64,
                "after_sha256": "b" * 64,
            }
        },
    }
    receipt["admission_sha256"] = _canonical_sha256(receipt)
    return receipt


def test_validate_receipt_recomputes_case_evidence() -> None:
    receipt = _passing_receipt()
    validate_receipt(receipt)
    cases = receipt["cases"]
    assert isinstance(cases, list)
    cases[0]["comparison"]["steps"][0]["rng"]["finite"] = False
    receipt["admission_sha256"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "admission_sha256"}
    )
    with pytest.raises(ValueError, match="pass flag"):
        validate_receipt(receipt)


def test_validate_receipt_accepts_explicit_transport_rejection() -> None:
    receipt = _passing_receipt()
    receipt["admitted_options"]["transport_reuse"] = False
    receipt["configured_options"]["transport_reuse"] = False
    receipt["admission_sha256"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "admission_sha256"}
    )
    validate_receipt(receipt)


def test_validate_receipt_rejects_extra_check_even_with_valid_hash() -> None:
    receipt = _passing_receipt()
    checks = receipt["checks"]
    assert isinstance(checks, dict)
    checks["source_parity"] = "PASS"
    receipt["admission_sha256"] = _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "admission_sha256"}
    )
    with pytest.raises(ValueError, match="checks"):
        validate_receipt(receipt)
