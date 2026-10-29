"""Reference-equivalence tests for consolidated finite validation."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from e_jepa_ttc.detection.finite_validation import (
    require_finite_fused,
    require_finite_reference,
)


def outcome(function: Callable[..., None], value: object, name: str = "output") -> object:
    try:
        function(value, name)
    except Exception as error:  # noqa: BLE001 - exact public failure equivalence is asserted.
        return type(error), str(error)
    return None


@pytest.mark.parametrize(
    "value",
    [
        {},
        [],
        (),
        {"empty": [(), {}], "scalar": 3},
        torch.tensor([], dtype=torch.float32),
        {
            "mixed": [
                torch.tensor([1, 2], dtype=torch.int64),
                torch.tensor([True, False]),
                torch.tensor([1.0, -2.0], dtype=torch.float16),
                torch.tensor([1.0], dtype=torch.float64),
            ]
        },
    ],
)
def test_success_matches_historical_reference(value: object) -> None:
    assert outcome(require_finite_fused, value) == outcome(require_finite_reference, value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_failure_matches_first_named_offender(bad: float) -> None:
    value = {
        "good": torch.tensor([1.0]),
        "nested": (
            [torch.tensor([2.0]), torch.tensor([bad])],
            {"later": torch.tensor([float("nan")])},
        ),
    }
    expected = (ValueError, "Nonfinite tensor: root.nested[0][1]")
    assert outcome(require_finite_reference, value, "root") == expected
    assert outcome(require_finite_fused, value, "root") == expected


def test_repeated_tensor_preserves_first_container_path() -> None:
    repeated = torch.tensor([float("nan")])
    value = {"first": [repeated], "second": repeated}
    expected = (ValueError, "Nonfinite tensor: batch.first[0]")
    assert outcome(require_finite_reference, value, "batch") == expected
    assert outcome(require_finite_fused, value, "batch") == expected


def test_one_host_decision_for_many_cpu_tensor_leaves() -> None:
    decisions: list[tuple[torch.device, bool]] = []

    def decide(flag: Tensor) -> bool:
        result = bool(flag.item())
        decisions.append((flag.device, result))
        return result

    require_finite_fused(
        {
            "a": torch.ones(4),
            "b": [torch.zeros(2, dtype=torch.float64), torch.tensor([1], dtype=torch.int32)],
        },
        host_decision=decide,
    )
    assert decisions == [(torch.device("cpu"), True)]


def test_empty_structure_makes_no_host_decision() -> None:
    def unexpected(_: Tensor) -> bool:
        raise AssertionError("empty structure must not synchronize")

    require_finite_fused({"empty": [(), {}]}, host_decision=unexpected)


def test_failure_uses_one_fused_decision_then_reference_slow_path() -> None:
    decisions = 0

    def decide(flag: Tensor) -> bool:
        nonlocal decisions
        decisions += 1
        return bool(flag.item())

    value = [torch.ones(2), torch.tensor([float("nan")]), torch.tensor([float("inf")])]
    with pytest.raises(ValueError, match=r"Nonfinite tensor: output\[1\]"):
        require_finite_fused(value, host_decision=decide)
    assert decisions == 1


def test_later_meta_tensor_cannot_replace_first_historical_nan_error() -> None:
    value = [torch.tensor([float("nan")]), torch.empty(1, device="meta")]
    expected = (ValueError, "Nonfinite tensor: root[0]")
    assert outcome(require_finite_reference, value, "root") == expected
    assert outcome(require_finite_fused, value, "root") == expected


def test_unsupported_sparse_layout_delegates_to_reference() -> None:
    with pytest.warns(UserWarning, match="Sparse invariant checks"):
        sparse = torch.sparse_coo_tensor(
            indices=torch.tensor([[0]]),
            values=torch.tensor([1.0]),
            size=(1,),
            check_invariants=False,
        )
    expected = outcome(require_finite_reference, [sparse])
    assert outcome(require_finite_fused, [sparse]) == expected
