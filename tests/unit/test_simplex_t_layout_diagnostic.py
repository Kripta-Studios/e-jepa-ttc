"""Layout diagnostics preserve differences and do not invent pass tolerances."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def difference(left, right):
    path = (
        Path(__file__).resolve().parents[2] / "scripts/diagnose_simplex_t_cached_current_layout.py"
    )
    spec = importlib.util.spec_from_file_location("layout_diagnostic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.difference(left, right)


def test_changed_infinite_values_are_not_hidden_by_finite_only_delta():
    result = difference(
        np.array([1, np.inf, -np.inf], np.float32), np.array([1, 2, np.inf], np.float32)
    )
    assert result == {"changed_values": 2, "finite_pattern_changes": 1, "max_abs_joint_finite": 0.0}


def test_small_differences_remain_observed_differences():
    a = np.array([1], np.float32)
    result = difference(a, np.nextafter(a, np.float32(2)))
    assert result["changed_values"] == 1
    assert result["max_abs_joint_finite"] > 0


@pytest.mark.parametrize("other", [np.ones(2, np.float32), np.ones(1, np.float64)])
def test_shape_and_dtype_mismatch_are_rejected(other):
    with pytest.raises(ValueError, match="schemas"):
        difference(np.ones(1, np.float32), other)
