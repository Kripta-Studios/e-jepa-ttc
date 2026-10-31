"""CPU fallback applies to every GPU abstention, independently of labels or q."""

from unittest.mock import Mock

import numpy as np
import pytest

from operational.phase_event_expansion.hybrid import estimate_hybrid
from operational.phase_event_expansion.model import ExpansionResult, SolverConfig


@pytest.mark.parametrize("q,b", [(None, None), (0.25, (0.1, -0.2))])
def test_every_invalid_gpu_attempt_returns_exact_cpu_result(q, b):
    gpu = ExpansionResult(q, b, None, False, "flat_objective", 0.1, {"gpu": True})
    cpu = ExpansionResult(0.251, (0.2, -0.1), None, False, "flat_objective", 1.0, {"cpu": True})
    gpu_fn, cpu_fn = Mock(return_value=gpu), Mock(return_value=cpu)
    events, roi, cache = np.zeros((0, 4)), np.array([0, 0, 1, 1]), object()
    config = SolverConfig(objective="temporal_ncc", reference="endpoint")
    output = estimate_hybrid(
        events, roi, 1000, config, cache, gpu_estimator=gpu_fn, cpu_estimator=cpu_fn
    )
    assert output.result is cpu
    assert output.gpu_attempt is gpu
    assert output.fallback_used
    assert output.cpu_wall_seconds >= 0
    assert output.total_wall_seconds >= output.cpu_wall_seconds
    gpu_fn.assert_called_once_with(events, roi, 1000, config, cache)
    cpu_fn.assert_called_once_with(events, roi, 1000, config)


def test_valid_gpu_result_never_invokes_cpu():
    gpu = ExpansionResult(2.0, (0.1, 0.2), 0.5, True, "available", 0.1, {})
    cpu_fn = Mock(side_effect=AssertionError("Unexpected CPU inference"))
    output = estimate_hybrid(
        np.zeros((0, 4)),
        [0, 0, 1, 1],
        1000,
        gpu_estimator=Mock(return_value=gpu),
        cpu_estimator=cpu_fn,
    )
    assert output.result is gpu and output.gpu_attempt is gpu
    assert not output.fallback_used and output.cpu_wall_seconds == 0
    cpu_fn.assert_not_called()
