"""The source audit counts the actual emission floor instead of exact decimal TTC."""

import numpy as np
import torch

from scripts.audit_phase_canonical_pair_source import output_policy


def test_output_floor_uses_inverse_magnitude_and_preserves_signed_zero():
    raw = np.array([0.0, -0.0, 0.0001, -0.0001, 0.1, -0.1], np.float32)
    result = output_policy(raw)
    assert result["inverse_magnitude_floor_rows"] == 4
    assert result["positive_floor"] == 3
    assert result["negative_floor"] == 1
    assert result["raw_zero_phase"] == 2
    assert result["raw_negative_zero_phase"] == 1
    assert torch.tensor(1 / 60, dtype=torch.float32).reciprocal().item() != 60


def test_output_policy_records_invalid_rows_separately():
    result = output_policy(np.array([np.nan, np.inf, 0.1], np.float32))
    assert result["nonfinite_raw_phase"] == 2
    assert result["finite_rows"] == 1
    assert result["inverse_magnitude_floor_rows"] == 0
