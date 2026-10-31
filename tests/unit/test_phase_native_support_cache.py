"""Native support is exact clock/count metadata, never reconstructed hidden state."""

import numpy as np
import pytest

from operational.phase_native_support.features import (
    INDICES,
    from_context_arrays,
    from_phase340,
    validate_support,
)
from scripts.build_phase_native_support_cache import exact_indices, read_fields


def phase():
    value = np.zeros((2, 340), np.float32)
    value[:, INDICES] = [0.04, 0.01, 0.1, 0.03, 0.625, 0.02, 0, 0.05, 0.02, 0.875]
    return value


def test_exact_coordinates_order_and_no_alias():
    source = phase()
    quality, known = from_phase340(source)
    np.testing.assert_array_equal(quality, source[:, INDICES])
    assert known.shape == (2, 2) and known.dtype == bool and known.all()
    quality[0, 0] = -1
    assert source[0, 165] == np.float32(0.04)


def test_unexecuted_contexts_have_explicit_zero_placeholders():
    quality, known = from_context_arrays({})
    assert quality.shape == (1, 10) and not quality.any()
    assert not known.any()


def test_event_only_uses_executed_context_clocks_and_mask():
    timing = np.zeros((1, 8, 4), np.float32)
    timing[0, -1] = [0.02, 0, 0, 0.02]
    valid = np.zeros((1, 8), bool)
    valid[0, -1] = True
    quality, known = from_context_arrays(
        {"E_CTX_MATCHED.input_1": timing, "E_CTX_MATCHED.input_2": valid}
    )
    np.testing.assert_array_equal(known, [[False, True]])
    np.testing.assert_array_equal(quality[0, 5:], [np.float32(0.02), 0, 0, np.float32(0.02), 0.125])
    assert not quality[0, :5].any()


@pytest.mark.parametrize(
    "fault", ["nan", "count", "identity", "negative", "single_interval", "unknown_payload"]
)
def test_support_rejects_invalid_clock_count_contract(fault):
    quality, known = from_phase340(phase())
    if fault == "nan":
        quality[0, 0] = np.nan
    elif fault == "count":
        quality[0, 4] = 0.2
    elif fault == "identity":
        quality[0, 0] = 0.3
    elif fault == "negative":
        quality[0, 2] = -0.1
    elif fault == "single_interval":
        quality[0, 4] = 0.125
    else:
        known[0, 0] = False
    with pytest.raises(ValueError):
        validate_support(quality, known)


def test_token_join_rejects_duplicates_and_missing_identity():
    np.testing.assert_array_equal(exact_indices(np.array(["a", "b"]), np.array(["b", "a"])), [1, 0])
    with pytest.raises(ValueError):
        exact_indices(np.array(["a", "a"]), np.array(["a"]))
    with pytest.raises(ValueError):
        exact_indices(np.array(["a"]), np.array(["missing"]))


def test_field_reader_rejects_target_access_before_opening_file():
    with pytest.raises(ValueError):
        read_fields("does-not-exist.npz", ["target"])
