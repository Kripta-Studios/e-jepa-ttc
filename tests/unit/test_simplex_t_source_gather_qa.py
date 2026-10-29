"""Full gather validation uses fixtures only, without a head or optimizer."""

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.cache import CachedQueries, fit_normalizer
from e_jepa_ttc.simplex_t.source_gather_qa import audit_cached_source


@pytest.mark.parametrize(
    "control,zero",
    [("NONE", False), ("PAST_REVERSED", False), ("REPEAT_CURRENT", False), ("NONE", True)],
)
def test_gather_qa_covers_every_query_and_control(control, zero):
    features = np.zeros((16, 145), dtype=np.float32)
    history = np.tile(np.arange(16, dtype=np.int64), (129, 1))
    times = np.arange(16, dtype=np.int64) * 50000
    source = CachedQueries(
        features,
        times,
        times,
        history,
        np.ones(129),
        np.full(129, 1 / 129),
        fit_normalizer(features, history, np.ones(16, dtype=bool)),
        "a" * 64,
        8,
        control,
        zero,
    )
    boundaries = []
    result = audit_cached_source(source, boundary=lambda: boundaries.append(True))
    assert result["queries"] == 129 and result["valid_slots"] == 129 * 8
    assert result["source_sha256"] == "a" * 64
    assert len(boundaries) == 3 and result["optimizer_updates"] == 0
