"""No target-based membership or accidental cross-producer cache reuse."""

from dataclasses import replace

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.context_dedup import ContextSource, deduplicate_contexts
from e_jepa_ttc.simplex_t.query_context import QueryContextInput


def source():
    return ContextSource(
        "sequence",
        "a" * 64,
        "b" * 64,
        QueryContextInput(
            "query",
            "c" * 64,
            1_000_000,
            1_020_000,
            1_020_000,
            0,
            ((750_000, 800_000), (850_000, 900_000), (950_000, 1_000_000)),
            (0.0, 0.0, 100.0, 100.0),
        ),
    )


def test_duplicate_query_names_do_not_duplicate_consumed_inputs():
    first = source()
    second = replace(first, current=replace(first.current, query_token="another"))
    keys, history = deduplicate_contexts([first, second])
    assert len(keys) == 16
    np.testing.assert_array_equal(history[0], history[1])


@pytest.mark.parametrize("field", ["producer_family_sha256", "roi_available_us", "square_xyxy"])
def test_distinct_dependencies_do_not_alias(field):
    first = source()
    values = {
        "producer_family_sha256": "d" * 64,
        "roi_available_us": 1_010_000,
        "square_xyxy": (1.0, 0.0, 101.0, 100.0),
    }
    second = replace(first, current=replace(first.current, **{field: values[field]}))
    keys, history = deduplicate_contexts([first, second])
    assert len(keys) == 32
    assert not set(history[0]) & set(history[1])


def test_cold_start_is_masked_without_repeating_current():
    first = source()
    keys, history = deduplicate_contexts(
        [replace(first, current=replace(first.current, source_start_us=700_000))]
    )
    assert len(keys) == 2
    assert history.tolist() == [[-1] * 14 + [0, 1]]


def test_ttc_perturbation_does_not_change_input_index():
    fixture = {"input": source(), "ttc": 0.5, "velocity": 100.0}
    before = deduplicate_contexts([fixture["input"]])
    fixture.update(ttc=-999.0, velocity=-500.0)
    after = deduplicate_contexts([fixture["input"]])
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
