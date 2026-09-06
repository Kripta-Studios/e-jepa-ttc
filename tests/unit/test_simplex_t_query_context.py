"""Contract tests for the separately disclosed query-conditioned alternative."""

from dataclasses import replace

import pytest

from e_jepa_ttc.simplex_t.query_context import QueryContextInput, query_context


def current():
    return QueryContextInput(
        "query",
        "a" * 64,
        1_000_000,
        1_020_000,
        1_020_000,
        0,
        ((750_000, 800_000), (850_000, 900_000), (950_000, 1_000_000)),
        (0.0, 0.0, 100.0, 100.0),
    )


def test_current_roi_availability_is_not_backdated():
    rows = query_context(current())
    assert len(rows) == 8
    assert [row.lag_us for row in rows] == list(range(350_000, -1, -50_000))
    assert all(row.available_us == 1_020_000 for row in rows)
    assert rows[0].sensor_available_us == 650_000
    assert all(row.input.square_xyxy == current().square_xyxy for row in rows)


def test_h1_and_cold_start_keep_current_without_fabricating_past():
    assert len(query_context(current(), 1)) == 1
    rows = query_context(replace(current(), source_start_us=700_000))
    assert [row.lag_us for row in rows] == [50_000, 0]


def test_no_target_or_past_query_table_in_input_contract():
    with pytest.raises(TypeError):
        QueryContextInput(**{**current().__dict__, "ttc": 1.0})
    with pytest.raises(ValueError, match="cutoff"):
        query_context(replace(current(), roi_available_us=1_020_001))
