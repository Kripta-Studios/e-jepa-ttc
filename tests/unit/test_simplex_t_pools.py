"""Input-order invariance, protected groups and pre-score availability decisions."""

import pytest

from e_jepa_ttc.simplex_t.pools import (
    QueryIdentity,
    balanced_queries,
    canonical_data,
    density_size,
    expanded_pool,
    expansion_producer,
    matched_density_pools,
)


def candidates(groups=6):
    return [
        QueryIdentity(f"g{g}-t{t}-q{q}", f"s{g}", f"t{t}", f"g{g}")
        for g in range(groups)
        for t in range(2)
        for q in range(3)
    ]


def test_balancing_is_input_order_invariant_and_group_then_track():
    rows = candidates()
    allowed = {row.acquisition_group for row in rows}
    selected = balanced_queries(rows, limit=12, allowed_groups=allowed)
    assert selected == balanced_queries(rows[::-1], limit=12, allowed_groups=allowed)
    assert [row.acquisition_group for row in selected] == [f"g{i}" for i in range(6)] * 2
    assert [row.track for row in selected] == ["t0"] * 6 + ["t1"] * 6


def test_expansion_preserves_original_and_requires_six_actual_groups():
    original = (QueryIdentity("original", "old", "track", "old"),)
    rows = candidates()
    available = expanded_pool(
        original,
        rows,
        original_train_groups={"old"},
        approved_expansion_groups={f"g{i}" for i in range(6)},
    )
    assert available.available and available.queries[:1] == original
    missing = expanded_pool(
        original,
        candidates(5),
        original_train_groups={"old"},
        approved_expansion_groups={f"g{i}" for i in range(6)},
    )
    assert not missing.available and missing.queries == original


def test_unknown_group_is_rejected_not_silently_filtered():
    with pytest.raises(ValueError, match="approved"):
        balanced_queries(candidates(), limit=5, allowed_groups={"g0"})


def test_duplicate_query_identity_is_not_extra_density():
    row = candidates()[0]
    with pytest.raises(ValueError, match="duplicate"):
        balanced_queries([row, row], limit=2, allowed_groups={row.acquisition_group})


def test_canonical_data_and_matched_density_are_fixed_by_availability():
    assert canonical_data((True, True, True)) == "D1"
    assert canonical_data((True, False, True)) == "D0"
    assert density_size(d0_count=5000, dense_old_available=12000, diverse_available=20000) == 12000
    assert density_size(d0_count=5000, dense_old_available=9999, diverse_available=20000) is None


def test_one_family_for_whole_sequence():
    families = ("a" * 64, "b" * 64, "c" * 64)
    assert expansion_producer("sequence", families) in families
    assert len({expansion_producer("sequence", families) for _ in range(16)}) == 1


def test_matched_pools_are_equal_unique_and_input_order_invariant():
    original = (QueryIdentity("old0", "old", "t", "old"),)
    dense = [*original, QueryIdentity("old1", "old", "t", "old")]
    diverse = [*original, *candidates()]
    kwargs = {"original_groups": {"old"}, "expansion_groups": {f"g{i}" for i in range(6)}}
    result = matched_density_pools(original, dense, diverse, **kwargs)
    assert result is not None
    assert len(result[0]) == len(result[1]) == 2
    assert result == matched_density_pools(original, dense[::-1], diverse[::-1], **kwargs)
    assert {row.acquisition_group for row in result[0]} == {"old"}


def test_matched_pools_validate_identity_even_below_count_gate():
    original = (QueryIdentity("old0", "old", "t", "old"),)
    assert (
        matched_density_pools(
            original,
            list(original),
            list(original),
            original_groups={"old"},
            expansion_groups=set(),
        )
        is None
    )
    with pytest.raises(ValueError, match="preserve"):
        matched_density_pools(
            original, [], list(original), original_groups={"old"}, expansion_groups=set()
        )
