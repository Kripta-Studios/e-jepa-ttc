"""Historical V4 JEPA adapter must not treat the token-order label as token IDs."""

from __future__ import annotations

import pytest

from e_jepa_ttc.data.canonical_token_identity import hash_ordered_token_ids
from e_jepa_ttc.data.scientific_recovery_v8_jepa_data import (
    LEXICOGRAPHIC_TOKEN_ORDER,
    resolve_historical_v4_token_order,
)


def _identity_tokens(tokens: list[str], order: object) -> tuple[str, ...]:
    if order == LEXICOGRAPHIC_TOKEN_ORDER:
        return tuple(sorted(tokens))
    if isinstance(order, (list, tuple)):
        return tuple(str(item) for item in order)
    raise TypeError("unsupported token_order in test fixture")


def _contract(tokens: list[str], *, order: object = LEXICOGRAPHIC_TOKEN_ORDER) -> dict[str, object]:
    return {
        "token_order": order,
        "rows": len(tokens),
        "ordered_token_ids_sha256": hash_ordered_token_ids(_identity_tokens(tokens, order)),
    }


def test_lexicographic_label_is_not_iterated_as_token_ids() -> None:
    tokens = ["seq_b_2", "seq_a_1", "seq_c_3"]
    ordered = resolve_historical_v4_token_order(
        sample_contract=_contract(tokens),
        cache_tokens=list(tokens),
    )
    assert ordered == ("seq_a_1", "seq_b_2", "seq_c_3")
    assert set(ordered) == set(tokens)


def test_iterating_the_order_label_would_have_rejected_a_valid_cache() -> None:
    tokens = ["seq_b_2", "seq_a_1", "seq_c_3"]
    bogus = tuple(str(character) for character in LEXICOGRAPHIC_TOKEN_ORDER)
    assert set(tokens).isdisjoint(bogus)
    resolve_historical_v4_token_order(
        sample_contract=_contract(tokens),
        cache_tokens=list(tokens),
    )


def test_explicit_token_list_must_match_cache_universe() -> None:
    tokens = ["a", "b", "c"]
    with pytest.raises(ValueError, match="does not match the frozen V8 token universe"):
        resolve_historical_v4_token_order(
            sample_contract=_contract(tokens, order=["a", "b", "missing"]),
            cache_tokens=list(tokens),
        )


def test_hash_mismatch_is_fatal() -> None:
    tokens = ["a", "b", "c"]
    contract = _contract(tokens)
    contract["ordered_token_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ordered_token_ids_sha256"):
        resolve_historical_v4_token_order(sample_contract=contract, cache_tokens=list(tokens))
