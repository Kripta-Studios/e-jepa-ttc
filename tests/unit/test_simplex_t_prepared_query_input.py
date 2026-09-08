"""Bounded input reuse tests; no experts or optimizer updates."""

import pytest
import torch

from e_jepa_ttc.simplex_t.prepared_query_input import PreparedQueryInput


def test_exact_reuse_discards_previous_entry_before_next_preparation():
    cache = PreparedQueryInput(max_bytes=16)
    value = torch.arange(4, dtype=torch.float32)
    first = cache.get((b"input-a",), lambda: value)
    second = cache.get((b"input-a",), lambda: pytest.fail("duplicate preparation"))
    assert first is second
    assert torch.equal(second, value)

    def next_input():
        assert cache.tensor is None
        return value + 1

    third = cache.get((b"input-b",), next_input)
    assert torch.equal(third, value + 1)
    assert (cache.preparations, cache.hits) == (2, 1)
    cache.clear()
    assert cache.tensor is None and cache.key is None


@pytest.mark.parametrize("tensor", [torch.zeros(5), torch.zeros(4, dtype=torch.float64)])
def test_refuses_oversize_or_wrong_precision_without_retaining_input(tensor):
    cache = PreparedQueryInput(max_bytes=16)
    with pytest.raises(ValueError):
        cache.get((b"input",), lambda: tensor)
    assert cache.tensor is None
    assert cache.preparations == 0
